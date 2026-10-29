"""Player-week, player-season, and player-static tables at an as-of date.

Everything here is built from an `AsOfSnapshot`, never from raw frames, so
the point-in-time guarantees of `features.asof` carry through.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import polars as pl

from dwar.config import ScoringConfig, data_dir
from dwar.features.asof import PLAY_COLUMNS, AsOfSnapshot
from dwar.features.scoring import fantasy_points

SKILL_POSITIONS = ("QB", "RB", "WR", "TE")
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class PlayerTables:
    as_of: date
    player_weeks: pl.DataFrame
    player_seasons: pl.DataFrame
    player_static: pl.DataFrame

    def frames(self) -> dict[str, pl.DataFrame]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "as_of"}


def _age(on: pl.Expr, birth: pl.Expr) -> pl.Expr:
    return ((on - birth).dt.total_days() / DAYS_PER_YEAR).round(3)


def _season_positions(snap: AsOfSnapshot, pairs: pl.DataFrame) -> pl.DataFrame:
    """Position for each (player_id, season): that season's roster row, else the
    nearest knowable roster season, else draft position. nflverse's
    player_stats `position` is today's position back-filled, so it is not used."""
    rosters = snap.rosters.select("player_id", "season", "position").sort("season")
    pairs = pairs.unique().sort("season")
    back = pairs.join_asof(rosters, on="season", by="player_id", strategy="backward", check_sortedness=False)
    fwd = pairs.join_asof(rosters, on="season", by="player_id", strategy="forward", check_sortedness=False)
    return (
        back.join(fwd, on=["player_id", "season"], suffix="_fwd")
        .join(snap.draft.select("player_id", "draft_position"), on="player_id", how="left")
        .select("player_id", "season", position=pl.coalesce("position", "position_fwd", "draft_position"))
    )


def player_weeks(snap: AsOfSnapshot, scoring: ScoringConfig) -> pl.DataFrame:
    """One row per skill player per regular-season game played.

    `plays` counts the player's own pass attempts, sacks taken, and
    carries, a proxy for snaps that exists for every season.

    A game played is a row in player_stats, or (2013 on) a game with offense
    snaps but no stat row, which scores zero. Without the latter, a receiver
    who played without a target would drop out and inflate points per game.
    """
    scored = fantasy_points(snap.stats, scoring).select(
        "player_id", "season", "week", "game_id", "gameday", "team", "opponent_team",
        "fpts", "knowable_on", source=pl.lit("stats"),
        # Null when the stats lack play counts (small test fixtures), so a
        # play filter fails loudly instead of seeing zeros.
        plays=pl.sum_horizontal([pl.col(c).fill_null(0) for c in PLAY_COLUMNS]).cast(pl.Int64)
        if set(PLAY_COLUMNS) <= set(snap.stats.columns) else pl.lit(None, pl.Int64),
    )
    snap_only = (
        snap.snaps.filter(pl.col("offense_snaps") > 0)
        .join(scored.select("player_id", "game_id"), on=["player_id", "game_id"], how="anti")
        .select(
            "player_id", "season", "week", "game_id", "gameday", "team", "opponent_team",
            fpts=pl.lit(0.0), knowable_on="knowable_on", source=pl.lit("snaps_only"), plays=pl.lit(0, pl.Int64),
        )
    )
    weeks = pl.concat([scored, snap_only], how="vertical_relaxed")
    positions = _season_positions(snap, weeks.select("player_id", "season"))
    return (
        weeks.join(positions, on=["player_id", "season"], how="left")
        .filter(pl.col("position").is_in(SKILL_POSITIONS))
        .sort("player_id", "gameday")
    )


def player_seasons(snap: AsOfSnapshot, weeks: pl.DataFrame) -> pl.DataFrame:
    season_end = snap.schedules.group_by("season").agg(last_gameday=pl.col("gameday").max())
    return (
        weeks.sort("gameday")
        .group_by("player_id", "season", maintain_order=True)
        .agg(
            position=pl.col("position").first(),
            games_played=pl.len(),
            fpts_total=pl.col("fpts").sum(),
            team_last=pl.col("team").last(),
            knowable_on=pl.col("knowable_on").max(),
        )
        .with_columns(fpts_per_game=pl.col("fpts_total") / pl.col("games_played"))
        .join(season_end, on="season", how="left")
        .with_columns(season_complete=pl.col("last_gameday").lt(pl.lit(snap.as_of)).fill_null(False))
        .join(snap.bio.select("player_id", "birth_date"), on="player_id", how="left")
        .with_columns(age_sept1=_age(pl.date(pl.col("season"), 9, 1), pl.col("birth_date")))
        .drop("last_gameday", "birth_date")
        .sort("player_id", "season")
    )


def _height_inches(ht: pl.Expr) -> pl.Expr:
    parts = ht.str.split("-")
    return parts.list.get(0, null_on_oob=True).cast(pl.Int32, strict=False) * 12 + parts.list.get(
        1, null_on_oob=True
    ).cast(pl.Int32, strict=False)


def player_static(snap: AsOfSnapshot, weeks: pl.DataFrame) -> pl.DataFrame:
    """One row per skill player known at `as_of`: anyone with a game played,
    plus anyone drafted, both strictly before `as_of`."""
    career = (
        weeks.sort("gameday")
        .group_by("player_id", maintain_order=True)
        .agg(
            position_last=pl.col("position").last(),
            team_last=pl.col("team").last(),
            first_season=pl.col("season").min(),
            last_season_played=pl.col("season").max(),
            last_gameday=pl.col("gameday").max(),
            career_games=pl.len(),
            weeks_knowable_on=pl.col("knowable_on").max(),
        )
    )
    universe = pl.concat(
        [career.select("player_id"), snap.draft.select("player_id")]
    ).unique()
    combine = snap.combine.select(
        "player_id", *[c for c in snap.combine.columns if c not in {"player_id", "draft_year", "ht", "knowable_on"}],
        ht_in=_height_inches(pl.col("ht")), combine_knowable_on="knowable_on",
    )
    return (
        universe.join(career, on="player_id", how="left")
        .join(snap.draft.rename({"knowable_on": "draft_knowable_on"}), on="player_id", how="left")
        .join(combine, on="player_id", how="left")
        .join(snap.bio.drop("knowable_on"), on="player_id", how="left")
        .with_columns(
            position=pl.coalesce("position_last", "draft_position"),
            team=pl.coalesce("team_last", "draft_team"),
            age_at_as_of=_age(pl.lit(snap.as_of), pl.col("birth_date")),
            knowable_on=pl.max_horizontal("weeks_knowable_on", "draft_knowable_on", "combine_knowable_on"),
        )
        .filter(pl.col("position").is_in(SKILL_POSITIONS))
        .drop("position_last", "team_last", "weeks_knowable_on", "draft_knowable_on", "combine_knowable_on")
        .select("player_id", "name", "position", "team", "birth_date", "age_at_as_of", pl.exclude(
            "player_id", "name", "position", "team", "birth_date", "age_at_as_of"
        ))
        .sort("player_id")
    )


def build_player_tables(snap: AsOfSnapshot, scoring: ScoringConfig) -> PlayerTables:
    weeks = player_weeks(snap, scoring)
    return PlayerTables(
        as_of=snap.as_of,
        player_weeks=weeks,
        player_seasons=player_seasons(snap, weeks),
        player_static=player_static(snap, weeks),
    )


def write_player_tables(tables: PlayerTables, scoring_name: str, root: Path | None = None) -> list[Path]:
    out = (root or data_dir() / "processed") / tables.as_of.isoformat() / scoring_name
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.frames().items():
        path = out / f"{name}.parquet"
        df.write_parquet(path)
        paths.append(path)
    return paths
