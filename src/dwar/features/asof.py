"""The single point-in-time gate (CLAUDE.md rule 1).

nflverse publishes current snapshots, not vintages, so "data as of a date"
has to be rebuilt here: every row gets the date it became knowable
(`knowable_on`) and is kept only if `knowable_on <= as_of`. Columns are
selected by allowlist so fields that describe today (current position,
current team, career outcomes) cannot pass through.

Knowable-on rules:
- weekly stats, snap counts: the day after the game (`gameday + 1`)
- season rosters (position, team): the season's first regular-season game
- draft picks: the day after the draft ends (config/draft_dates.yaml)
- combine: March 15 of the draft year
- schedules (game dates only): June 1 of the season
- birth dates: static, always knowable

Known limits: nflverse overwrites stat corrections in place, and a season's
roster row reflects the latest state of that season, so an in-progress
season can show a mid-season position change slightly early.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, timedelta

import polars as pl

from dwar.config import DraftDates, load_draft_dates
from dwar.features.scoring import STAT_COLUMNS
from dwar.ingest.nflverse import load_raw

COMBINE_KNOWABLE_MONTH_DAY = (3, 15)
# The NFL releases each season's schedule in April or May.
SCHEDULE_KNOWABLE_MONTH_DAY = (6, 1)
REGULAR_SEASON = "REG"

STAT_KEEP = (
    "player_id", "player_display_name", "season", "week", "game_id", "team", "opponent_team",
    *sorted({c for cols in STAT_COLUMNS.values() for c in cols}),
)
COMBINE_KEEP = ("ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle")


@dataclass(frozen=True)
class RawTables:
    """The raw nflverse inputs a snapshot is built from."""

    player_stats: pl.DataFrame
    schedules: pl.DataFrame
    rosters: pl.DataFrame
    draft_picks: pl.DataFrame
    combine: pl.DataFrame
    snap_counts: pl.DataFrame
    players: pl.DataFrame

    @classmethod
    def from_cache(cls, pull_date: date | None = None) -> RawTables:
        return cls(**{f.name: load_raw(f.name, pull_date=pull_date) for f in fields(cls)})


@dataclass(frozen=True)
class AsOfSnapshot:
    """Everything a fit may see at `as_of`. Each frame has `knowable_on`."""

    as_of: date
    stats: pl.DataFrame      # REG player-weeks, scoring inputs only
    snaps: pl.DataFrame      # REG player-weeks with offense snaps, keyed to gsis ids
    rosters: pl.DataFrame    # season, player_id, position, team
    draft: pl.DataFrame      # player_id, draft_year, draft_round, draft_pick, draft_position, draft_team
    combine: pl.DataFrame    # player_id, draft_year, measurables
    bio: pl.DataFrame        # player_id, name, birth_date
    schedules: pl.DataFrame  # REG game dates for seasons whose schedule was published

    def frames(self) -> dict[str, pl.DataFrame]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "as_of"}


def _gamedays(schedules: pl.DataFrame) -> pl.DataFrame:
    return schedules.filter(pl.col("game_type") == REGULAR_SEASON).select(
        "game_id", "season", "week", gameday=pl.col("gameday").str.to_date()
    )


def _pfr_to_gsis(players: pl.DataFrame) -> pl.DataFrame:
    return players.select(pfr_id="pfr_id", player_id="gsis_id").drop_nulls().unique("pfr_id")


def _gate(df: pl.DataFrame, as_of: date) -> pl.DataFrame:
    return df.filter(pl.col("knowable_on") <= as_of)


def asof_snapshot(
    raw: RawTables, as_of: date, draft_dates: DraftDates | None = None
) -> AsOfSnapshot:
    draft_dates = draft_dates or load_draft_dates()
    games = _gamedays(raw.schedules)
    pfr_map = _pfr_to_gsis(raw.players)

    stats = _gate(
        raw.player_stats.filter(pl.col("player_id").is_not_null())
        .select(STAT_KEEP)
        .join(games.select("game_id", "gameday"), on="game_id", how="inner")  # inner = REG only
        .with_columns(knowable_on=pl.col("gameday") + timedelta(days=1)),
        as_of,
    )

    snaps = _gate(
        raw.snap_counts.join(pfr_map, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .join(games.select("game_id", "gameday"), on="game_id", how="inner")
        .select(
            "player_id", "season", "week", "game_id", "team", "gameday",
            pl.col("opponent").alias("opponent_team"),
            pl.col("offense_snaps").fill_null(0),
        )
        .with_columns(knowable_on=pl.col("gameday") + timedelta(days=1)),
        as_of,
    )

    season_start = games.group_by("season").agg(knowable_on=pl.col("gameday").min())
    rosters = _gate(
        raw.rosters.filter(pl.col("gsis_id").is_not_null())
        .select("season", player_id="gsis_id", position="position", team="team")
        .unique(["season", "player_id"], keep="last", maintain_order=True)
        .join(season_start, on="season", how="inner"),
        as_of,
    )

    picks = raw.draft_picks.join(
        pfr_map.rename({"player_id": "pfr_gsis"}), left_on="pfr_player_id", right_on="pfr_id", how="left"
    )
    draft_years = picks["season"].unique().to_list()
    knowable = pl.DataFrame(
        {"season": draft_years, "knowable_on": [draft_dates.end_of(y) + timedelta(days=1) for y in draft_years]},
        schema={"season": picks.schema["season"], "knowable_on": pl.Date},
    )
    draft = (
        picks.select(
            player_id=pl.coalesce("gsis_id", "pfr_gsis"),
            draft_year="season",
            draft_round="round",
            draft_pick="pick",
            draft_position="position",
            draft_team="team",
        )
        .filter(pl.col("player_id").is_not_null())
        .join(knowable.rename({"season": "draft_year"}), on="draft_year", how="inner")
        .pipe(_gate, as_of)
        .sort("draft_year")
        .unique("player_id", keep="first", maintain_order=True)
    )

    month, day = COMBINE_KNOWABLE_MONTH_DAY
    combine = (
        raw.combine.join(pfr_map, on="pfr_id", how="inner")
        .select("player_id", pl.col("season").alias("draft_year"), *COMBINE_KEEP)
        .with_columns(knowable_on=pl.date(pl.col("draft_year"), month, day))
        .pipe(_gate, as_of)
        .sort("draft_year")
        .unique("player_id", keep="last", maintain_order=True)
    )

    # The players table lists everyone nflverse knows today, so restrict it to
    # players who exist in some gated frame: their existence is itself knowledge.
    known = pl.concat([f.select("player_id") for f in (stats, snaps, rosters, draft, combine)]).unique()
    bio = (
        raw.players.filter(pl.col("gsis_id").is_not_null())
        .select(
            player_id="gsis_id",
            name="display_name",
            birth_date=pl.col("birth_date").str.to_date(strict=False),
            knowable_on=pl.lit(date.min),
        )
        .join(known, on="player_id", how="semi")
    )

    return AsOfSnapshot(
        as_of=as_of,
        stats=stats,
        snaps=snaps,
        rosters=rosters,
        draft=draft,
        combine=combine,
        bio=bio,
        schedules=_gate(
            games.with_columns(knowable_on=pl.date(pl.col("season"), *SCHEDULE_KNOWABLE_MONTH_DAY)),
            as_of,
        ),
    )
