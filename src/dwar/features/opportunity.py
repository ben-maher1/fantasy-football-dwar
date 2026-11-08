"""Opportunity points: what each player's weekly role was worth, apart from
how well he used it.

Every target, carry and QB dropback is valued at the average stat line that
plays like it produced (same kind of play, depth of target, field position,
down) in earlier, complete seasons. A player's week is the sum over his
plays, scored with the league's scoring config. A red-zone target is worth
far more than a short target at midfield, whoever catches it; touchdowns
count at their typical rate, not their actual one.

Point in time: the value table for an as-of date uses only complete seasons
whose last game was before that date, and only plays on game days before
it are valued. A season's own plays never value that season.

Play-by-play before 2009 often lacks the targeted receiver on incomplete
passes, so opportunity starts in 2009 (`FIRST_SEASON`).
"""

from __future__ import annotations

from datetime import date
from functools import cache

import polars as pl

from dwar.config import ScoringConfig
from dwar.features.player_tables import PlayerTables
from dwar.features.scoring import fantasy_points

FIRST_SEASON = 2009
MIN_BUCKET_PLAYS = 30  # smaller buckets use their play kind's overall average

DEPTH_EDGES = [0, 5, 10, 15, 20, 30]  # air yards: <0, 0-4, 5-9, 10-14, 15-19, 20-29, 30+
ZONE_EDGES = [6, 11, 21, 41]          # yards to the end zone: 1-5, 6-10, 11-20, 21-40, 41+

RECEIVING = ("receptions", "receiving_yards", "receiving_tds", "receiving_fumbles_lost")
RUSHING = ("rushing_yards", "rushing_tds", "rushing_fumbles_lost")
PASSING = ("passing_yards", "passing_tds", "passing_interceptions", "sack_fumbles_lost")
STATS = (*RECEIVING, *RUSHING, *PASSING)
NOT_OPPORTUNITY = ("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions", "special_teams_tds")


def _bins(col: str, edges: list[int], prefix: str) -> pl.Expr:
    """Bin label `{prefix}{i}`: i edges at or below the value; `{prefix}na` if null."""
    x = pl.col(col)
    idx = pl.sum_horizontal([(x >= e).cast(pl.Int32) for e in edges])
    return pl.when(x.is_null()).then(pl.lit(f"{prefix}na")).otherwise(pl.lit(prefix) + idx.cast(pl.String))


def _flag(col: str) -> pl.Expr:
    return pl.col(col).fill_null(0).cast(pl.Float64)


def plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (player, play) opportunity: player_id, season, week,
    game_date, kind, bucket, and the stat line the play produced for him
    (other stats 0). Regular season only; two-point tries, kneels and spikes
    are not opportunities."""
    p = pbp.filter(
        (pl.col("season_type") == "REG") & (_flag("two_point_attempt") == 0)
        & (_flag("qb_kneel") == 0) & (_flag("qb_spike") == 0)
    ).with_columns(
        depth=_bins("air_yards", DEPTH_EDGES, "d"),
        zone=_bins("yardline_100", ZONE_EDGES, "z"),
        late_down=(pl.col("down").fill_null(1) >= 3),
        complete=_flag("complete_pass"),
        yards=pl.col("yards_gained").fill_null(0).cast(pl.Float64),
        lost=_flag("fumble_lost"),
    )
    keys = ["season", "week", "game_date"]
    zero = {s: pl.lit(0.0) for s in STATS}

    def lost_by(col: str) -> pl.Expr:
        return pl.col("lost") * (pl.col("fumbled_1_player_id") == pl.col(col)).fill_null(False).cast(pl.Float64)

    targets = p.filter((_flag("pass_attempt") == 1) & (_flag("sack") == 0) & pl.col("receiver_player_id").is_not_null())
    targets = targets.select(
        *keys, player_id="receiver_player_id", kind=pl.lit("target"),
        bucket=pl.concat_str("depth", "zone", separator="_"),
        **(zero | {
            "receptions": pl.col("complete"),
            "receiving_yards": pl.col("complete") * pl.col("yards"),
            "receiving_tds": _flag("pass_touchdown"),
            "receiving_fumbles_lost": lost_by("receiver_player_id"),
        }),
    )
    carries = p.filter((_flag("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
    carries = carries.select(
        *keys, player_id="rusher_player_id",
        kind=pl.when(_flag("qb_scramble") == 1).then(pl.lit("scramble")).otherwise(pl.lit("carry")),
        bucket=pl.concat_str("zone", pl.when("late_down").then(pl.lit("late")).otherwise(pl.lit("early")), separator="_"),
        **(zero | {
            "rushing_yards": pl.col("yards"),
            "rushing_tds": _flag("rush_touchdown"),
            "rushing_fumbles_lost": lost_by("rusher_player_id"),
        }),
    )
    dropbacks = p.filter((_flag("pass_attempt") == 1) & pl.col("passer_player_id").is_not_null())
    sacked = _flag("sack") == 1
    dropbacks = dropbacks.select(
        *keys, player_id="passer_player_id",
        kind=pl.when(sacked).then(pl.lit("sack")).otherwise(pl.lit("dropback")),
        bucket=pl.when(sacked).then("zone").otherwise(pl.concat_str("depth", "zone", separator="_")),
        **(zero | {
            "passing_yards": pl.when(sacked).then(0.0).otherwise(pl.col("complete") * pl.col("yards")),
            "passing_tds": _flag("pass_touchdown"),
            "passing_interceptions": _flag("interception"),
            "sack_fumbles_lost": pl.when(sacked).then(lost_by("passer_player_id")).otherwise(0.0),
        }),
    )
    return pl.concat([targets, carries, dropbacks], how="vertical_relaxed").filter(pl.col("season") >= FIRST_SEASON)


def complete_seasons(p: pl.DataFrame, as_of: date) -> list[int]:
    """Seasons whose last game day is before `as_of`."""
    last = p.group_by("season").agg(last=pl.col("game_date").max())
    return sorted(last.filter(pl.col("last") < as_of.isoformat())["season"].to_list())


def value_table(p: pl.DataFrame, seasons: list[int]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Average stat line per (kind, bucket) over `seasons`. Buckets with
    fewer than MIN_BUCKET_PLAYS plays take their kind's average. Also returns
    the per-kind averages."""
    train = p.filter(pl.col("season").is_in(seasons))
    by_kind = train.group_by("kind").agg(*[pl.col(s).mean().alias(f"{s}_kind") for s in STATS])
    by_bucket = train.group_by("kind", "bucket").agg(n=pl.len(), *[pl.col(s).mean() for s in STATS])
    return by_bucket.join(by_kind, on="kind").select(
        "kind", "bucket", "n",
        *[pl.when(pl.col("n") >= MIN_BUCKET_PLAYS).then(pl.col(s)).otherwise(pl.col(f"{s}_kind")).alias(s) for s in STATS],
    ), by_kind.rename({f"{s}_kind": s for s in STATS})


def opportunity_weeks(
    pbp: pl.DataFrame, as_of: date, scoring: ScoringConfig, positions: pl.DataFrame,
) -> pl.DataFrame:
    """player_id, season, week, xfpts (opportunity points under `scoring`),
    plus counts of targets, carries and dropbacks, for every player-week with
    a play on a game day before `as_of`. `positions` (player_id, season,
    position) sets each player-season's position for position bonuses such
    as a TE premium. Empty when no complete season precedes `as_of`."""
    p = plays(pbp)
    seasons = complete_seasons(p, as_of)
    if not seasons:
        return pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int32, "week": pl.Int32, "xfpts": pl.Float64})
    table, by_kind = value_table(p, seasons)
    valued = (
        p.filter(pl.col("game_date") < as_of.isoformat())
        .select("player_id", "season", "week", "kind", "bucket")
        .join(table.drop("n"), on=["kind", "bucket"], how="left")
        .join(by_kind, on="kind", how="left", suffix="_kind")
        .with_columns([pl.coalesce(s, f"{s}_kind").alias(s) for s in STATS])
    )
    weekly = valued.group_by("player_id", "season", "week").agg(
        *[pl.col(s).sum() for s in STATS],
        targets=(pl.col("kind") == "target").sum(),
        carries=pl.col("kind").is_in(["carry", "scramble"]).sum(),
        dropbacks=pl.col("kind").is_in(["dropback", "sack"]).sum(),
    )
    weekly = weekly.join(positions.select("player_id", "season", "position"), on=["player_id", "season"], how="left")
    # Two-point tries and return touchdowns are not opportunities: score them as zero.
    absent = {c: pl.lit(0.0) for c in NOT_OPPORTUNITY}
    scored = fantasy_points(weekly.with_columns(pl.col("position").fill_null(""), **absent), scoring)
    return scored.select(
        "player_id", pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), xfpts="fpts",
        targets="targets", carries="carries", dropbacks="dropbacks",
    ).sort("player_id", "season", "week")


@cache
def _cached_plays() -> pl.DataFrame:
    from dwar.ingest.nflverse import load_raw

    return load_raw("pbp", list(range(FIRST_SEASON, date.today().year + 1)))


def opportunity_table(tables: PlayerTables, scoring: ScoringConfig, pbp: pl.DataFrame | None = None) -> pl.DataFrame:
    """`opportunity_weeks` as of the tables' date, from the cached play-by-play
    unless `pbp` is given, with positions from the tables' player-seasons."""
    pbp = _cached_plays() if pbp is None else pbp
    return opportunity_weeks(pbp, tables.as_of, scoring, tables.player_seasons)
