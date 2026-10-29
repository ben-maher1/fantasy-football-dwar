"""Turn step 2 player tables into the arrays the PyMC model consumes.

Pure functions only. Everything comes from `PlayerTables`, so the model
inherits the as-of guarantees of `features.asof`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from dwar.features.player_tables import DAYS_PER_YEAR, PlayerTables

log = logging.getLogger(__name__)

AGE_CENTER = 26.0
UNDRAFTED_PICK = 260
COMBINE_FEATURES = ("forty", "ht_in", "wt")


@dataclass(frozen=True)
class ModelData:
    position: str
    # one entry per observed game
    y: np.ndarray            # weekly fantasy points
    zero_week_mean: float    # mean points over weeks at or below zero (QBs go negative)
    obs_ps: np.ndarray       # index into player-seasons
    # one entry per player-season, sorted by player then season
    ps_keys: pl.DataFrame    # player_id, season, games_played
    ps_player: np.ndarray    # index into players
    ps_age_c: np.ndarray     # age on Sept 1 minus AGE_CENTER
    ps_is_first: np.ndarray  # first in-window season for the player
    ps_gap: np.ndarray       # seasons since the player's previous season (1 = consecutive)
    ps_start: np.ndarray     # index of the player's first player-season
    season_x: np.ndarray     # (n_ps, k) covariates shifting talent; k may be 0
    season_x_names: tuple[str, ...]
    # one entry per player
    player_ids: tuple[str, ...]
    log_pick_z: np.ndarray   # standardized log draft pick (undrafted at pick 260)
    undrafted: np.ndarray
    pre_window: np.ndarray   # career began before the fitting window: no rookie prior
    prospect_x: np.ndarray   # (n_players, m) standardized prospect covariates
    prospect_x_names: tuple[str, ...]

    @property
    def n_obs(self) -> int:
        return len(self.y)

    @property
    def n_ps(self) -> int:
        return len(self.ps_player)

    @property
    def n_players(self) -> int:
        return len(self.player_ids)


FLAG_MERGE_SHARE = 0.01  # flags differing on fewer than this share of rows are merged


def standardize_with_flags(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """z-score each column over its non-null values; nulls become 0 with a
    missing flag so a missing combine drill is neither dropped nor mistaken
    for an average one. Flags that are always 0 are dropped, and flags that
    are identical or nearly so (height and weight are missing together, bar
    a player or two) are merged into one that is set if either is, since
    near-duplicate columns leave their coefficients barely identified and
    the sampler mixes poorly on them."""
    zs, flags = [], []  # flags: [names, missing array]
    for c in columns:
        x = pl.col(c).cast(pl.Float64)
        sd = df[c].cast(pl.Float64).std()
        zs.append(((x - x.mean()) / (sd if sd and sd > 0 else 1.0)).fill_null(0.0).alias(f"{c}_z"))
        missing = df[c].is_null().to_numpy()
        if not missing.any():
            continue
        for flag in flags:
            if (flag[1] != missing).mean() < FLAG_MERGE_SHARE:
                flag[0].append(c)
                flag[1] = flag[1] | missing
                break
        else:
            flags.append([[c], missing])
    flag_cols = [pl.Series("_".join(names) + "_missing", key, dtype=pl.Float64) for names, key in flags]
    return df.select(zs).with_columns(flag_cols)


def _unplayed_rookie_seasons(tables: PlayerTables, position: str, first_season: int) -> pl.DataFrame:
    static = tables.player_static
    empty = pl.DataFrame(schema={"player_id": pl.String, "season": pl.Int32, "games_played": pl.UInt32, "age_sept1": pl.Float64})
    if "draft_year" not in static.columns or static["draft_year"].drop_nulls().is_empty():
        return empty
    latest = static["draft_year"].max()
    if latest < first_season:
        return empty
    return static.filter(
        (pl.col("draft_year") == latest) & (pl.col("position") == position) & pl.col("first_season").is_null()
    ).select(
        "player_id",
        season=pl.lit(latest).cast(pl.Int32),
        games_played=pl.lit(0).cast(pl.UInt32),
        age_sept1=((pl.date(latest, 9, 1) - pl.col("birth_date")).dt.total_days() / DAYS_PER_YEAR).round(3),
    )


def build_model_data(
    tables: PlayerTables,
    position: str = "WR",
    first_season: int = 2006,
    season_covariates: pl.DataFrame | None = None,
    prospect_covariates: pl.DataFrame | None = None,
    unplayed_rookies: bool = True,
    min_plays: int = 0,
) -> ModelData:
    """`season_covariates` is keyed by (player_id, season) and
    `prospect_covariates` by player_id; every remaining column is a numeric
    covariate. Both slots are empty in v1 and filled by later steps.

    With `unplayed_rookies`, players from the latest draft class who have not
    played get a rookie-season row with no games, so the model draws their
    talent from the rookie prior and they can be valued before Week 1.

    Weeks with fewer than `min_plays` plays (an injury exit, a mop-up
    series) still count as games played but are left out of the observed
    points, because they say little about the player's scoring rate."""
    weeks = tables.player_weeks.filter(
        (pl.col("position") == position) & (pl.col("season") >= first_season)
    )
    ps = (
        weeks.group_by("player_id", "season")
        .agg(games_played=pl.len())
        .join(tables.player_seasons.select("player_id", "season", "age_sept1"), on=["player_id", "season"], how="left")
    )
    if unplayed_rookies:
        ps = pl.concat([ps, _unplayed_rookie_seasons(tables, position, first_season)], how="vertical_relaxed")
    no_age = ps.filter(pl.col("age_sept1").is_null())["player_id"].unique()
    if len(no_age):
        log.warning("dropping %d %s players with no birth date", len(no_age), position)
        ps = ps.filter(~pl.col("player_id").is_in(no_age.implode()))
        weeks = weeks.filter(~pl.col("player_id").is_in(no_age.implode()))
    if ps.is_empty():
        raise ValueError(f"no {position} games from {first_season} in these tables")

    ps = (
        ps.sort("player_id", "season")
        .with_row_index("ps_idx")
        .with_columns(
            player_idx=pl.col("player_id").rank("dense").cast(pl.Int64) - 1,
            prev_season=pl.col("season").shift(1).over("player_id"),
            ps_start=pl.col("ps_idx").min().over("player_id"),
        )
        .with_columns(
            is_first=pl.col("prev_season").is_null(),
            gap=(pl.col("season") - pl.col("prev_season")).fill_null(0),
        )
    )
    # Explicit sorts after every join: polars does not promise join order, and
    # both alignment and bit-for-bit reproducible sampling depend on it.
    observed = weeks
    if min_plays:
        if "plays" not in weeks.columns or weeks["plays"].null_count() == weeks.height:
            raise ValueError("min_plays needs weekly play counts, and these tables have none")
        observed = weeks.filter(pl.col("plays") >= min_plays)
    obs = (
        observed.with_row_index("_row")
        .join(ps.select("player_id", "season", "ps_idx"), on=["player_id", "season"], how="inner")
        .sort("_row")
    )

    players = ps.select("player_id", "player_idx").unique().sort("player_idx")
    static = players.join(tables.player_static, on="player_id", how="left").sort("player_idx")
    drafted = static.filter(pl.col("draft_pick").is_not_null())
    log_pick = pl.col("draft_pick").fill_null(UNDRAFTED_PICK).cast(pl.Float64).log()
    mu = drafted.select(log_pick.mean()).item() if drafted.height else 0.0
    sd = drafted.select(log_pick.std()).item() if drafted.height > 1 else 1.0
    player_feats = static.select(
        log_pick_z=(log_pick - mu) / (sd or 1.0),
        undrafted=pl.col("draft_pick").is_null().cast(pl.Float64),
        pre_window=(pl.col("first_season").fill_null(first_season) < first_season).cast(pl.Float64),
    )

    prospect = standardize_with_flags(static, COMBINE_FEATURES)
    if prospect_covariates is not None:
        extra = (
            players.join(prospect_covariates, on="player_id", how="left")
            .sort("player_idx")
            .drop("player_id", "player_idx")
        )
        prospect = pl.concat([prospect, extra], how="horizontal")

    if season_covariates is not None:
        season_x = (
            ps.select("ps_idx", "player_id", "season")
            .join(season_covariates, on=["player_id", "season"], how="left")
            .sort("ps_idx")
            .drop("ps_idx", "player_id", "season")
        )
        if season_x.null_count().sum_horizontal().item():
            raise ValueError("season_covariates must cover every player-season with no nulls")
    else:
        season_x = pl.DataFrame()

    y = obs["fpts"].to_numpy().astype(float)
    return ModelData(
        position=position,
        y=y,
        zero_week_mean=float(y[y <= 0].mean()) if (y <= 0).any() else 0.0,
        obs_ps=obs["ps_idx"].to_numpy().astype(np.int64),
        ps_keys=ps.select("player_id", "season", "games_played"),
        ps_player=ps["player_idx"].to_numpy(),
        ps_age_c=(ps["age_sept1"] - AGE_CENTER).to_numpy().astype(float),
        ps_is_first=ps["is_first"].to_numpy(),
        ps_gap=ps["gap"].to_numpy().astype(float),
        ps_start=ps["ps_start"].to_numpy().astype(np.int64),
        season_x=season_x.to_numpy().astype(float).reshape(ps.height, season_x.width),
        season_x_names=tuple(season_x.columns),
        player_ids=tuple(players["player_id"].to_list()),
        log_pick_z=player_feats["log_pick_z"].to_numpy(),
        undrafted=player_feats["undrafted"].to_numpy(),
        pre_window=player_feats["pre_window"].to_numpy(),
        prospect_x=prospect.to_numpy().astype(float),
        prospect_x_names=tuple(prospect.columns),
    )
