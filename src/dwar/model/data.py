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
    season_vol_x: np.ndarray      # (n_ps, j) flags widening that season's random-walk step; j may be 0
    season_vol_x_names: tuple[str, ...]
    # one entry per player
    player_ids: tuple[str, ...]
    log_pick_z: np.ndarray   # standardized log draft pick (undrafted at pick 260)
    undrafted: np.ndarray
    pre_window: np.ndarray   # career began before the fitting window: no rookie prior
    prospect_x: np.ndarray   # (n_players, m) standardized prospect covariates
    prospect_x_names: tuple[str, ...]
    round1: np.ndarray | None = None        # (n_players,) first-round pick, for the rookie prior's first-round step
    # (n_ps,) 1 = rookie season (draft year, else first NFL season); 0 when the
    # career began before the window: such players start from the veteran
    # prior, which already reflects their career stage, so growth terms skip them.
    ps_career_year: np.ndarray | None = None
    x: np.ndarray | None = None             # (n_obs,) weekly opportunity points; NaN where unknown
    obs_week: np.ndarray | None = None      # (n_obs,) week of the season
    obs_career_games: np.ndarray | None = None  # (n_obs,) games the player had played before this one, whole career
    obs_qb: np.ndarray | None = None        # (n_obs,) rating of that game's QB (features/qb_rating.py); 0 when unknown
    ps_qb_end: np.ndarray | None = None     # (n_ps,) the QB rating in the player's last game of that season
    ps_career_games: np.ndarray | None = None  # (n_ps,) career games before that season's first game
    ps_career_games_end: np.ndarray | None = None  # (n_ps,) career games after that season's last game
    obs_exit: np.ndarray | None = None      # (n_obs,) injury-exit game (features/exits.py), kept only with injury_exits="flag"
    obs_usage: np.ndarray | None = None     # (n_obs, 2) usage from the season's earlier games (usage_design); None without usage columns
    ps_usage_next: np.ndarray | None = None  # (n_ps, 2) usage as of a next game after each player-season's last (value layer)

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


def standardize_with_flags(
    df: pl.DataFrame, columns: tuple[str, ...], ref: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """z-score each column over its non-null values; nulls become 0 with a
    missing flag so a missing combine drill is neither dropped nor mistaken
    for an average one. Flags that are always 0 are dropped, and flags that
    are identical or nearly so (height and weight are missing together, bar
    a player or two) are merged into one that is set if either is, since
    near-duplicate columns leave their coefficients barely identified and
    the sampler mixes poorly on them.

    With `ref`, the means, SDs and flag columns come from `ref` and are
    applied to `df`, so new players are scored on the scale a fit learned."""
    ref = df if ref is None else ref
    zs, flags = [], []  # flags: [names, missing array in ref]
    for c in columns:
        x = pl.col(c).cast(pl.Float64)
        r = ref[c].cast(pl.Float64)
        sd = r.std()
        zs.append(((x - r.mean()) / (sd if sd and sd > 0 else 1.0)).fill_null(0.0).alias(f"{c}_z"))
        missing = ref[c].is_null().to_numpy()
        if not missing.any():
            continue
        for flag in flags:
            if (flag[1] != missing).mean() < FLAG_MERGE_SHARE:
                flag[0].append(c)
                flag[1] = flag[1] | missing
                break
        else:
            flags.append([[c], missing])
    flag_cols = [
        pl.any_horizontal([pl.col(n).is_null() for n in names]).cast(pl.Float64).alias("_".join(names) + "_missing")
        for names, _ in flags
    ]
    return df.select(*zs, *flag_cols)


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


def _per_season(ps: pl.DataFrame, covariates: pl.DataFrame | None, name: str) -> pl.DataFrame:
    if covariates is None:
        return pl.DataFrame()
    x = (
        ps.select("ps_idx", "player_id", "season")
        .join(covariates, on=["player_id", "season"], how="left")
        .sort("ps_idx")
        .drop("ps_idx", "player_id", "season")
    )
    if x.null_count().sum_horizontal().item():
        raise ValueError(f"{name} must cover every player-season with no nulls")
    return x


CARRY_SHARE_REF = 0.30  # about the average RB game's carry share (2013-2019), the usage term's zero
USAGE_NAMES = ("carry_share_l3", "snap_share_change")


def usage_design(obs: pl.DataFrame) -> np.ndarray:
    """(n_obs, 2) usage known before each game, from the same season's
    earlier games only (no lookahead): carry share over the last (up to) 3
    games less CARRY_SHARE_REF, and snap share over the last 3 less the
    season's games before those (needs 4 earlier games). 0 when unknown
    (a season's first game, or no snap counts before 2013)."""
    snap = pl.col("offense_pct") if "offense_pct" in obs.columns else pl.lit(None, pl.Float64)
    o = obs.with_row_index("_u").sort("player_id", "season", "week").with_columns(
        _cs=pl.col("carry_share").cast(pl.Float64), _sn=snap.cast(pl.Float64))
    by = ["player_id", "season"]
    o = o.with_columns(
        _k=pl.int_range(pl.len()).over(by),
        _cs3=pl.col("_cs").shift(1).rolling_mean(3, min_samples=1).over(by),
        _sn3=pl.col("_sn").shift(1).rolling_mean(3, min_samples=3).over(by),
        _snpre=pl.col("_sn").shift(4).cum_sum().over(by) / (pl.int_range(pl.len()).over(by) - 3),
    )
    o = o.with_columns(
        u1=(pl.col("_cs3") - CARRY_SHARE_REF).fill_null(0.0).fill_nan(0.0),
        u2=pl.when(pl.col("_k") >= 4).then(pl.col("_sn3") - pl.col("_snpre")).otherwise(None).fill_null(0.0).fill_nan(0.0),
    ).sort("_u")
    return o.select("u1", "u2").to_numpy().astype(float)


def build_model_data(
    tables: PlayerTables,
    position: str = "WR",
    first_season: int = 2006,
    season_covariates: pl.DataFrame | None = None,
    prospect_covariates: pl.DataFrame | None = None,
    season_range_covariates: pl.DataFrame | None = None,
    unplayed_rookies: bool = True,
    min_plays: int = 0,
    opportunity: pl.DataFrame | None = None,
    reference_players: tuple[str, ...] | None = None,
    injury_exits: str = "keep",
) -> ModelData:
    """`season_covariates` is keyed by (player_id, season) and
    `prospect_covariates` by player_id; every remaining column is a numeric
    covariate. Both slots are empty in v1 and filled by later steps.
    `season_range_covariates`, keyed like `season_covariates`, scale each
    season's random-walk step (see model/talent.py).

    With `unplayed_rookies`, players from the latest draft class who have not
    played get a rookie-season row with no games, so the model draws their
    talent from the rookie prior and they can be valued before Week 1.

    Weeks with fewer than `min_plays` plays (an injury exit, a mop-up
    series) still count as games played but are left out of the observed
    points, because they say little about the player's scoring rate.

    `opportunity` (player_id, season, week, xfpts; features/opportunity.py)
    adds each observed week's opportunity points. A week in a season the
    table covers but with no row had no opportunities (0); weeks in seasons
    it does not cover are NaN and carry no opportunity information.

    `injury_exits` "drop" leaves injury-exit games (features/exits.py:
    the player left early hurt) out of the observed points like short
    weeks; "flag" keeps them and marks them in `obs_exit`.

    `reference_players` (a fit's `player_ids`) sets the draft-pick and
    combine standardization, so data built later for new players matches
    the scale that fit learned (used to step a fit through later seasons)."""
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
    if injury_exits == "drop":
        if "injury_exit" not in weeks.columns:
            raise ValueError("injury_exits='drop' needs player_weeks.injury_exit")
        observed = observed.filter(~pl.col("injury_exit").fill_null(False))
    obs = (
        observed.with_row_index("_row")
        .join(ps.select("player_id", "season", "ps_idx"), on=["player_id", "season"], how="inner")
        .sort("_row")
    )

    players = ps.select("player_id", "player_idx").unique().sort("player_idx")
    static = players.join(tables.player_static, on="player_id", how="left").sort("player_idx")
    ref_static = static if reference_players is None else (
        pl.DataFrame({"player_id": list(reference_players)}).join(tables.player_static, on="player_id", how="left")
    )
    drafted = ref_static.filter(pl.col("draft_pick").is_not_null())
    log_pick = pl.col("draft_pick").fill_null(UNDRAFTED_PICK).cast(pl.Float64).log()
    mu = drafted.select(log_pick.mean()).item() if drafted.height else 0.0
    sd = drafted.select(log_pick.std()).item() if drafted.height > 1 else 1.0
    player_feats = static.select(
        log_pick_z=(log_pick - mu) / (sd or 1.0),
        undrafted=pl.col("draft_pick").is_null().cast(pl.Float64),
        pre_window=(pl.col("first_season").fill_null(first_season) < first_season).cast(pl.Float64),
    )

    prospect = standardize_with_flags(static, COMBINE_FEATURES, None if reference_players is None else ref_static)
    if prospect_covariates is not None:
        extra = (
            players.join(prospect_covariates, on="player_id", how="left")
            .sort("player_idx")
            .drop("player_id", "player_idx")
        )
        prospect = pl.concat([prospect, extra], how="horizontal")

    season_x = _per_season(ps, season_covariates, "season_covariates")
    season_vol_x = _per_season(ps, season_range_covariates, "season_range_covariates")

    x = None
    if opportunity is not None:
        covered = opportunity["season"].unique().implode()
        joined = (
            obs.select("_row", "player_id", "season", "week")
            .join(opportunity.select("player_id", "season", "week", "xfpts"), on=["player_id", "season", "week"], how="left")
            .sort("_row")
        )
        x = joined.select(
            pl.when(pl.col("season").is_in(covered)).then(pl.col("xfpts").fill_null(0.0)).otherwise(None)
        ).to_series().cast(pl.Float64).fill_null(float("nan")).to_numpy()
    static_all = tables.player_static
    for c in ("draft_year", "draft_round", "first_season"):
        if c not in static_all.columns:
            static_all = static_all.with_columns(pl.lit(None, dtype=pl.Int64).alias(c))
    entry = pl.coalesce("draft_year", "first_season").fill_null(pl.col("season"))
    career = (
        ps.select("ps_idx", "player_id", "season")
        .join(static_all.select("player_id", "draft_year", "first_season"), on="player_id", how="left")
        .sort("ps_idx")
        .select(cy=(pl.col("season") - entry + 1).clip(lower_bound=1))
    )

    if "week" not in obs.columns:  # tables without weeks (small tests): rows are in game order
        obs = obs.with_columns(week=(pl.int_range(pl.len()).over("player_id", "season", order_by="_row") + 1).cast(pl.Int64))
    all_weeks = tables.player_weeks
    if "week" not in all_weeks.columns:
        all_weeks = all_weeks.with_row_index("_r").with_columns(
            week=(pl.int_range(pl.len()).over("player_id", "season", order_by="_r") + 1).cast(pl.Int64)
        )
    prior_games = (
        all_weeks.select("player_id", "season", pl.col("week").cast(pl.Int64))
        .unique()
        .sort("player_id", "season", "week")
        .with_columns(career_games=pl.int_range(pl.len()).over("player_id"))
    )
    obs = obs.with_columns(pl.col("week").cast(pl.Int64)).join(
        prior_games, on=["player_id", "season", "week"], how="left"
    ).sort("_row")
    y = obs["fpts"].to_numpy().astype(float)
    obs_usage = usage_design(obs) if "carry_share" in obs.columns else None
    ps_usage_next = None
    if obs_usage is not None:
        # One extra (unplayed) game after each player-season's last, so its
        # usage is what the next game would see.
        nxt = (obs.group_by("player_id", "season", "ps_idx").agg(pl.col("week").max() + 1)
               .with_columns(_next=pl.lit(True)))
        cols = ["player_id", "season", "week", "carry_share", *(["offense_pct"] if "offense_pct" in obs.columns else [])]
        ext = pl.concat([obs.select(*cols, "ps_idx").with_columns(_next=pl.lit(False)), nxt], how="diagonal_relaxed")
        u = usage_design(ext)
        flag = ext["_next"].to_numpy()
        ps_usage_next = np.zeros((ps.height, 2))
        ps_usage_next[ext["ps_idx"].to_numpy()[flag]] = u[flag]
    obs_qb = (obs["qb_rating"].fill_null(0.0).to_numpy().astype(float) if "qb_rating" in obs.columns
              else np.zeros(obs.height))
    ends = (pl.DataFrame({"ps_idx": obs["ps_idx"], "week": obs["week"], "q": obs_qb})
            .sort("ps_idx", "week").group_by("ps_idx").agg(pl.col("q").last()))
    ps_qb_end = np.zeros(ps.height)
    ps_qb_end[ends["ps_idx"].to_numpy()] = ends["q"].to_numpy()
    cg = (prior_games.group_by("player_id", "season")
          .agg(cg0=pl.col("career_games").min(), cg1=pl.col("career_games").max() + 1))
    cg = ps.select("player_id", "season", "ps_idx").join(cg, on=["player_id", "season"], how="left").sort("ps_idx")
    cg = cg.with_columns(pl.col("cg1").fill_null(strategy="forward").over("player_id").fill_null(0))
    cg = cg.with_columns(pl.col("cg0").fill_null(pl.col("cg1").shift(1).over("player_id")).fill_null(0))
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
        season_vol_x=season_vol_x.to_numpy().astype(float).reshape(ps.height, season_vol_x.width),
        season_vol_x_names=tuple(season_vol_x.columns),
        player_ids=tuple(players["player_id"].to_list()),
        log_pick_z=player_feats["log_pick_z"].to_numpy(),
        undrafted=player_feats["undrafted"].to_numpy(),
        pre_window=player_feats["pre_window"].to_numpy(),
        prospect_x=prospect.to_numpy().astype(float),
        prospect_x_names=tuple(prospect.columns),
        round1=players.join(static_all.select("player_id", "draft_round"), on="player_id", how="left").sort("player_idx")
        .select((pl.col("draft_round") == 1).fill_null(False).cast(pl.Float64)).to_series().to_numpy(),
        ps_career_year=np.where(
            player_feats["pre_window"].to_numpy()[ps["player_idx"].to_numpy()] > 0, 0, career["cy"].to_numpy()
        ).astype(np.int64),
        x=x,
        obs_week=obs["week"].to_numpy().astype(np.int64),
        obs_career_games=obs["career_games"].fill_null(0).to_numpy().astype(float),
        obs_qb=obs_qb,
        obs_usage=obs_usage,
        ps_usage_next=ps_usage_next,
        ps_qb_end=ps_qb_end,
        ps_career_games=cg["cg0"].to_numpy().astype(float),
        ps_career_games_end=cg["cg1"].to_numpy().astype(float),
        obs_exit=obs["injury_exit"].fill_null(False).to_numpy() if injury_exits == "flag" and "injury_exit" in obs.columns
        else None,
    )
