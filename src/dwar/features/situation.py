"""Situation factors: each player's team context at the start of a season.

Every factor for (player, season s) is what was knowable on the season's
*preseason date*: the day before s's first game, or the as-of date if that
is earlier (a weekly run projecting a season that has not started). Fits
and forecasts use the same definition, so the model learns what a
preseason situation predicts, never what happened later in that season.

Sources and when each row becomes knowable:
- play-by-play, weekly stats: the day after the game
- weekly rosters and (through 2024) weekly depth charts: the day before
  that week's first game (week 1 rosters are set at the late-August cutdown)
- daily depth charts (2025 on): the snapshot's own date
- play-callers: a season's opening play-caller is announced in the
  offseason, so it is known on the preseason date; for a season with no
  games yet, the caller of the team's latest game before the as-of date
- draft picks: as gated by `features.asof`

Level factors shift a season's expected scoring; range factors (0/1 flags)
widen that season's random-walk step. See model/talent.py.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, timedelta

import numpy as np
import polars as pl

from dwar.features.asof import REGULAR_SEASON, AsOfSnapshot
from dwar.features.player_tables import PlayerTables
from dwar.ingest import playcallers
from dwar.ingest.nflverse import load_raw

# Every source's team codes mapped to the current nflverse franchise code.
FRANCHISE = {
    "OAK": "LV", "LVR": "LV", "SD": "LAC", "SDG": "LAC", "STL": "LA", "SL": "LA", "LAR": "LA",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "GNB": "GB", "KAN": "KC",
    "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB",
}
ROSTER_STATUSES = ("ACT", "INA")  # active roster; not reserve lists or practice squad
QB_LOOKBACK = (3.0, 2.0, 1.0)     # weights on the last 3 seasons of EPA per dropback
QB_SHRINK_DROPBACKS = 200.0       # dropbacks' worth of weight on the draft-round prior
CALLER_LOOKBACK = 3               # seasons of a play-caller's history
CALLER_SHRINK_PLAYS = 300.0       # neutral plays' worth of weight on the league average (0)
NEUTRAL_SCORE = 10
NEUTRAL_MAX_QTR = 3
WEAPONS_TOP = 3
WEAPONS_MIN_GAMES = 4
RECEIVING_GROUP = {"WR": ("WR", "TE"), "TE": ("WR", "TE"), "RB": ("RB",), "QB": ("QB",)}

LEVEL = {
    "QB": ("qb_starter", "caller_pass_oe", "team_pace", "weapons"),
    "RB": ("qb_quality", "caller_pass_oe", "team_pace", "competition", "vacated", "rookie_capital"),
    "WR": ("qb_quality", "caller_pass_oe", "team_pace", "competition", "vacated", "rookie_capital"),
    "TE": ("qb_quality", "caller_pass_oe", "team_pace", "competition", "vacated", "rookie_capital"),
}
RANGE = {
    "QB": ("changed_team", "new_caller", "qb_threat"),
    "RB": ("changed_team", "new_caller", "new_qb"),
    "WR": ("changed_team", "new_caller", "new_qb"),
    "TE": ("changed_team", "new_caller", "new_qb"),
}
# Factor groups, for leave-one-group-out validation.
GROUPS = {
    "qb": ("qb_quality", "qb_starter"),
    "coaching": ("caller_pass_oe", "team_pace", "new_caller"),
    "competition": ("competition", "vacated", "rookie_capital", "weapons"),
    "stability": ("changed_team", "new_qb", "qb_threat"),
}


def franchise(col: str | pl.Expr) -> pl.Expr:
    e = pl.col(col) if isinstance(col, str) else col
    return e.replace(FRANCHISE)


@dataclass(frozen=True)
class SituationRaw:
    """Raw inputs beyond `RawTables` (cached by `ingest`)."""

    pbp: pl.DataFrame
    depth_charts: pl.DataFrame
    depth_charts_daily: pl.DataFrame
    rosters_weekly: pl.DataFrame
    schedules: pl.DataFrame
    playcallers: pl.DataFrame

    @classmethod
    def from_cache(cls, pull_date: date | None = None) -> SituationRaw:
        frames = {f.name: load_raw(f.name, pull_date=pull_date) for f in fields(cls) if f.name != "playcallers"}
        return cls(**frames, playcallers=playcallers.load())


@dataclass(frozen=True)
class Sources:
    """Situation inputs normalized and tagged with `knowable_on`; still
    ungated, so one build serves every preseason date."""

    games: pl.DataFrame      # REG team-games: season, week, team, gameday, game_id
    week_dates: pl.DataFrame  # season, week, knowable_on (day before the week's first game)
    pbp: pl.DataFrame        # REG plays with gameday, offense franchise
    stats: pl.DataFrame      # REG player-games from play-by-play: player_id, season, team, targets, carries
    rosters: pl.DataFrame    # season, week, team, player_id, knowable_on
    qb1: pl.DataFrame        # season, team, player_id, knowable_on (QB1 snapshots)
    callers: pl.DataFrame    # season, week, team, gameday, caller


def prepare(sraw: SituationRaw) -> Sources:
    sched = sraw.schedules.filter(pl.col("game_type") == REGULAR_SEASON).with_columns(
        gameday=pl.col("gameday").str.to_date()
    )
    games = pl.concat([
        sched.select("season", "week", "game_id", "gameday", team=franchise("home_team")),
        sched.select("season", "week", "game_id", "gameday", team=franchise("away_team")),
    ])
    week_dates = sched.group_by("season", "week").agg(knowable_on=pl.col("gameday").min() - timedelta(days=1))
    gamedays = sched.select("game_id", "gameday")

    pbp = (
        sraw.pbp.filter(pl.col("season_type") == REGULAR_SEASON)
        .join(gamedays, on="game_id", how="inner")
        .with_columns(team=franchise("posteam"))
        .filter(pl.col("team").is_not_null())
    )
    # Targets and carries from play-by-play: nflverse weekly stats have no
    # targets before 2009.
    keys = ["season", "week", "gameday", "game_id", "team"]
    targets = pbp.filter((pl.col("pass") == 1) & pl.col("receiver_player_id").is_not_null()).group_by(
        *keys, player_id="receiver_player_id").agg(targets=pl.len())
    carries = pbp.filter((pl.col("rush") == 1) & pl.col("rusher_player_id").is_not_null()).group_by(
        *keys, player_id="rusher_player_id").agg(carries=pl.len())
    stats = targets.join(carries, on=[*keys, "player_id"], how="full", coalesce=True).with_columns(
        pl.col("targets").fill_null(0), pl.col("carries").fill_null(0)
    )
    rosters = (
        sraw.rosters_weekly.filter(
            (pl.col("game_type") == REGULAR_SEASON) & pl.col("status").is_in(ROSTER_STATUSES)
            & pl.col("gsis_id").is_not_null()
        )
        .select("season", "week", team=franchise("team"), player_id="gsis_id")
        .unique()
        .join(week_dates, on=["season", "week"], how="inner")
    )
    weekly_qb = (
        sraw.depth_charts.filter(
            (pl.col("game_type") == REGULAR_SEASON) & (pl.col("position") == "QB")
            & (pl.col("depth_team") == "1") & (pl.col("formation") == "Offense") & pl.col("gsis_id").is_not_null()
        )
        .select("season", "week", team=franchise("club_code"), player_id="gsis_id")
        .join(week_dates, on=["season", "week"], how="inner")
        .drop("week")
    )
    daily_qb = sraw.depth_charts_daily.filter(
        (pl.col("pos_abb") == "QB") & (pl.col("pos_rank") == 1) & pl.col("gsis_id").is_not_null()
    ).select(
        pl.col("season").cast(pl.Int32), team=franchise("team"), player_id="gsis_id",
        knowable_on=pl.col("dt").str.slice(0, 10).str.to_date(),
    )
    qb1 = pl.concat([weekly_qb, daily_qb], how="vertical_relaxed").unique(
        ["season", "team", "knowable_on"], keep="first", maintain_order=True
    )
    callers = (
        sraw.playcallers.select("season", "week", team=franchise("team"), caller="off_play_caller")
        .join(games.select("season", "week", "team", "gameday"), on=["season", "week", "team"], how="inner")
    )
    return Sources(games, week_dates, pbp, stats, rosters, qb1, callers)


def preseason_date(src: Sources, season: int, as_of: date) -> date:
    starts = src.week_dates.filter(pl.col("season") == season)
    if starts.is_empty():
        return as_of
    return min(as_of, starts["knowable_on"].min())


def _latest_by_team(df: pl.DataFrame, on: date, season: int) -> pl.DataFrame:
    """Rows of each team's latest snapshot knowable on `on`, from seasons up to `season`."""
    df = df.filter((pl.col("knowable_on") <= on) & (pl.col("season") <= season))
    latest = df.group_by("team").agg(pl.col("knowable_on").max())
    return df.join(latest, on=["team", "knowable_on"], how="semi")


def _qb_epa(pbp: pl.DataFrame, season: int) -> pl.DataFrame:
    """Weighted EPA per dropback and weighted dropbacks over the 3 seasons before `season`."""
    rows = pbp.filter(
        (pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null()
        & pl.col("season").is_between(season - len(QB_LOOKBACK), season - 1)
    )
    w = pl.lit(0.0)
    for k, weight in enumerate(QB_LOOKBACK, start=1):
        w = pl.when(pl.col("season") == season - k).then(weight).otherwise(w)
    return (
        rows.with_columns(w=w)
        .group_by(player_id="passer_player_id")
        .agg(wsum=(pl.col("w") * pl.col("qb_epa").fill_null(0)).sum(), wdb=pl.col("w").sum())
    )


def _qb_prior(pbp: pl.DataFrame, draft: pl.DataFrame) -> dict[str, float]:
    """Mean EPA per dropback in a QB's first two seasons, by draft-round group."""
    db = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null())
    first = db.group_by("passer_player_id").agg(first=pl.col("season").min())
    early = db.join(first, on="passer_player_id").filter(pl.col("season") <= pl.col("first") + 1)
    grouped = early.join(
        draft.select(passer_player_id="player_id", draft_round="draft_round"), on="passer_player_id", how="left"
    ).with_columns(group=_round_group(pl.col("draft_round")))
    out = grouped.group_by("group").agg(pl.col("qb_epa").mean())
    prior = {g: float(v) for g, v in out.iter_rows() if v is not None}
    overall = float(early["qb_epa"].mean()) if early.height else 0.0
    return {g: prior.get(g, overall) for g in ("r1", "r23", "late")}


def _round_group(r: pl.Expr) -> pl.Expr:
    return pl.when(r == 1).then(pl.lit("r1")).when(r <= 3).then(pl.lit("r23")).otherwise(pl.lit("late"))


def _caller_pass_oe(pbp: pl.DataFrame, callers: pl.DataFrame, season: int) -> pl.DataFrame:
    neutral = pbp.filter(
        pl.col("season").is_between(season - CALLER_LOOKBACK, season - 1)
        & pl.col("pass_oe").is_not_null() & (pl.col("qtr") <= NEUTRAL_MAX_QTR)
        & (pl.col("score_differential").abs() <= NEUTRAL_SCORE)
    )
    return (
        neutral.join(callers.select("season", "week", "team", "caller"), on=["season", "week", "team"], how="inner")
        .group_by("caller")
        .agg(caller_pass_oe=pl.col("pass_oe").sum() / (pl.len() + CALLER_SHRINK_PLAYS))
    )


def season_situation(
    src: Sources, snap: AsOfSnapshot, tables: PlayerTables, keys: pl.DataFrame, season: int,
) -> pl.DataFrame:
    """Raw situation factors for `keys` (player_id, position) in `season`,
    using only what was knowable on its preseason date."""
    on = preseason_date(src, season, snap.as_of)
    prev = season - 1
    pbp = src.pbp.filter(pl.col("gameday") < on)
    stats = src.stats.filter(pl.col("gameday") < on)
    weeks = tables.player_weeks.filter(pl.col("gameday") < on)
    draft = snap.draft.with_columns(draft_team=franchise("draft_team"))

    roster = _latest_by_team(src.rosters, on, season).select("team", "player_id").unique()
    # A player's team: his preseason roster team, else (a past season he
    # joined late) his first team that season, else his latest team.
    on_roster = roster.unique("player_id", keep="first")
    first_game = (
        tables.player_weeks.filter(pl.col("season") == season).sort("gameday")
        .group_by("player_id").agg(team_first=pl.col("team").first())
    )
    played = weeks.filter(pl.col("season") < season)
    last_season = played.group_by("player_id").agg(prev_season=pl.col("season").max())
    prev_team = (
        played.join(last_season, on="player_id").filter(pl.col("season") == pl.col("prev_season"))
        .group_by("player_id", "prev_season", "team").agg(n=pl.len(), last=pl.col("gameday").max())
        .sort("n", "last", descending=True).group_by("player_id").first()
        .select("player_id", "prev_season", prev_team="team")
    )
    latest_team = weeks.sort("gameday").group_by("player_id").agg(team_latest=pl.col("team").last())
    base = (
        keys.join(on_roster.rename({"team": "team_roster"}), on="player_id", how="left")
        .join(first_game, on="player_id", how="left")
        .join(latest_team, on="player_id", how="left")
        .join(prev_team, on="player_id", how="left")
        .with_columns(team=pl.coalesce("team_roster", "team_first", "team_latest"))
        .with_columns(team=franchise("team"), prev_team=franchise("prev_team"))
    )

    # QB quality: the team's QB1, his recent EPA per dropback shrunk toward a draft-round prior.
    qb1 = _latest_by_team(src.qb1, on, season).unique("team", keep="first").select("team", qb1="player_id")
    prior = _qb_prior(pbp, draft)
    epa = _qb_epa(pbp, season).rename({"player_id": "qb1"})
    qb_quality = (
        qb1.join(epa, on="qb1", how="left")
        .join(draft.select(qb1="player_id", draft_round="draft_round"), on="qb1", how="left")
        .with_columns(
            p=_round_group(pl.col("draft_round")).replace_strict(prior, default=prior["late"]),
            wsum=pl.col("wsum").fill_null(0.0), wdb=pl.col("wdb").fill_null(0.0),
        )
        .select("team", "qb1", qb_quality=(pl.col("wsum") + QB_SHRINK_DROPBACKS * pl.col("p"))
                / (pl.col("wdb") + QB_SHRINK_DROPBACKS))
    )
    main_passer = (
        pbp.filter((pl.col("season") == prev) & (pl.col("qb_dropback") == 1) & pl.col("passer_player_id").is_not_null())
        .group_by("team", "passer_player_id").agg(n=pl.len())
        .sort("n", descending=True).group_by("team").first()
        .select(prev_team="team", prev_qb="passer_player_id")
    )

    # Play-callers: the opening caller of `season` when its games are
    # scheduled, else the caller of the team's latest game before `on`.
    season_callers = src.callers.filter(pl.col("season") == season).sort("gameday").group_by("team").first()
    recent_callers = src.callers.filter(pl.col("gameday") < on).sort("gameday").group_by("team").last()
    caller = pl.concat([
        season_callers.select("team", "caller"),
        recent_callers.join(season_callers, on="team", how="anti").select("team", "caller"),
    ])
    prev_caller = (
        src.callers.filter((pl.col("season") == prev) & (pl.col("gameday") < on))
        .sort("gameday").group_by("team").last().select(prev_team="team", prev_caller="caller")
    )
    tendency = _caller_pass_oe(pbp, src.callers, season)
    pace = (
        pbp.filter((pl.col("season") == prev) & ((pl.col("pass") == 1) | (pl.col("rush") == 1)))
        .group_by("team").agg(team_pace=pl.len() / pl.col("game_id").n_unique())
    )

    # Competition, from last season's per-game opportunity of every teammate
    # on the preseason roster.
    prev_stats = stats.filter(pl.col("season") == prev)
    positions = snap.rosters.filter(pl.col("season") == prev).select("player_id", "position")
    per_player = (
        prev_stats.group_by("player_id")
        .agg(tgt_pg=pl.col("targets").sum() / pl.len(), opp_pg=(pl.col("targets") + pl.col("carries")).sum() / pl.len())
        .join(positions, on="player_id", how="left")
    )
    n_team_games = prev_stats.group_by("team").agg(g=pl.col("gameday").n_unique())
    team_tgt = prev_stats.group_by("team").agg(t=pl.col("targets").sum()).join(n_team_games, on="team")
    league_tgt_pg = float((team_tgt["t"] / team_tgt["g"]).mean()) if team_tgt.height else np.nan
    rb_prev = prev_stats.join(positions.filter(pl.col("position") == "RB"), on="player_id", how="semi")
    rb_team = rb_prev.group_by("team").agg(o=(pl.col("targets") + pl.col("carries")).sum()).join(n_team_games, on="team")
    league_rb_pg = float((rb_team["o"] / rb_team["g"]).mean()) if rb_team.height else np.nan

    mates = base.select("player_id", "position", "team").join(
        roster.rename({"player_id": "mate"}), on="team", how="inner"
    ).filter(pl.col("mate") != pl.col("player_id")).join(
        per_player.rename({"player_id": "mate", "position": "mate_pos"}), on="mate", how="left"
    )
    competition = mates.group_by("player_id").agg(
        comp_recv=pl.col("tgt_pg").fill_null(0).sum() / league_tgt_pg,
        comp_rb=pl.col("opp_pg").filter(pl.col("mate_pos") == "RB").fill_null(0).sum() / league_rb_pg,
    )
    stay = roster.select("team", "player_id")
    vac_tgt = (
        prev_stats.join(stay, on=["team", "player_id"], how="anti").group_by("team").agg(gone=pl.col("targets").sum())
        .join(team_tgt, on="team", how="right").select("team", vacated_recv=pl.col("gone").fill_null(0) / pl.col("t"))
    )
    vac_rb = (
        rb_prev.join(stay, on=["team", "player_id"], how="anti")
        .group_by("team").agg(gone=(pl.col("targets") + pl.col("carries")).sum())
        .join(rb_team, on="team", how="right").select("team", vacated_rb=pl.col("gone").fill_null(0) / pl.col("o"))
    )
    rookies = draft.filter((pl.col("draft_year") == season) & (pl.col("draft_round") <= 3)).select(
        rookie="player_id", team="draft_team", rookie_pos="draft_position", capital=4 - pl.col("draft_round"),
    )
    capital = (
        base.select("player_id", "position", "team").join(rookies, on="team", how="inner")
        .filter((pl.col("rookie") != pl.col("player_id")) & pl.col("rookie_pos").is_in(["WR", "TE", "RB"]))
        .with_columns(same=pl.col("position").replace_strict(
            {p: list(g) for p, g in RECEIVING_GROUP.items()}, return_dtype=pl.List(pl.String)
        ).list.contains(pl.col("rookie_pos")))
        .group_by("player_id").agg(rookie_capital=pl.col("capital").filter(pl.col("same")).sum())
    )
    threat = draft.filter(
        pl.col("draft_year").is_between(season - 1, season) & (pl.col("draft_round") == 1) & (pl.col("draft_position") == "QB")
    ).select(team="draft_team", rookie="player_id")
    qb_threat = (
        base.select("player_id", "team").join(threat, on="team", how="inner")
        .filter(pl.col("rookie") != pl.col("player_id")).group_by("player_id").agg(qb_threat=pl.lit(1.0))
    )
    prev_ppg = (
        weeks.filter(pl.col("season") == prev).group_by("player_id", "position")
        .agg(ppg=pl.col("fpts").mean(), g=pl.len()).filter(pl.col("g") >= WEAPONS_MIN_GAMES)
    )
    weapons = (
        base.filter(pl.col("position") == "QB").select("player_id", "team")
        .join(roster.rename({"player_id": "mate"}), on="team", how="inner")
        .join(prev_ppg.filter(pl.col("position").is_in(["WR", "TE", "RB"])).select(mate="player_id", ppg="ppg"),
              on="mate", how="inner")
        .sort("ppg", descending=True).group_by("player_id").agg(weapons=pl.col("ppg").head(WEAPONS_TOP).sum())
    )

    has_prev = pl.col("prev_team").is_not_null()
    return (
        base.join(qb_quality, on="team", how="left")
        .join(main_passer, on="prev_team", how="left")
        .join(caller, on="team", how="left")
        .join(prev_caller, on="prev_team", how="left")
        .join(tendency, on="caller", how="left")
        .join(pace, on="team", how="left")
        .join(competition, on="player_id", how="left")
        .join(vac_tgt, on="team", how="left")
        .join(vac_rb, on="team", how="left")
        .join(capital, on="player_id", how="left")
        .join(qb_threat, on="player_id", how="left")
        .join(weapons, on="player_id", how="left")
        .with_columns(
            rb=pl.col("position") == "RB",
            qb_starter=(pl.col("qb1") == pl.col("player_id")).fill_null(False).cast(pl.Float64),
            caller_pass_oe=pl.col("caller_pass_oe").fill_null(0.0),
            rookie_capital=pl.col("rookie_capital").fill_null(0).cast(pl.Float64),
            qb_threat=pl.col("qb_threat").fill_null(0.0),
            weapons=pl.col("weapons").fill_null(0.0),
            changed_team=(has_prev & (pl.col("team") != pl.col("prev_team"))).fill_null(False).cast(pl.Float64),
            new_caller=(has_prev & pl.col("caller").is_not_null() & pl.col("prev_caller").is_not_null()
                        & (pl.col("caller") != pl.col("prev_caller"))).fill_null(False).cast(pl.Float64),
            new_qb=(has_prev & pl.col("qb1").is_not_null() & pl.col("prev_qb").is_not_null()
                    & (pl.col("qb1") != pl.col("prev_qb"))).fill_null(False).cast(pl.Float64),
        )
        .with_columns(
            competition=pl.when("rb").then("comp_rb").otherwise("comp_recv").fill_null(0.0),
            vacated=pl.when("rb").then("vacated_rb").otherwise("vacated_recv"),
            season=pl.lit(season, pl.Int32),
        )
        .select("player_id", "season", "position", "team", "qb1", "caller",
                *sorted({c for cols in (*LEVEL.values(), *RANGE.values()) for c in cols}))
        .sort("player_id")
    )


def situation_table(
    src: Sources, snap: AsOfSnapshot, tables: PlayerTables, keys: pl.DataFrame,
) -> pl.DataFrame:
    """Raw factors for `keys` (player_id, season, position), one season at a time."""
    frames = [
        season_situation(src, snap, tables, keys.filter(pl.col("season") == s).select("player_id", "position"), s)
        for s in sorted(keys["season"].unique().to_list())
    ]
    return pl.concat(frames, how="vertical_relaxed")


@dataclass(frozen=True)
class Covariates:
    """Model-ready factors for one position: level factors z-scored with the
    fit rows' mean and SD (missing -> 0, the position average), range flags
    as 0/1."""

    level: pl.DataFrame   # player_id, season, *level names
    range: pl.DataFrame   # player_id, season, *range names
    center: dict[str, tuple[float, float]]
    raw: pl.DataFrame     # player_id, season, every factor unstandardized

    def roles(self, factor: str) -> pl.DataFrame | None:
        """(player_id, season, role) from a 0/1 factor, if computed."""
        if factor not in self.raw.columns:
            return None
        return self.raw.select("player_id", "season", role=pl.col(factor).cast(pl.Float64)).drop_nulls()


def covariates(
    table: pl.DataFrame, position: str, reference: pl.DataFrame,
    level: tuple[str, ...] | None = None, range_: tuple[str, ...] | None = None,
) -> Covariates:
    """`reference` (player_id, season) marks the rows whose mean and SD
    standardize every row, normally the fit's player-seasons."""
    level = LEVEL[position] if level is None else level
    range_ = RANGE[position] if range_ is None else range_
    t = table.filter(pl.col("position") == position)
    ref = t.join(reference, on=["player_id", "season"], how="semi")
    center = {}
    for c in level:
        x = ref[c].cast(pl.Float64)
        sd = x.std()
        center[c] = (float(x.mean() or 0.0), float(sd) if sd and sd > 0 else 1.0)
    keys = ["player_id", "season"]
    lvl = t.select(*keys, *[((pl.col(c) - center[c][0]) / center[c][1]).fill_null(0.0).fill_nan(0.0).alias(c) for c in level])
    rng = t.select(*keys, *[pl.col(c).fill_null(0.0).cast(pl.Float64) for c in range_])
    raw = t.select(*keys, *[c for c in t.columns if c in {f for cols in (*LEVEL.values(), *RANGE.values()) for f in cols}])
    return Covariates(level=lvl, range=rng, center=center, raw=raw)


def model_covariates(
    src: Sources, snap: AsOfSnapshot, tables: PlayerTables, first_season: int,
    level: dict[str, tuple[str, ...]], range_: dict[str, tuple[str, ...]], upcoming: tuple[int, ...],
) -> dict[str, Covariates]:
    """Each position's factors for its fit rows (every played season from
    `first_season`, plus unplayed rookies) and for each `upcoming` season
    of every known player, standardized on the played rows."""
    played = tables.player_seasons.filter(pl.col("season") >= first_season).select("player_id", "season", "position")
    ahead = [tables.player_static.select("player_id", "position", season=pl.lit(s, pl.Int32)) for s in upcoming]
    keys = pl.concat([played, *[a.select(played.columns) for a in ahead]], how="vertical_relaxed").unique(
        ["player_id", "season"], keep="first", maintain_order=True
    )
    table = situation_table(src, snap, tables, keys)
    reference = played.select("player_id", "season")
    positions = sorted(set(level) | set(range_))
    return {p: covariates(table, p, reference, level.get(p, ()), range_.get(p, ())) for p in positions}
