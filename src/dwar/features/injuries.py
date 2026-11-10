"""Injury history: weeks each player missed to injury, by body part.

Sources (both nflverse, both regular season only):
- weekly injury reports (2009 on): body part and game status (Out,
  Doubtful, Questionable). Players already on injured reserve are not listed.
- weekly rosters: status `RES` marks reserve lists (IR, PUP, NFI).

A player-week counts as *missed to injury* when the player's team played,
the player did not, and the player was on a reserve list for an injury or
was reported Out or Doubtful with an injury. Reports say "knee", never
"ACL", so the body part is the finest detail available and severity comes
from how many weeks were missed. Illness and non-injury absences (rest,
personal, coach's decision) are not injuries. A reserve week with no report
that season takes the group "other".

Point in time: a team-week's report and roster status are knowable on that
week's game day (reports are final before kickoff). The current status at
an as-of date is the player's latest knowable team-week.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import polars as pl

from dwar.features.player_tables import SKILL_POSITIONS
from dwar.ingest.nflverse import load_raw

REGULAR_SEASON = "REG"
FIRST_SEASON = 2009          # first season of nflverse injury reports
GROUPS = (
    "knee", "ankle_foot", "soft_tissue", "concussion", "hip_back", "shoulder",
    "achilles", "hand_arm", "torso", "leg_bone", "other",
)
NOT_INJURY = "none"
INJURED_REPORT = ("Out", "Doubtful")
# Reserve designations that are not injuries: suspended (R40), left squad or
# exempt (R27), non-football illness (R49).
NON_INJURY_RESERVE = ("R40", "R27", "R49")
# Body-part keywords, checked in order; the first match wins.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (NOT_INJURY, ("not injury", "illness", "ill ", "coach", "personal", "rest", "covid", "non-football", "non football",
                  "arrhythmia", "kidney", "spleen", "breath", "stomach")),
    ("concussion", ("concussion", "head")),
    ("achilles", ("achilles",)),
    ("knee", ("knee", "acl", "mcl", "pcl", "patella", "meniscus")),
    ("leg_bone", ("fibula", "tibia", "shin", "lower leg")),
    ("ankle_foot", ("ankle", "foot", "feet", "toe", "heel", "arch", "lisfranc")),
    ("soft_tissue", ("hamstring", "groin", "calf", "thigh", "quad", "oblique", "glute", "pectoral", "abdom",
                     "core", "hernia", "adductor")),
    ("hip_back", ("hip", "back", "neck", "pelvis", "spine", "stinger", "sternoclavicular")),
    ("shoulder", ("shoulder",)),
    ("hand_arm", ("hand", "finger", "thumb", "wrist", "elbow", "forearm", "collarbone", "clavicle", "bicep",
                  "tricep", "arm")),
    ("torso", ("rib", "chest")),
)


def body_group(text: str | None) -> str:
    """The injury group for a report's body part; `none` for non-injuries."""
    if text is None or not text.strip():
        return "other"
    t = f"{text.lower().strip()} "
    if t.startswith("player was ill"):
        return NOT_INJURY
    first = re.split(r"[/,]", t)[0] + " "
    for group, words in _KEYWORDS:
        if any(w in first for w in words):
            return group
    return "other"


@dataclass(frozen=True)
class InjuryRaw:
    injuries: pl.DataFrame
    rosters_weekly: pl.DataFrame

    @classmethod
    def from_cache(cls, pull_date: date | None = None) -> InjuryRaw:
        return cls(load_raw("injuries", pull_date=pull_date), load_raw("rosters_weekly", pull_date=pull_date))


def _franchise(col: str) -> pl.Expr:
    from dwar.features.situation import (
        FRANCHISE,  # local: situation imports player tables too
    )

    return pl.col(col).replace(FRANCHISE)


def team_weeks(schedules: pl.DataFrame) -> pl.DataFrame:
    """One row per team per regular-season game: season, week, team, gameday."""
    games = schedules.filter(pl.col("game_type") == REGULAR_SEASON) if "game_type" in schedules.columns else schedules
    if games.schema["gameday"] != pl.Date:
        games = games.with_columns(gameday=pl.col("gameday").str.to_date())
    return pl.concat([
        games.select("season", "week", "gameday", team=_franchise("home_team")),
        games.select("season", "week", "gameday", team=_franchise("away_team")),
    ]).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)).unique(["season", "week", "team"])


def injury_weeks(raw: InjuryRaw, schedules: pl.DataFrame, played: pl.DataFrame, as_of: date) -> pl.DataFrame:
    """One row per skill player per team game knowable by `as_of`, from the
    weekly rosters (a player on the team that week, active or not).

    `played` has player_id, season, week for games the player played.
    Columns: player_id, season, week, gameday, played, reserve, out, group."""
    tw = team_weeks(schedules).filter(pl.col("gameday") <= as_of)
    roster = (
        raw.rosters_weekly.filter(
            (pl.col("game_type") == REGULAR_SEASON) & pl.col("gsis_id").is_not_null()
            & pl.col("position").is_in(SKILL_POSITIONS) & pl.col("status").is_in(["ACT", "INA", "RES"])
        )
        .select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), team=_franchise("team"),
            player_id="gsis_id",
            reserve=(pl.col("status") == "RES")
            & ~pl.col("status_description_abbr").fill_null("").is_in(NON_INJURY_RESERVE),
        )
        .unique(["season", "week", "player_id"], keep="last", maintain_order=True)
        .join(tw, on=["season", "week", "team"], how="inner")
    )
    inj = raw.injuries
    part = pl.col("report_primary_injury")
    if "practice_primary_injury" in inj.columns:
        part = pl.coalesce(part, pl.col("practice_primary_injury"))
    reports = (
        inj.filter((pl.col("game_type") == REGULAR_SEASON) & pl.col("gsis_id").is_not_null())
        .select(
            pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), player_id="gsis_id",
            report_status="report_status", part=part,
        )
        .unique(["season", "week", "player_id"], keep="last", maintain_order=True)
    )
    groups = reports["part"].unique().to_list()
    group_map = pl.DataFrame({"part": groups, "group": [body_group(p) for p in groups]}, schema={"part": pl.Utf8, "group": pl.Utf8})
    reports = reports.join(group_map, on="part", how="left", nulls_equal=True)
    pl_ = played.select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), "player_id").unique().with_columns(
        played=pl.lit(True)
    )
    weeks = (
        roster.join(reports, on=["season", "week", "player_id"], how="left")
        .join(pl_, on=["season", "week", "player_id"], how="left")
        .with_columns(pl.col("played").fill_null(False))
        .sort("player_id", "season", "week")
        .with_columns(
            report_injury=(pl.col("group") != NOT_INJURY).fill_null(False),
            # A missed week takes the body part last reported for an injury,
            # carried across seasons (an injury in December can put a player
            # on reserve the next September); none known is "other".
            group=pl.when(pl.col("group") != NOT_INJURY).then("group").forward_fill().over("player_id").fill_null("other"),
        )
    )
    reported_out = pl.col("report_status").is_in(INJURED_REPORT).fill_null(False) & pl.col("report_injury")
    return weeks.select(
        "player_id", "season", "week", "gameday", "played", "reserve",
        out=~pl.col("played") & (pl.col("reserve") | reported_out),
        group="group",
    )


def injury_seasons(weeks: pl.DataFrame) -> pl.DataFrame:
    """Weeks missed to injury per player-season: `missed` in total and
    `missed_{group}` per body-part group."""
    out = weeks.filter("out")
    return (
        weeks.group_by("player_id", "season")
        .agg(team_games=pl.len())
        .join(
            out.group_by("player_id", "season").agg(
                missed=pl.len(), **{f"missed_{g}": (pl.col("group") == g).sum() for g in GROUPS}
            ),
            on=["player_id", "season"], how="left",
        )
        .fill_null(0)
        .sort("player_id", "season")
    )


def current_status(weeks: pl.DataFrame, season: int) -> pl.DataFrame:
    """Each player's status after the latest knowable team game of `season`:
    `injured` (missed that game to injury), `reserve`, and `streak` (games
    missed to injury in a row, ending with that one)."""
    w = weeks.filter(pl.col("season") == season).sort("player_id", "week")
    if w.is_empty():
        return pl.DataFrame(schema={"player_id": pl.Utf8, "injured": pl.Boolean, "reserve": pl.Boolean, "streak": pl.Int64})
    run = (pl.col("out") != pl.col("out").shift(1)).fill_null(True).cum_sum().over("player_id")
    return (
        w.with_columns(run=run)
        .with_columns(streak=pl.int_range(1, pl.len() + 1).over(["player_id", "run"]))
        .group_by("player_id", maintain_order=True)
        .last()
        .select("player_id", injured="out", reserve=pl.col("reserve") & pl.col("out"),
                streak=pl.when("out").then("streak").otherwise(0).cast(pl.Int64))
    )
