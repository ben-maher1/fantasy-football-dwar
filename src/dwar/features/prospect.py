"""College prospect profiles for NFL players (plan "prospect profiles for
young WRs", approved by ben 2026-11-08).

Eight metrics from college seasons before a player's NFL draft (for
undrafted players, before his first NFL season), so every value is known on
draft day:

- ``breakout_age``: age on Sept 1 of the first season with a dominator of
  20%+; never broke out = age in his last college season + 1.
- ``dominator``: best season's average of his share of team receiving yards
  and team receiving TDs.
- ``team_sp``: his team's SP+ rating in that best season (FBS only).
- ``yptpa``: best season's receiving yards per team pass attempt (FBS only).
- ``yptpa_age``: best season's yards per team pass attempt minus the
  average for NFL WR prospects his age that season.
- ``final_share``: share of team receptions in his last college season.
- ``early_declare``: fewer than four college seasons with receiving stats.
- ``recruit``: his high-school recruiting rating (247 composite, 0-1).

Team receiving totals are the sums over the team's players in CFBD's
season stats, so FCS seasons get a dominator too; team pass attempts and
SP+ come from FBS team tables only.

College players are matched to NFL players by normalized name, college and
draft year (`match`); matches the code is not sure of are listed by
`match_report` and left missing. A missing metric is a flag in the model
(model/data.py standardize_with_flags), so it adds nothing.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import date

import polars as pl

from dwar.features.player_tables import DAYS_PER_YEAR

METRICS = ("breakout_age", "dominator", "team_sp", "yptpa", "yptpa_age", "final_share", "early_declare", "recruit")
BREAKOUT_DOMINATOR = 0.20
MIN_TEAM_PASS_ATT = 100     # team-seasons with fewer attempts give no yards per attempt
MAX_COLLEGE_SEASONS = 6     # college seasons looked at before the draft
COLLEGE_MATCH = 0.80        # normalized-college similarity that counts as the same school
# CFBD player season stats start in 2004, so a player drafted before 2008
# may have college seasons we cannot see: no breakout age or season count.
FIRST_FULL_DRAFT = 2008
AGE_REF_RANGE = (18, 24)    # ages outside are pooled into the nearest end for yptpa_age

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
COLLEGE_ALIASES = {
    "southern california": "usc", "louisiana state": "lsu", "texas christian": "tcu", "brigham young": "byu",
    "central florida": "ucf", "southern methodist": "smu", "mississippi": "ole miss",
    "alabama birmingham": "uab", "nevada las vegas": "unlv", "texas el paso": "utep",
    "texas san antonio": "utsa", "pittsburgh": "pitt", "north carolina state": "nc state",
    "florida international": "fiu", "louisiana lafayette": "louisiana", "louisiana monroe": "ul monroe",
    "middle tennessee state": "middle tennessee", "hawaii": "hawai i", "miami fl": "miami",
    "miami florida": "miami", "miami ohio": "miami oh", "massachusetts": "umass",
    "connecticut": "uconn", "southern mississippi": "southern miss", "texas a m": "texas a m",
    "boston col": "boston college", "virginia tech": "virginia tech", "san jose state": "san jose state",
    "app state": "appalachian state", "army": "army", "navy": "navy",
}


def norm_name(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())


def norm_college(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bst\.?$", "state", s.strip())
    s = re.sub(r"\bst\.", "state", s)
    s = re.sub(r"\bcol\.", "college", s)
    s = re.sub(r"\bu\.", "university", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\b(university|univ)\b", "", s)
    s = " ".join(s.split())
    return COLLEGE_ALIASES.get(s, s)


def college_similarity(a: str | None, b: str | None) -> float:
    x, y = norm_college(a), norm_college(b)
    if not x or not y:
        return 0.0
    return 1.0 if x == y else difflib.SequenceMatcher(None, x, y).ratio()


def combine_sources(games: pl.DataFrame, receiving: pl.DataFrame) -> pl.DataFrame:
    """Player-team-seasons from FBS box scores, plus the season-stats
    endpoint's rows for team-seasons the box scores do not cover (FCS teams;
    box scores only hold their games against FBS teams)."""
    cols = ["season", "cfbd_id", "player", "team", "conference", "rec", "yds", "td"]
    # A team-season counts as covered when its box scores hold most of a season (FCS teams appear for 1-2 games).
    box_games = games.group_by("season", "team").agg(g=pl.col("games").max())
    covered = box_games.filter(pl.col("g") >= 6).select("season", "team")
    other = receiving.join(covered, on=["season", "team"], how="anti")
    box = games.join(covered, on=["season", "team"], how="semi")
    return pl.concat([box.select(cols), other.select(cols)], how="vertical_relaxed")


def college_seasons(receiving: pl.DataFrame, team: pl.DataFrame, sp: pl.DataFrame) -> pl.DataFrame:
    """One row per CFBD player-team-season with his shares of the team's
    receiving and the team's pass attempts and SP+. `receiving` is
    `combine_sources`' output (or any table with its columns)."""
    totals = receiving.group_by("season", "team").agg(
        team_rec=pl.col("rec").sum(), team_yds=pl.col("yds").sum(), team_td=pl.col("td").sum()
    )
    return (
        receiving.join(totals, on=["season", "team"], how="left")
        .join(team.select("season", "team", "pass_att"), on=["season", "team"], how="left")
        .join(sp.select("season", "team", "sp"), on=["season", "team"], how="left")
        .with_columns(
            yds_share=pl.when(pl.col("team_yds") > 0).then(pl.col("yds") / pl.col("team_yds")),
            td_share=pl.when(pl.col("team_td") > 0).then(pl.col("td") / pl.col("team_td")),
            rec_share=pl.when(pl.col("team_rec") > 0).then(pl.col("rec") / pl.col("team_rec")),
            yptpa=pl.when(pl.col("pass_att") >= MIN_TEAM_PASS_ATT).then(pl.col("yds") / pl.col("pass_att")),
        )
        .with_columns(dominator=pl.coalesce((pl.col("yds_share") + pl.col("td_share")) / 2, pl.col("yds_share")))
        .sort("cfbd_id", "season", "team")
    )


def nfl_prospects(players: pl.DataFrame, draft_picks: pl.DataFrame, positions: tuple[str, ...]) -> pl.DataFrame:
    """player_id, name, college, entry_year (draft year, else first NFL
    season) and birth_date for NFL players at `positions`."""
    picks = (
        draft_picks.filter(pl.col("gsis_id").is_not_null())
        .select(player_id="gsis_id", draft_year="season", draft_college="college", draft_position="position")
        .unique("player_id", keep="first")
    )
    p = players.filter(pl.col("gsis_id").is_not_null()).select(
        player_id="gsis_id", name="display_name", college_name="college_name", position="position",
        birth_date=pl.col("birth_date").str.to_date(strict=False), rookie_season="rookie_season",
        p_draft_year="draft_year",
    )
    return (
        p.join(picks, on="player_id", how="left")
        .filter(pl.col("position").is_in(positions) | pl.col("draft_position").is_in(positions))
        .select(
            "player_id", "name", "birth_date",
            college=pl.coalesce("draft_college", "college_name"),
            entry_year=pl.coalesce("draft_year", "p_draft_year", "rookie_season").cast(pl.Int32),
        )
        .filter(pl.col("entry_year").is_not_null())
    )


SURE = ("name+college", "initial+college", "lastname+college")


def match(prospects: pl.DataFrame, seasons: pl.DataFrame) -> pl.DataFrame:
    """player_id -> cfbd_id with how it matched. `how` is one of
    name+college, initial+college (sure), or ambiguous / name only /
    none (not used)."""
    cs = seasons.select("cfbd_id", "player", "team", "season").with_columns(
        nname=pl.col("player").map_elements(norm_name, return_dtype=pl.Utf8)
    )
    by_name: dict[str, list[tuple]] = {}
    by_last: dict[str, list[tuple]] = {}
    for cid, nname, team, season in cs.select("cfbd_id", "nname", "team", "season").iter_rows():
        by_name.setdefault(nname, []).append((cid, team, season, nname))
        if nname:
            by_last.setdefault(nname.split()[-1], []).append((cid, team, season, nname))

    def best(cands: list[tuple], college: str | None, entry: int) -> dict[str, tuple[float, int]]:
        ids: dict[str, tuple[float, int]] = {}
        for cid, team, season, _ in cands:
            if not (entry - MAX_COLLEGE_SEASONS <= season < entry):
                continue
            sim, last = ids.get(cid, (0.0, 0))
            ids[cid] = (max(sim, college_similarity(team, college)), max(last, season))
        return ids

    rows = []
    for pid, name, college, entry in prospects.select("player_id", "name", "college", "entry_year").iter_rows():
        nn = norm_name(name)
        ids = best(by_name.get(nn, []), college, entry)
        sure = {k: v for k, v in ids.items() if v[0] >= COLLEGE_MATCH}
        how, cid = "none", None
        if len(sure) == 1:
            how, cid = "name+college", next(iter(sure))
        elif len(sure) > 1:
            # Same name at the same school: take the one whose college career ends just before the draft.
            ends = sorted(sure.items(), key=lambda kv: -kv[1][1])
            how, cid = ("name+college", ends[0][0]) if ends[0][1][1] > ends[1][1][1] else ("ambiguous", None)
        elif ids:
            how = "name only"
        else:
            # Nicknames (D.J. / Hollywood): same last name and first initial at the same school.
            # Failing that (Tank / Nathaniel Dell), the only player with his
            # last name at his school whose college career ends just before the draft.
            parts = nn.split()
            if parts:
                last = by_last.get(parts[-1], [])
                for how_, cands in (("initial+college", [c for c in last if c[3][:1] == parts[0][:1]]),
                                    ("lastname+college", last)):
                    near = {k: v for k, v in best(cands, college, entry).items()
                            if v[0] >= COLLEGE_MATCH and v[1] >= entry - 2}
                    if len(near) == 1:
                        how, cid = how_, next(iter(near))
                        break
                    if len(near) > 1:
                        how = "ambiguous"
                        break
        rows.append((pid, cid, how))
    return pl.DataFrame(rows, schema={"player_id": pl.Utf8, "cfbd_id": pl.Utf8, "how": pl.Utf8}, orient="row")


def _age_on_sept1(season: pl.Expr, birth: pl.Expr) -> pl.Expr:
    return (pl.date(season, 9, 1) - birth).dt.total_days() / DAYS_PER_YEAR


def metrics(
    prospects: pl.DataFrame, matches: pl.DataFrame, seasons: pl.DataFrame, recruits: pl.DataFrame,
    as_of: date,
) -> pl.DataFrame:
    """The eight metrics per player_id (null where unknown). The age
    reference for `yptpa_age` uses only players who entered the NFL by
    `as_of` (no lookahead), and only college seasons before each player's
    entry year are read."""
    m = prospects.join(matches.filter(pl.col("cfbd_id").is_not_null()), on="player_id", how="inner")
    s = (
        m.select("player_id", "cfbd_id", "entry_year", "birth_date")
        .join(seasons, on="cfbd_id", how="inner")
        .filter((pl.col("season") < pl.col("entry_year")) & (pl.col("season") >= pl.col("entry_year") - MAX_COLLEGE_SEASONS))
        .with_columns(age=_age_on_sept1(pl.col("season"), pl.col("birth_date")))
    )
    lo, hi = AGE_REF_RANGE
    s = s.with_columns(age_bin=pl.col("age").floor().clip(lo, hi))
    entered = as_of.year if as_of >= date(as_of.year, 5, 1) else as_of.year - 1
    ref = (
        s.filter((pl.col("entry_year") <= entered) & pl.col("yptpa").is_not_null())
        .group_by("age_bin").agg(ref_yptpa=pl.col("yptpa").mean())
    )
    s = s.join(ref, on="age_bin", how="left").with_columns(yptpa_age=pl.col("yptpa") - pl.col("ref_yptpa"))

    # A season split between two schools counts once (its shares summed would mix teams: take the bigger one).
    per_season = s.sort("dominator", descending=True, nulls_last=True).unique(["player_id", "season"], keep="first")
    best_dom = per_season.sort("dominator", descending=True, nulls_last=True).unique("player_id", keep="first").select(
        "player_id", dominator="dominator", team_sp="sp"
    )
    first_bo = (
        per_season.filter(pl.col("dominator") >= BREAKOUT_DOMINATOR).sort("season")
        .unique("player_id", keep="first").select("player_id", bo_age="age")
    )
    agg = per_season.group_by("player_id").agg(
        n_seasons=pl.col("season").n_unique(),
        last_season=pl.col("season").max(),
        last_age=pl.col("age").sort_by("season").last(),
        yptpa=pl.col("yptpa").max(),
        yptpa_age=pl.col("yptpa_age").max(),
        final_share=pl.col("rec_share").sort_by("season").last(),
        entry_year=pl.col("entry_year").first(),
    )
    full = pl.col("entry_year") >= FIRST_FULL_DRAFT
    out = (
        agg.join(best_dom, on="player_id", how="left").join(first_bo, on="player_id", how="left")
        .with_columns(
            breakout_age=pl.when(full).then(pl.coalesce("bo_age", pl.col("last_age") + 1)),
            early_declare=pl.when(full).then((pl.col("n_seasons") < 4).cast(pl.Float64)),
        )
    )
    # Recruiting rating: CFBD's athlete id links recruits to their college stats.
    rec = recruits.filter(pl.col("cfbd_id").is_not_null() & pl.col("rating").is_not_null()).group_by("cfbd_id").agg(
        recruit=pl.col("rating").max(), class_year=pl.col("class_year").min()
    )
    out = out.join(m.select("player_id", "cfbd_id", "name", "college"), on="player_id", how="left").join(
        rec, on="cfbd_id", how="left")
    out = out.with_columns(recruit=pl.when(pl.col("class_year") < pl.col("entry_year")).then("recruit"))
    # Failing the id link, the one recruit with his name in a class 2-6 years
    # before his entry who signed with a school he played for.
    by_name: dict[str, list[tuple]] = {}
    for name, cls, school, rating in recruits.filter(pl.col("rating").is_not_null()).select(
            "name", "class_year", "committed_to", "rating").iter_rows():
        by_name.setdefault(norm_name(name), []).append((cls, school, rating))
    teams = per_season.group_by("player_id").agg(pl.col("team").unique())
    teams = dict(zip(teams["player_id"], teams["team"].to_list()))
    fill = {}
    for pid, name, college, entry in out.filter(pl.col("recruit").is_null()).select(
            "player_id", "name", "college", "entry_year").iter_rows():
        schools = set(teams.get(pid, [])) | {college}
        hits = {(cls, rating) for cls, school, rating in by_name.get(norm_name(name), [])
                if entry - MAX_COLLEGE_SEASONS <= cls <= entry - 2
                and max(college_similarity(school, t) for t in schools) >= COLLEGE_MATCH}
        if len(hits) == 1:
            fill[pid] = next(iter(hits))[1]
    if fill:
        out = out.with_columns(recruit=pl.coalesce(
            "recruit", pl.col("player_id").replace_strict(fill, default=None, return_dtype=pl.Float64)))
    return out.select("player_id", *METRICS).sort("player_id")


def match_report(prospects: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Players the matcher was not sure of, for review."""
    return prospects.join(matches, on="player_id").filter(~pl.col("how").is_in(SURE))


def prospect_table(as_of: date, positions: tuple[str, ...] = ("WR",)) -> pl.DataFrame:
    """The metrics for every NFL player at `positions`, from the cached CFBD
    and nflverse pulls (ingest/cfbd.py, ingest/nflverse.py)."""
    from dwar.ingest import cfbd
    from dwar.ingest.nflverse import load_raw

    seasons = college_seasons(combine_sources(cfbd.load("games"), cfbd.load("receiving")), cfbd.load("team"), cfbd.load("sp"))
    prospects = nfl_prospects(load_raw("players"), load_raw("draft_picks"), positions)
    matches = match(prospects, seasons)
    return metrics(prospects, matches, seasons, cfbd.load("recruits"), as_of)
