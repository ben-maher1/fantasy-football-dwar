"""College football data from the collegefootballdata.com (CFBD) API.

Used for WR prospect profiles (features/prospect.py). Four datasets, one
API call per season each:

- ``receiving``: player season receiving stats (receptions, yards, TDs),
  FBS and FCS. This endpoint misses players (Will Fuller's 2015 at Notre
  Dame is absent, and 2004-2011 are thin), so FBS seasons come from
  ``games`` instead and this one fills in FCS teams.
- ``games``: FBS game box-score receiving, summed per player-team-season
  (one call per week, about 18 a season).
- ``team``: team season totals (pass attempts, completions, yards, TDs).
- ``sp``: team SP+ ratings (FBS only).
- ``recruits``: high-school recruit ratings, every position, by class year.

The key is read from the ``CFBD_API_KEY`` environment variable and never
stored. CFBD's terms allow caching and model training but not republishing
the raw data, so pulls are cached under ``data/raw/cfbd/`` like every other
source and never committed. A completed season's file is never re-pulled,
which keeps the monthly call count small (the free tier allows 1,000).

Source: https://collegefootballdata.com (terms: /terms).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from dwar.config import data_dir
from dwar.ingest.cache import latest_file, write_files

log = logging.getLogger(__name__)

SOURCE = "cfbd"
BASE_URL = "https://api.collegefootballdata.com"
FIRST_SEASON = 2004   # player season stats start here
FIRST_CLASS = 2000    # recruiting classes start here
TEAM_STATS = ("passAttempts", "passCompletions", "netPassingYards", "passingTDs", "games")

DATASETS = {
    "receiving": ("/stats/player/season", {"category": "receiving"}),
    "team": ("/stats/season", {}),
    "sp": ("/ratings/sp", {}),
    "recruits": ("/recruiting/players", {"classification": "HighSchool"}),
    "games": ("/games/players", {"category": "receiving"}),
}
MAX_WEEK = 17  # regular-season weeks asked for; empty weeks are skipped


class MissingKeyError(RuntimeError):
    """CFBD_API_KEY is not set."""


def _get(path: str, params: dict) -> list[dict]:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise MissingKeyError("Set CFBD_API_KEY (collegefootballdata.com API key) in the environment")
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        remaining = resp.headers.get("X-CallLimit-Remaining")
        body = json.loads(resp.read())
    log.info("cfbd %s %s: %d rows (calls left this month: %s)", path, params, len(body), remaining)
    return body


def parse_receiving(rows: list[dict]) -> pl.DataFrame:
    """Long stat rows -> one row per player-team-season with rec, yds, td."""
    schema = {"season": pl.Int32, "cfbd_id": pl.Utf8, "player": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8,
              "conference": pl.Utf8, "stat_type": pl.Utf8, "stat": pl.Float64}
    if not rows:
        return pl.DataFrame(schema={k: v for k, v in schema.items() if k not in ("stat_type", "stat")}
                            | {"rec": pl.Float64, "yds": pl.Float64, "td": pl.Float64})
    df = pl.DataFrame(
        {
            "season": [r["season"] for r in rows], "cfbd_id": [str(r["playerId"]) for r in rows],
            "player": [r["player"] for r in rows], "position": [r.get("position") for r in rows],
            "team": [r["team"] for r in rows], "conference": [r.get("conference") for r in rows],
            "stat_type": [r["statType"] for r in rows], "stat": [float(r["stat"]) for r in rows],
        },
        schema=schema,
    ).filter(pl.col("stat_type").is_in(["REC", "YDS", "TD"]))
    wide = df.pivot(on="stat_type", index=["season", "cfbd_id", "player", "position", "team", "conference"],
                    values="stat", aggregate_function="sum")
    for c in ("REC", "YDS", "TD"):
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(c))
    return wide.rename({"REC": "rec", "YDS": "yds", "TD": "td"}).select(
        "season", "cfbd_id", "player", "position", "team", "conference", pl.col("rec", "yds", "td").fill_null(0.0)
    ).sort("season", "team", "cfbd_id")


def parse_team(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(
        {"season": [r["season"] for r in rows], "team": [r["team"] for r in rows],
         "conference": [r.get("conference") for r in rows], "stat": [r["statName"] for r in rows],
         "value": [float(r["statValue"]) for r in rows]},
        schema={"season": pl.Int32, "team": pl.Utf8, "conference": pl.Utf8, "stat": pl.Utf8, "value": pl.Float64},
    ).filter(pl.col("stat").is_in(TEAM_STATS))
    wide = df.pivot(on="stat", index=["season", "team", "conference"], values="value", aggregate_function="sum")
    for c in TEAM_STATS:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, pl.Float64).alias(c))
    return wide.select(
        "season", "team", "conference",
        pass_att=pl.col("passAttempts"), pass_cmp=pl.col("passCompletions"),
        pass_yds=pl.col("netPassingYards"), pass_td=pl.col("passingTDs"), games=pl.col("games"),
    ).sort("season", "team")


def parse_sp(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {"season": [r["year"] for r in rows], "team": [r["team"] for r in rows],
         "conference": [r.get("conference") for r in rows],
         "sp": [None if r.get("rating") is None else float(r["rating"]) for r in rows]},
        schema={"season": pl.Int32, "team": pl.Utf8, "conference": pl.Utf8, "sp": pl.Float64},
    ).filter(pl.col("team") != "nationalAverages").sort("season", "team")


def parse_recruits(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {"class_year": [r["year"] for r in rows],
         "cfbd_id": [None if r.get("athleteId") is None else str(r["athleteId"]) for r in rows],
         "name": [r["name"] for r in rows], "position": [r.get("position") for r in rows],
         "committed_to": [r.get("committedTo") for r in rows], "stars": [r.get("stars") for r in rows],
         "rating": [None if r.get("rating") is None else float(r["rating"]) for r in rows]},
        schema={"class_year": pl.Int32, "cfbd_id": pl.Utf8, "name": pl.Utf8, "position": pl.Utf8,
                "committed_to": pl.Utf8, "stars": pl.Int32, "rating": pl.Float64},
    ).sort("class_year", "name")


def parse_games(rows: list[dict], season: int) -> pl.DataFrame:
    """Box-score rows -> one row per player-team-season with rec, yds, td
    and games with a reception line."""
    out: dict[tuple[str, str], dict] = {}
    for g in rows:
        for t in g.get("teams", []):
            for c in t.get("categories", []):
                if c.get("name") != "receiving":
                    continue
                seen = set()
                for ty in c.get("types", []):
                    key = {"REC": "rec", "YDS": "yds", "TD": "td"}.get(ty["name"])
                    if key is None:
                        continue
                    for a in ty.get("athletes", []):
                        if str(a["id"]).startswith("-") or a["name"] in ("Team", " Team"):
                            continue
                        r = out.setdefault((str(a["id"]), t["team"]), {
                            "season": season, "cfbd_id": str(a["id"]), "player": a["name"], "position": None,
                            "team": t["team"], "conference": t.get("conference"), "rec": 0.0, "yds": 0.0, "td": 0.0,
                            "games": 0,
                        })
                        try:
                            r[key] += float(a["stat"])
                        except (TypeError, ValueError):
                            pass
                        if a["id"] not in seen:
                            seen.add(a["id"])
                            r["games"] += 1 if key == "rec" else 0
    schema = {"season": pl.Int32, "cfbd_id": pl.Utf8, "player": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8,
              "conference": pl.Utf8, "rec": pl.Float64, "yds": pl.Float64, "td": pl.Float64, "games": pl.Int32}
    return pl.DataFrame(list(out.values()), schema=schema).sort("season", "team", "cfbd_id")


def _get_games(season: int) -> list[dict]:
    path, params = DATASETS["games"]
    rows: list[dict] = []
    for week in range(1, MAX_WEEK + 1):
        got = _get(path, params | {"year": season, "week": week, "seasonType": "regular"})
        if not got and week > 12:
            break
        rows += got
    for week in range(1, 6):  # bowls are postseason week 1; later playoff rounds follow
        got = _get(path, params | {"year": season, "week": week, "seasonType": "postseason"})
        if not got:
            break
        rows += got
    return rows


PARSERS = {"receiving": parse_receiving, "team": parse_team, "sp": parse_sp, "recruits": parse_recruits}


def _file(season: int) -> str:
    return f"season={season}.parquet"


def _final(season: int, pulled_at: datetime) -> bool:
    """A college season is complete once January after it has passed."""
    return pulled_at.date() > datetime(season + 1, 1, 31).date()


def pull(dataset: str, seasons: list[int], *, now: datetime | None = None, root: Path | None = None) -> list[Path]:
    """Pull seasons not yet cached (or cached before the season was over)."""
    path, params = DATASETS[dataset]
    root = root or data_dir() / "raw"
    now = now or datetime.now(UTC)
    written = []
    for season in seasons:
        cached = latest_file(root, SOURCE, dataset, _file(season))
        if cached is not None and _final(season, cached.pulled_at):
            continue
        if dataset == "games":
            df = parse_games(_get_games(season), season)
        else:
            df = PARSERS[dataset](_get(path, params | {"year": season}))
        written += write_files(root, SOURCE, dataset, {_file(season): df}, now, "cfbd-api-v2")
    return written


def pull_all(last_season: int, **kwargs) -> dict[str, list[Path]]:
    seasons = list(range(FIRST_SEASON, last_season + 1))
    out = {d: pull(d, seasons, **kwargs) for d in ("receiving", "team", "sp", "games")}
    out["recruits"] = pull("recruits", list(range(FIRST_CLASS, last_season + 1)), **kwargs)
    return out


def load(dataset: str, root: Path | None = None) -> pl.DataFrame:
    """Every cached season of `dataset`, newest pull of each."""
    root = root or data_dir() / "raw"
    base = root / SOURCE
    files: dict[str, Path] = {}
    if base.exists():
        for name in sorted({p.name for p in base.glob(f"*/{dataset}/season=*.parquet")}):
            cached = latest_file(root, SOURCE, dataset, name)
            if cached is not None:
                files[name] = cached.path
    if not files:
        raise FileNotFoundError(f"No cached CFBD {dataset}; run ingest.cfbd.pull_all() first")
    return pl.concat([pl.read_parquet(p) for p in files.values()], how="diagonal_relaxed")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print({k: len(v) for k, v in pull_all(int(sys.argv[1]) if len(sys.argv) > 1 else 2025).items()})
