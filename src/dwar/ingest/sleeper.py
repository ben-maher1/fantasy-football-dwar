"""A Sleeper fantasy league, read from Sleeper's public API (api.sleeper.app).

Read-only and keyless: the API serves league settings, rosters, users,
weekly matchups (the schedule too: unplayed weeks come back paired with
0 points) and traded rookie picks for any league ID, plus one large
player list that Sleeper asks clients to fetch at most once a day. Nothing
here writes to a league.

Each pull is saved as raw JSON under ``data/raw/sleeper/{date}/`` (the player
list once per date, the league under its ID) and never committed, so a run
can be repeated from the exact bytes it read.

Sleeper player IDs are its own. `crosswalk` maps them to the gsis IDs the
board uses: Sleeper's own ``gsis_id`` where it has one, then the
``sleeper_id`` nflverse carries on its rosters, then ESPN IDs, then name
with birth date, then name with position. Anything left is reported.

Source: https://docs.sleeper.com (ben, 2026-11-22: team-specific values).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from dwar.config import data_dir

log = logging.getLogger(__name__)

SOURCE = "sleeper"
BASE_URL = "https://api.sleeper.app/v1"
LEAGUE_FILES = ("league", "rosters", "users", "traded_picks")
FANTASY_POSITIONS = ("QB", "RB", "WR", "TE")


def _get(path: str) -> Any:
    req = urllib.request.Request(f"{BASE_URL}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _root(root: Path | None) -> Path:
    return (root or data_dir() / "raw") / SOURCE


def pull(league_id: str, *, today: date | None = None, root: Path | None = None) -> Path:
    """Fetch a league (and today's player list, if not already saved) and
    write it as raw JSON. Returns the league's directory."""
    today = today or date.today()  # noqa: DTZ011 (the pull's own local date)
    day = _root(root) / today.isoformat()
    out = day / str(league_id)
    out.mkdir(parents=True, exist_ok=True)
    state = _get("/state/nfl")
    (out / "state.json").write_text(json.dumps(state))
    for name, path in zip(LEAGUE_FILES, ("", "/rosters", "/users", "/traded_picks"), strict=True):
        (out / f"{name}.json").write_text(json.dumps(_get(f"/league/{league_id}{path}")))
    league = json.loads((out / "league.json").read_text())
    settings = league.get("settings", {})
    last = int(settings.get("last_scored_leg") or 0)
    # Unplayed regular-season weeks too: they carry the schedule (pairings, 0 points).
    reg = max(int(settings.get("playoff_week_start") or 15) - 1, last)
    for w in range(1, reg + 1):
        (out / f"matchups_{w}.json").write_text(json.dumps(_get(f"/league/{league_id}/matchups/{w}")))
    players = day / "players.json"
    if not players.exists():
        players.write_text(json.dumps(_get("/players/nfl")))
    log.info("sleeper league %s: %d scored weeks saved to %s", league_id, last, out)
    return out


@dataclass(frozen=True)
class League:
    league: dict
    rosters: list[dict]
    users: list[dict]
    traded_picks: list[dict]
    matchups: dict[int, list[dict]] = field(default_factory=dict)
    players: dict[str, dict] = field(default_factory=dict)
    pull_date: date | None = None

    @property
    def season(self) -> int:
        return int(self.league["season"])


def load(league_id: str, pull_date: date | None = None, root: Path | None = None) -> League:
    """A saved pull: the given date's, or the latest one."""
    base = _root(root)
    dates = sorted(p.name for p in base.iterdir() if (p / str(league_id)).is_dir()) if base.exists() else []
    if not dates:
        raise FileNotFoundError(f"no saved Sleeper pull for league {league_id} under {base}: run pull() first")
    day = pull_date.isoformat() if pull_date else dates[-1]
    d = base / day / str(league_id)
    read = lambda name: json.loads((d / f"{name}.json").read_text())
    matchups = {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in d.glob("matchups_*.json")}
    players_file = base / day / "players.json"
    players = json.loads(players_file.read_text()) if players_file.exists() else {}
    return League(read("league"), read("rosters"), read("users"), read("traded_picks"), matchups, players,
                  date.fromisoformat(day))


# --- tables -----------------------------------------------------------------


def _owners(lg: League) -> dict[str, str]:
    return {u["user_id"]: u.get("display_name") or u["user_id"] for u in lg.users}


def standings(lg: League) -> pl.DataFrame:
    """One row per team: record and points so far (Sleeper's own totals)."""
    owners = _owners(lg)
    rows = []
    for r in lg.rosters:
        s = r.get("settings") or {}
        rows.append({
            "roster_id": int(r["roster_id"]), "owner": owners.get(r.get("owner_id"), str(r.get("owner_id"))),
            "division": int(s.get("division") or 0), "wins": int(s.get("wins") or 0),
            "losses": int(s.get("losses") or 0), "ties": int(s.get("ties") or 0),
            "points_for": float(s.get("fpts") or 0) + float(s.get("fpts_decimal") or 0) / 100,
        })
    return pl.DataFrame(rows).sort("roster_id")


def weekly_results(lg: League) -> pl.DataFrame:
    """Scored weeks: each team's points and its opponent (same matchup_id).
    Saved weeks past `last_scored_leg` are schedule only and left out."""
    last = int(lg.league.get("settings", {}).get("last_scored_leg") or max(lg.matchups, default=0))
    rows = []
    for w, games in sorted(lg.matchups.items()):
        if w > last:
            continue
        by_match: dict[int, list[dict]] = {}
        for g in games:
            if g.get("matchup_id") is not None:
                by_match.setdefault(int(g["matchup_id"]), []).append(g)
        for pair in by_match.values():
            for a in pair:
                opp = next((b for b in pair if b is not a), None)
                rows.append({"week": w, "roster_id": int(a["roster_id"]), "points": float(a.get("points") or 0),
                             "opponent": int(opp["roster_id"]) if opp else None})
    schema = {"week": pl.Int64, "roster_id": pl.Int64, "points": pl.Float64, "opponent": pl.Int64}
    return pl.DataFrame(rows, schema=schema).sort(["week", "roster_id"])


def rostered(lg: League) -> pl.DataFrame:
    """Every rostered player: team, Sleeper ID and where he sits (active,
    reserve = IR, taxi)."""
    owners = _owners(lg)
    rows = []
    for r in lg.rosters:
        reserve, taxi = set(r.get("reserve") or []), set(r.get("taxi") or [])
        for pid in r.get("players") or []:
            p = lg.players.get(pid, {})
            rows.append({
                "roster_id": int(r["roster_id"]), "owner": owners.get(r.get("owner_id"), str(r.get("owner_id"))),
                "sleeper_id": str(pid), "name": p.get("full_name") or pid, "position": p.get("position"),
                "nfl_team": p.get("team"),
                "slot": "reserve" if pid in reserve else "taxi" if pid in taxi else "active",
            })
    return pl.DataFrame(rows).sort(["roster_id", "position", "name"])


def pick_owners(lg: League, seasons: list[int] | None = None, rounds: int | None = None) -> pl.DataFrame:
    """Who owns each future rookie pick: every team starts with its own,
    then Sleeper's traded picks move them (`roster_id` is the team the pick
    came from, `owner_id` the team that holds it now). Default seasons are
    the next three (Sleeper trades three years ahead)."""
    rounds = rounds or int(lg.league.get("settings", {}).get("draft_rounds") or 5)
    seasons = seasons or [lg.season + k for k in (1, 2, 3)]
    ids = sorted(int(r["roster_id"]) for r in lg.rosters)
    owner = {(s, k, t): t for s in seasons for k in range(1, rounds + 1) for t in ids}
    for tp in lg.traded_picks:
        key = (int(tp["season"]), int(tp["round"]), int(tp["roster_id"]))
        if key in owner:
            owner[key] = int(tp["owner_id"])
    return pl.DataFrame(
        [{"season": s, "round": k, "original_roster_id": t, "owner_roster_id": o} for (s, k, t), o in owner.items()]
    ).sort(["season", "round", "original_roster_id"])


# --- player IDs -------------------------------------------------------------

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str | None) -> str:
    """Lower case, letters only, suffixes dropped: "D.J. Moore Jr." -> "djmoore"."""
    s = _SUFFIX.sub("", (name or "").lower().replace(".", ""))
    return re.sub(r"[^a-z]", "", s)


def crosswalk(players: dict[str, dict], nfl_rosters: pl.DataFrame, nfl_players: pl.DataFrame,
              sleeper_ids: list[str] | None = None) -> pl.DataFrame:
    """Sleeper ID -> gsis ID for fantasy positions, with the method that
    matched. `nfl_rosters`: nflverse rosters (any seasons) with gsis_id and
    sleeper_id; `nfl_players`: nflverse players with gsis_id, display_name,
    position, birth_date and espn_id. Unmatched players get a null gsis_id."""
    ids = sleeper_ids if sleeper_ids is not None else [k for k, p in players.items()
                                                       if p.get("position") in FANTASY_POSITIONS]
    by_sleeper = {str(s): g for s, g in nfl_rosters.select(
        pl.col("sleeper_id").cast(pl.Utf8), "gsis_id").drop_nulls().unique("sleeper_id").iter_rows()}
    np_ = nfl_players.filter(pl.col("gsis_id").is_not_null())
    by_espn = {str(e): g for e, g in np_.select(pl.col("espn_id").cast(pl.Utf8), "gsis_id").drop_nulls()
               .unique("espn_id", keep="none").iter_rows()}
    named = np_.select("gsis_id", key=pl.col("display_name").map_elements(norm_name, return_dtype=pl.Utf8),
                       pos="position", bd=pl.col("birth_date").cast(pl.Utf8).str.slice(0, 10))
    by_name_bd = {(k, b): g for g, k, _, b in named.drop_nulls("bd").unique(["key", "bd"], keep="none").iter_rows()}
    by_name_pos = {(k, p): g for g, k, p, _ in named.unique(["key", "pos"], keep="none").iter_rows()}
    known = set(np_["gsis_id"].to_list()) | set(by_sleeper.values())
    rows = []
    for sid in ids:
        p = players.get(sid, {})
        key, pos = norm_name(p.get("full_name")), p.get("position")
        g, how = p.get("gsis_id"), "sleeper_gsis"
        g = g.strip() if isinstance(g, str) and g.strip() in known else None
        for cand, method in ((by_sleeper.get(str(sid)), "nflverse_sleeper_id"),
                             (by_espn.get(str(p.get("espn_id"))) if p.get("espn_id") else None, "espn_id"),
                             (by_name_bd.get((key, (p.get("birth_date") or "")[:10])), "name_birth_date"),
                             (by_name_pos.get((key, pos)), "name_position")):
            if g is None and cand is not None:
                g, how = cand, method
        rows.append({"sleeper_id": str(sid), "gsis_id": g, "method": how if g else None,
                     "name": p.get("full_name"), "position": pos})
    return pl.DataFrame(rows, schema={"sleeper_id": pl.Utf8, "gsis_id": pl.Utf8, "method": pl.Utf8,
                                      "name": pl.Utf8, "position": pl.Utf8})
