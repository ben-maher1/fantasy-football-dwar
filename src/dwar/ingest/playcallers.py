"""Offensive play-callers by team and game, from Sam Hoppen's public table.

nflverse lists head coaches only. Hoppen's NFL_public repository keeps one
row per team-game from 1999 with the offensive and defensive play-caller and
the head coach, and pre-fills scheduled games of the current season. The
repository has no license file, so the table is cached under `data/raw/` like
every other pull and never committed.

Source: https://github.com/samhoppen/NFL_public (data/all_playcallers.csv).
"""

from __future__ import annotations

import io
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from dwar.config import data_dir
from dwar.ingest.cache import SchemaChangedError, latest_file, latest_schema, schema_of, write_files

SOURCE = "playcallers"
DATASET = "playcallers"
FILE = "all.parquet"
URL = "https://raw.githubusercontent.com/samhoppen/NFL_public/main/data/all_playcallers.csv"
COLUMNS = ("season", "week", "team", "game_id", "off_play_caller", "def_play_caller", "head_coach")
DEFAULT_MAX_AGE = timedelta(days=7)


def parse(csv_bytes: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(csv_bytes), schema_overrides={"season": pl.Int32, "week": pl.Int32})
    missing = sorted(set(COLUMNS) - set(df.columns))
    if missing:
        raise SchemaChangedError(f"play-caller table no longer has {missing}")
    return df.select(COLUMNS)


def pull(
    *, now: datetime | None = None, max_age: timedelta = DEFAULT_MAX_AGE, root: Path | None = None,
    allow_schema_change: bool = False,
) -> list[Path]:
    root = root or data_dir() / "raw"
    now = now or datetime.now(UTC)
    cached = latest_file(root, SOURCE, DATASET, FILE)
    if cached is not None and now - cached.pulled_at <= max_age:
        return []
    with urllib.request.urlopen(URL, timeout=60) as resp:
        df = parse(resp.read())
    previous = latest_schema(root, SOURCE, DATASET)
    if previous is not None and previous != schema_of(df) and not allow_schema_change:
        raise SchemaChangedError("play-caller table schema changed; review, then allow_schema_change=True")
    return write_files(root, SOURCE, DATASET, {FILE: df}, now, "github-main")


def load(root: Path | None = None) -> pl.DataFrame:
    root = root or data_dir() / "raw"
    cached = latest_file(root, SOURCE, DATASET, FILE)
    if cached is None:
        raise FileNotFoundError("No cached play-caller table; run ingest.playcallers.pull() first")
    return pl.read_parquet(cached.path)
