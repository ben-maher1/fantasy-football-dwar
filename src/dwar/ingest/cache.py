"""Dated parquet snapshots of raw pulls.

Layout: ``{root}/{source}/{pull_date}/{dataset}/{partition}.parquet`` with a
``_manifest.json`` beside the files recording when each file was pulled, its
row count, the package version, and the schema. Earlier snapshot directories
are never rewritten, so a past run can be reproduced from the exact bytes it
read. The one exception is a same-day refresh of a file in today's directory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

MANIFEST = "_manifest.json"


class SchemaChangedError(RuntimeError):
    """A fresh pull's columns or dtypes differ from the previous snapshot."""


@dataclass(frozen=True)
class CachedFile:
    path: Path
    pull_date: date
    pulled_at: datetime
    rows: int


def schema_of(df: pl.DataFrame) -> dict[str, str]:
    return {name: str(dtype) for name, dtype in df.schema.items()}


def schema_hash(schema: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]


def dataset_dir(root: Path, source: str, pull_date: date, dataset: str) -> Path:
    return root / source / pull_date.isoformat() / dataset


def read_manifest(directory: Path) -> dict:
    path = directory / MANIFEST
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def snapshot_dates(root: Path, source: str, dataset: str) -> list[date]:
    """Pull dates that hold this dataset, oldest first."""
    base = root / source
    if not base.exists():
        return []
    dates = []
    for child in base.iterdir():
        if (child / dataset / MANIFEST).exists():
            dates.append(date.fromisoformat(child.name))
    return sorted(dates)


def latest_file(
    root: Path, source: str, dataset: str, filename: str, on_or_before: date | None = None
) -> CachedFile | None:
    """Newest cached copy of one partition file, optionally pinned to a pull date."""
    for d in reversed(snapshot_dates(root, source, dataset)):
        if on_or_before is not None and d > on_or_before:
            continue
        directory = dataset_dir(root, source, d, dataset)
        entry = read_manifest(directory).get("files", {}).get(filename)
        if entry is not None:
            return CachedFile(
                path=directory / filename,
                pull_date=d,
                pulled_at=datetime.fromisoformat(entry["pulled_at"]),
                rows=entry["rows"],
            )
    return None


def latest_schema(root: Path, source: str, dataset: str) -> dict[str, str] | None:
    dates = snapshot_dates(root, source, dataset)
    if not dates:
        return None
    return read_manifest(dataset_dir(root, source, dates[-1], dataset)).get("schema")


def write_files(
    root: Path,
    source: str,
    dataset: str,
    frames: dict[str, pl.DataFrame],
    pulled_at: datetime,
    package_version: str,
) -> list[Path]:
    """Write partition files into today's snapshot and update its manifest."""
    directory = dataset_dir(root, source, pulled_at.date(), dataset)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(directory) or {"dataset": dataset, "source": source, "files": {}}
    written = []
    for filename, df in frames.items():
        path = directory / filename
        df.write_parquet(path)
        manifest["files"][filename] = {"pulled_at": pulled_at.isoformat(), "rows": df.height}
        schema = schema_of(df)
        manifest["schema"] = schema
        manifest["schema_hash"] = schema_hash(schema)
        written.append(path)
    manifest["package_version"] = package_version
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return written
