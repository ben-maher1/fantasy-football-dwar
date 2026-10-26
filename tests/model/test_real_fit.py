"""Fit WRs on the real nflverse cache (marked network: pulls history if
missing and takes a few minutes). A sanity check to eyeball, not a test of
which names come out on top."""

from datetime import date

import polars as pl
import pytest

from dwar.config import load_model_config, load_scoring
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.model.data import build_model_data
from dwar.model.fit import fit

AS_OF = date(2025, 9, 1)


@pytest.mark.network
def test_real_wr_fit(tmp_path):
    from dwar.ingest.nflverse import pull

    for name in ("player_stats", "schedules", "rosters", "draft_picks", "combine", "snap_counts", "players"):
        pull(name)
    tables = build_player_tables(asof_snapshot(RawTables.from_cache(), AS_OF), load_scoring("ppr"))
    cfg = load_model_config()
    d = build_model_data(tables, "WR", cfg.first_season)
    result = fit(d, cfg, tmp_path, meta={"as_of": AS_OF.isoformat(), "scoring": "ppr"})

    assert result.diagnostics.passed
    assert result.talent["season"].max() == 2024  # nothing from the 2025 season leaks in
    top = (
        result.talent.filter(pl.col("season") == 2024)
        .join(tables.player_static.select("player_id", "name"), on="player_id")
        .sort("talent_mean", descending=True)
        .head(15)
    )
    print(top.select("name", "games_played", "talent_p05", "talent_mean", "talent_p95"))
