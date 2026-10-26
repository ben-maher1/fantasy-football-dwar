"""One DWAR run: as-of tables -> talent fits for every position -> DWAR.

Writes `outputs/{as_of}/{format}/dwar.parquet` and `diagnostics.json`. If any
fit fails the gate, only the diagnostics are written (marked failed) and
`FitFailed` is raised: nothing is published. The step 6 CLI wraps `run`.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path

from dwar.config import REPO_ROOT, load_format, load_model_config, load_scoring, load_value_config
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.model.diagnostics import FitFailed
from dwar.model.fit import fit_positions
from dwar.value.dwar import dwar_table

log = logging.getLogger(__name__)


def output_dir(as_of: date, format_name: str, root: Path | None = None) -> Path:
    return (root or REPO_ROOT / "outputs") / as_of.isoformat() / format_name


def run(as_of: date, format_name: str, root: Path | None = None, raw: RawTables | None = None) -> Path:
    """Run DWAR for one league format; returns the path of `dwar.parquet`."""
    start = time.perf_counter()
    fmt = load_format(format_name)
    model_cfg, value_cfg = load_model_config(), load_value_config()
    out = output_dir(as_of, format_name, root)
    out.mkdir(parents=True, exist_ok=True)
    snap = asof_snapshot(raw or RawTables.from_cache(), as_of)
    tables = build_player_tables(snap, load_scoring(fmt.scoring))
    meta = {"as_of": as_of.isoformat(), "format": format_name, "scoring": fmt.scoring}

    diagnostics = {**meta, "passed": False, "fits": {}}
    try:
        fits = fit_positions(tables, model_cfg, out / "fits", meta)
    except FitFailed:
        for p in model_cfg.positions:
            path = out / "fits" / f"{p.lower()}_diagnostics.json"
            if path.exists():
                diagnostics["fits"][p] = json.loads(path.read_text())
        (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))
        raise

    table, summary = dwar_table(fits, tables, snap.schedules, fmt, value_cfg, model_cfg.likelihood, model_cfg.seed)
    path = out / "dwar.parquet"
    table.write_parquet(path)
    diagnostics.update(
        passed=True,
        runtime_s=round(time.perf_counter() - start, 1),
        fits={p: r.diagnostics.to_dict() for p, r in fits.items()},
        value=summary,
        n_players=table.height,
    )
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))
    log.info("wrote %s (%d players)", path, table.height)
    return path
