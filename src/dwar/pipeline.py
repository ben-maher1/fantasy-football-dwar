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

from dwar.config import REPO_ROOT, load_format, load_model_config, load_scoring, load_value_config, uses_situation
from dwar.features.asof import RawTables, asof_snapshot
from dwar.features.player_tables import build_player_tables
from dwar.features.situation import SituationRaw, model_covariates, prepare
from dwar.model.diagnostics import FitFailed
from dwar.model.fit import fit_positions
from dwar.value.dwar import dwar_table, season_clock

log = logging.getLogger(__name__)


def output_dir(as_of: date, format_name: str, root: Path | None = None) -> Path:
    return (root or REPO_ROOT / "outputs") / as_of.isoformat() / format_name


def run(
    as_of: date, format_name: str, root: Path | None = None, raw: RawTables | None = None,
    situation_raw: SituationRaw | None = None,
) -> Path:
    """Run DWAR for one league format; returns the path of `dwar.parquet`."""
    start = time.perf_counter()
    fmt = load_format(format_name)
    model_cfg, value_cfg = load_model_config(), load_value_config()
    out = output_dir(as_of, format_name, root)
    out.mkdir(parents=True, exist_ok=True)
    snap = asof_snapshot(raw or RawTables.from_cache(), as_of)
    tables = build_player_tables(snap, load_scoring(fmt.scoring))
    meta = {"as_of": as_of.isoformat(), "format": format_name, "scoring": fmt.scoring}
    cov = {}
    if uses_situation(model_cfg):
        # Factors for the season after each player's latest one: this season
        # (offseason, or not yet played) or next season (played this season).
        current = season_clock(snap.schedules, as_of).current
        cov = model_covariates(
            prepare(situation_raw or SituationRaw.from_cache()), snap, tables, model_cfg.first_season,
            model_cfg.situation_level, model_cfg.situation_range, (current, current + 1),
        )

    diagnostics = {**meta, "passed": False, "fits": {}}
    try:
        fits = fit_positions(tables, model_cfg, out / "fits", meta, cov)
    except FitFailed:
        for p in model_cfg.positions:
            path = out / "fits" / f"{p.lower()}_diagnostics.json"
            if path.exists():
                diagnostics["fits"][p] = json.loads(path.read_text())
        (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))
        raise

    table, summary = dwar_table(
        fits, tables, snap.schedules, fmt, value_cfg, model_cfg.likelihoods(), model_cfg.seed, model_cfg.min_plays,
        cov,
    )
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
