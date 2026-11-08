"""Situation-factor validation runs (the approved plan's runs A to D).

Each run is the step 5 model plus a set of situation factors, backtested on
a set of seasons. A run writes a copy of `config/` with its factors under
`data/experiments/{name}/config` and backtests with that config, so every
fit and forecast lands under `data/experiments/{name}/`.

    uv run python -m dwar.validate.experiments A
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import yaml

from dwar.config import config_dir, data_dir
from dwar.features.situation import GROUPS, LEVEL, RANGE
from dwar.validate.run import DEV_SEASONS, HOLDOUT_SEASONS, validate
from dwar.validate.score import summarize

log = logging.getLogger(__name__)

SCREEN_SEASONS = DEV_SEASONS[::2]  # 2012, 2014, ..., 2022


def _without(group: str | None) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    drop = set(GROUPS[group]) if group else set()
    level = {p: [c for c in cols if c not in drop] for p, cols in LEVEL.items()}
    range_ = {p: [c for c in cols if c not in drop] for p, cols in RANGE.items()}
    return level, range_


@dataclass(frozen=True)
class Run:
    seasons: tuple[int, ...]
    level: dict[str, list[str]] = field(default_factory=dict)
    range: dict[str, list[str]] = field(default_factory=dict)
    model: dict = field(default_factory=dict)   # other model.yaml overrides
    value: dict = field(default_factory=dict)   # value.yaml overrides
    fits_from: str | None = None                # reuse another run's fits (same model, new forecasts)


def _all(seasons: tuple[int, ...], group: str | None = None, **kw) -> Run:
    level, range_ = _without(group)
    return Run(seasons, level, range_, **kw)


FINAL_LEVEL = {"QB": ["qb_starter"], "RB": ["competition", "vacated", "rookie_capital"]}  # run F's factors

RUNS: dict[str, Run] = {
    "step5": Run(DEV_SEASONS),  # the step 5 model, re-run on today's data for paired comparisons
    "A": _all(DEV_SEASONS),
    # A's fits; QB short-game chances by preseason role (QB1 or not).
    "A_role": _all(DEV_SEASONS, value={"short_by_role": ["QB"]}, fits_from="A"),
    **{f"B_no_{g}": _all(SCREEN_SEASONS, g) for g in GROUPS},
    # The final set, from runs B: a group stays at a position where removing
    # it makes MAE or the floored log score clearly worse (90% interval),
    # and neither clearly better.
    "F": Run(
        DEV_SEASONS,
        level={"QB": ["qb_starter"], "RB": ["competition", "vacated", "rookie_capital"]},
        value={"short_by_role": ["QB"]},
    ),
    # Run C: the QB spread factor off, and a hinge QB age curve (QB refit only).
    "C_nospread": _all(DEV_SEASONS, value={"short_by_role": ["QB"], "calibrate_spread": []}, fits_from="A"),
    "C_hinge": _all(DEV_SEASONS, model={"positions": ["QB"], "age_curve": {"QB": "hinge"}},
                    value={"short_by_role": ["QB"]}),
    # Usage and early-career prior plan. U_base is the current model; U_A adds
    # opportunity points + efficiency (RB/WR/TE), the rookie first-round step
    # and career-year growth scaled by draft capital (all positions). QB
    # opportunity was dropped: QB dropbacks are valued alike, so QB talent
    # moved almost wholly into efficiency, the veteran spread collapsed and
    # the 2012 QB fit failed its checks (30 divergences, R-hat 1.05).
    "U_base": Run(DEV_SEASONS, level=FINAL_LEVEL),
    "U_A": Run(DEV_SEASONS, level=FINAL_LEVEL, model={
        "opportunity": ["RB", "WR", "TE"], "rookie_round1": ["QB", "RB", "WR", "TE"],
        "growth": ["QB", "RB", "WR", "TE"],
    }),
    # Screening (even seasons 2012-2018) put U_A behind the current model at
    # RB, WR and TE on MAE and floored log score, with wider forecasts, while
    # its growth terms removed the second-year under-forecast. U_G keeps only
    # the early-career terms to separate the two.
    "U_G": Run(DEV_SEASONS, level=FINAL_LEVEL, model={
        "rookie_round1": ["QB", "RB", "WR", "TE"], "growth": ["QB", "RB", "WR", "TE"],
    }),
    # U_G edged the current model at every position. U_E adds opportunity
    # back for career years 1-2 only, where a few games of points say least.
    "U_E": Run(DEV_SEASONS, level=FINAL_LEVEL, model={
        "rookie_round1": ["QB", "RB", "WR", "TE"], "growth": ["QB", "RB", "WR", "TE"],
        "opportunity": ["RB", "WR", "TE"], "opportunity_years": 2,
    }),
    # Screening: U_E trailed U_G at RB, WR and TE (floored log score -0.02 to
    # -0.06, MAE worse). It helped rookie forecasts a little but hurt
    # forecasts after a season of points, so opportunity is dropped.
}


def write_config(name: str, run: Run, root: Path) -> Path:
    out = root / "config"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(config_dir(), out)
    model = yaml.safe_load((out / "model.yaml").read_text())
    model |= {"situation_level": run.level, "situation_range": run.range} | run.model
    (out / "model.yaml").write_text(yaml.safe_dump(model, sort_keys=False))
    if run.value:
        value = yaml.safe_load((out / "value.yaml").read_text()) | run.value
        (out / "value.yaml").write_text(yaml.safe_dump(value, sort_keys=False))
    return out


def run_experiment(name: str, truth_as_of: date | None = None, label: str | None = None,
                   seasons: tuple[int, ...] | None = None) -> dict:
    run = RUNS[name]
    root = data_dir() / "experiments" / name
    os.environ["DWAR_CONFIG_DIR"] = str(write_config(name, run, root))
    if run.fits_from and not (root / "fits").exists():
        (root / "fits").symlink_to(data_dir() / "experiments" / run.fits_from / "fits", target_is_directory=True)
    seasons = seasons or run.seasons
    label = label or ("holdout" if set(seasons) <= set(HOLDOUT_SEASONS) else "dev")
    return validate(root, truth_as_of or datetime.now(UTC).date(), seasons, label)


def rescore(name: str, label: str) -> dict:
    """Recompute a finished run's scores from its saved results with the
    current scoring rules, keeping the rest of its summary."""
    out = data_dir() / "experiments" / name / label
    summary = json.loads((out / "summary.json").read_text())
    results = pl.read_parquet(out / "results.parquet")
    positions = tuple(sorted(results["position"].unique().to_list(), key="QBRBWRTE".find))
    summary |= summarize(results, positions)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    # name [seasons label], e.g. F 2023,2024,2025 holdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    seasons = tuple(int(s) for s in sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    summary = run_experiment(sys.argv[1], label=sys.argv[3] if len(sys.argv) > 3 else None, seasons=seasons)
    print({g: summary["groups"][g]["dwar"] for g in summary["groups"]})
