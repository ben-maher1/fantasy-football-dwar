"""Situation-factor validation runs (the approved plan's runs A to D).

Each run is the step 5 model plus a set of situation factors, backtested on
a set of seasons. A run writes a copy of `config/` with its factors under
`data/experiments/{name}/config` and backtests with that config, so every
fit and forecast lands under `data/experiments/{name}/`.

    uv run python -m dwar.validate.experiments A
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from dwar.config import config_dir, data_dir
from dwar.features.situation import GROUPS, LEVEL, RANGE
from dwar.validate.run import DEV_SEASONS, HOLDOUT_SEASONS, validate

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


def _all(seasons: tuple[int, ...], group: str | None = None, **kw) -> Run:
    level, range_ = _without(group)
    return Run(seasons, level, range_, **kw)


RUNS: dict[str, Run] = {
    "step5": Run(DEV_SEASONS),  # the step 5 model, re-run on today's data for paired comparisons
    "A": _all(DEV_SEASONS),
    **{f"B_no_{g}": _all(SCREEN_SEASONS, g) for g in GROUPS},
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
    seasons = seasons or run.seasons
    label = label or ("holdout" if set(seasons) <= set(HOLDOUT_SEASONS) else "dev")
    return validate(root, truth_as_of or datetime.now(UTC).date(), seasons, label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    summary = run_experiment(sys.argv[1])
    print({g: summary["groups"][g]["dwar"] for g in summary["groups"]})
