"""Convergence checks. A fit that fails any of them must not be published
(CLAUDE.md rule 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import arviz as az
import numpy as np
from xarray import DataTree

from dwar.config import DiagnosticThresholds


@dataclass
class Diagnostics:
    passed: bool
    failures: list[str]
    divergences: int
    max_rhat: float
    max_rhat_param: str
    min_ess_bulk: float
    min_ess_bulk_param: str
    min_ess_tail: float
    min_ess_tail_param: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FitFailed(RuntimeError):
    def __init__(self, diagnostics: Diagnostics):
        self.diagnostics = diagnostics
        super().__init__("; ".join(diagnostics.failures))


def _extreme(stat: DataTree, worst: str) -> tuple[float, str]:
    """Worst value over every element of every variable, and where it is."""
    best_val, best_name = (-np.inf if worst == "max" else np.inf), ""
    for name, da in stat.dataset.data_vars.items():
        values = np.asarray(da.values, dtype=float)
        if values.size == 0:
            continue
        flat = values.reshape(-1)
        idx = int(np.nanargmax(flat) if worst == "max" else np.nanargmin(flat))
        val = float(flat[idx])
        if np.isnan(val):
            val = np.inf if worst == "max" else -np.inf
        if (worst == "max" and val > best_val) or (worst == "min" and val < best_val):
            where = np.unravel_index(idx, values.shape) if values.ndim else ()
            best_val, best_name = val, f"{name}{list(map(int, where)) if where else ''}"
    return best_val, best_name


def check_fit(posterior: DataTree, thresholds: DiagnosticThresholds) -> Diagnostics:
    divergences = int(np.asarray(posterior["sample_stats"]["diverging"]).sum())
    max_rhat, rhat_at = _extreme(az.rhat(posterior), "max")
    min_bulk, bulk_at = _extreme(az.ess(posterior, method="bulk"), "min")
    min_tail, tail_at = _extreme(az.ess(posterior, method="tail"), "min")

    failures = []
    if divergences > thresholds.max_divergences:
        failures.append(f"{divergences} divergent transitions (max {thresholds.max_divergences})")
    if not max_rhat <= thresholds.max_rhat:
        failures.append(f"R-hat {max_rhat:.4f} at {rhat_at} (max {thresholds.max_rhat})")
    if not min_bulk >= thresholds.min_ess_bulk:
        failures.append(f"bulk ESS {min_bulk:.0f} at {bulk_at} (min {thresholds.min_ess_bulk:.0f})")
    if not min_tail >= thresholds.min_ess_tail:
        failures.append(f"tail ESS {min_tail:.0f} at {tail_at} (min {thresholds.min_ess_tail:.0f})")

    return Diagnostics(
        passed=not failures,
        failures=failures,
        divergences=divergences,
        max_rhat=max_rhat,
        max_rhat_param=rhat_at,
        min_ess_bulk=min_bulk,
        min_ess_bulk_param=bulk_at,
        min_ess_tail=min_tail,
        min_ess_tail_param=tail_at,
    )
