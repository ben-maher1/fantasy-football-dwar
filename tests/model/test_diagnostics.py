import json

import arviz as az
import numpy as np
import pytest

from dwar.config import load_model_config
from dwar.model import fit as fit_module
from dwar.model.diagnostics import FitFailed, check_fit

from .simulate import simulate

THRESHOLDS = load_model_config().diagnostics
rng = np.random.default_rng(0)


def posterior(a=None, b=None, diverging=None):
    return az.from_dict({
        "posterior": {
            "a": rng.normal(size=(4, 500)) if a is None else a,
            "b": rng.normal(size=(4, 500, 3)) if b is None else b,
        },
        "sample_stats": {"diverging": np.zeros((4, 500), bool) if diverging is None else diverging},
    })


def test_healthy_chains_pass():
    d = check_fit(posterior(), THRESHOLDS)
    assert d.passed and d.failures == []
    assert d.max_rhat <= 1.01 and d.min_ess_bulk > 400


def test_one_divergence_fails():
    div = np.zeros((4, 500), bool)
    div[2, 17] = True
    d = check_fit(posterior(diverging=div), THRESHOLDS)
    assert not d.passed and d.divergences == 1
    assert "divergent" in d.failures[0]


def test_disagreeing_chain_fails_rhat_and_names_parameter():
    b = rng.normal(size=(4, 500, 3))
    b[1, :, 2] += 0.5
    d = check_fit(posterior(b=b), THRESHOLDS)
    assert not d.passed
    assert d.max_rhat > 1.01 and d.max_rhat_param == "b[2]"


def test_sticky_chain_fails_ess():
    sticky = np.repeat(rng.normal(size=(4, 25)), 20, axis=1)  # each value held 20 draws
    d = check_fit(posterior(a=sticky), THRESHOLDS)
    assert not d.passed
    assert d.min_ess_bulk < 400 and d.min_ess_bulk_param == "a"


def test_failed_fit_writes_only_failed_diagnostics(tmp_path, monkeypatch):
    sim = simulate(n_players=20, weeks_per_season=4)
    div = np.zeros((4, 500), bool)
    div[0, 0] = True
    monkeypatch.setattr(fit_module, "sample", lambda d, cfg: posterior(diverging=div))
    with pytest.raises(FitFailed, match="divergent"):
        fit_module.fit(sim.data, load_model_config(), tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["wr_diagnostics.json"]
    written = json.loads((tmp_path / "wr_diagnostics.json").read_text())
    assert written["passed"] is False and written["divergences"] == 1
    assert written["extra"]["seed"] == load_model_config().seed
