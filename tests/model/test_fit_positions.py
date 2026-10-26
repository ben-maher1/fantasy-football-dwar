"""`fit_positions` tries every configured position and fails the run if any fit fails."""

import pytest

from dwar.config import load_model_config
from dwar.model import fit as fit_module
from dwar.model.diagnostics import Diagnostics, FitFailed


def test_all_positions_attempted_and_any_failure_fails_the_run(monkeypatch, tmp_path):
    attempted = []
    monkeypatch.setattr(fit_module, "build_model_data", lambda tables, position, first_season: position)

    def fake_fit(d, cfg, out_dir, meta=None):
        attempted.append(d)
        if d == "RB":
            raise FitFailed(Diagnostics(False, ["3 divergences"], 3, 1.0, "", 1e3, "", 1e3, ""))
        return d

    monkeypatch.setattr(fit_module, "fit", fake_fit)
    cfg = load_model_config()
    with pytest.raises(FitFailed):
        fit_module.fit_positions(None, cfg, tmp_path)
    assert attempted == list(cfg.positions)


def test_all_positions_pass(monkeypatch, tmp_path):
    monkeypatch.setattr(fit_module, "build_model_data", lambda tables, position, first_season: position)
    monkeypatch.setattr(fit_module, "fit", lambda d, cfg, out_dir, meta=None: f"fit-{d}")
    cfg = load_model_config()
    assert fit_module.fit_positions(None, cfg, tmp_path) == {p: f"fit-{p}" for p in cfg.positions}
