"""A failed fit publishes nothing: only diagnostics.json, marked failed."""

import json
from datetime import date
from types import SimpleNamespace

import pytest

from dwar import pipeline
from dwar.model.diagnostics import Diagnostics, FitFailed


def test_failed_fit_writes_only_failed_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "asof_snapshot", lambda raw, as_of: SimpleNamespace(schedules=None))
    monkeypatch.setattr(pipeline, "build_player_tables", lambda snap, scoring: None)

    def failing(tables, cfg, out_dir, meta, cov=None):
        out_dir.mkdir(parents=True)
        diag = Diagnostics(False, ["2 divergences"], 2, 1.0, "", 1e3, "", 1e3, "")
        (out_dir / "rb_diagnostics.json").write_text(json.dumps(diag.to_dict()))
        raise FitFailed(diag)

    monkeypatch.setattr(pipeline, "fit_positions", failing)
    with pytest.raises(FitFailed):
        pipeline.run(date(2026, 10, 26), "superflex_ppr", root=tmp_path, raw=object())
    out = tmp_path / "2026-10-26" / "superflex_ppr"
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["passed"] is False and "RB" in diag["fits"]
    assert not (out / "dwar.parquet").exists()
