"""Ingest tests run offline: nflreadpy loaders are replaced by fixture frames."""

from datetime import UTC, date, datetime, timedelta

import nflreadpy
import polars as pl
import pytest

from dwar.ingest import nflverse
from dwar.ingest.cache import SchemaChangedError, read_manifest
from dwar.ingest.nflverse import DRAFT_PICK_COLUMNS, is_final, load_raw, pull

MIDSEASON = datetime(2024, 10, 15, 12, tzinfo=UTC)


class FakeLoader:
    """Stands in for an nflreadpy loader and records every call."""

    def __init__(self, make):
        self.make = make
        self.calls: list = []

    def __call__(self, seasons=None, **kwargs):
        self.calls.append(seasons)
        return self.make(seasons)


def stats_frame(seasons, extra_col=False, value=1):
    rows = [{"season": s, "week": 1, "player_id": "00-1", "receptions": value} for s in seasons]
    df = pl.DataFrame(rows)
    return df.with_columns(new_col=pl.lit(0)) if extra_col else df


@pytest.fixture
def stats_loader(monkeypatch):
    loader = FakeLoader(stats_frame)
    monkeypatch.setattr(nflreadpy, "load_player_stats", loader)
    return loader


def test_writes_dated_season_partitions(tmp_path, stats_loader):
    written = pull("player_stats", [2023, 2024], now=MIDSEASON, root=tmp_path)
    base = tmp_path / "nflverse" / "2024-10-15" / "player_stats"
    assert sorted(written) == [base / "season=2023.parquet", base / "season=2024.parquet"]
    manifest = read_manifest(base)
    assert manifest["files"]["season=2023.parquet"]["rows"] == 1
    assert manifest["package_version"]
    assert "receptions" in manifest["schema"]


def test_final_seasons_never_redownloaded(tmp_path, stats_loader):
    pull("player_stats", [2023, 2024], now=MIDSEASON, root=tmp_path)
    later = MIDSEASON + timedelta(days=30)
    pull("player_stats", [2023, 2024], now=later, root=tmp_path)
    # 2023 was final when first pulled (after 2024-03-01); only 2024 re-pulls.
    assert stats_loader.calls == [[2023, 2024], [2024]]


def test_current_season_respects_max_age(tmp_path, stats_loader):
    pull("player_stats", [2024], now=MIDSEASON, root=tmp_path)
    assert pull("player_stats", [2024], now=MIDSEASON + timedelta(hours=23), root=tmp_path) == []
    assert pull("player_stats", [2024], now=MIDSEASON + timedelta(hours=25), root=tmp_path)
    assert stats_loader.calls == [[2024], [2024]]


def test_season_pulled_midseason_refreshes_after_it_ends(tmp_path, stats_loader):
    pull("player_stats", [2024], now=MIDSEASON, root=tmp_path)
    after = datetime(2025, 4, 1, tzinfo=UTC)
    pull("player_stats", [2024], now=after, root=tmp_path)
    pull("player_stats", [2024], now=after + timedelta(days=60), root=tmp_path)
    assert stats_loader.calls == [[2024], [2024]]


def test_is_final_boundary():
    assert not is_final(2024, datetime(2025, 2, 28, 23, tzinfo=UTC))
    assert is_final(2024, datetime(2025, 3, 1, tzinfo=UTC))


def test_earlier_snapshots_are_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(nflreadpy, "load_player_stats", FakeLoader(lambda s: stats_frame(s, value=1)))
    pull("player_stats", [2024], now=MIDSEASON, root=tmp_path)
    first = tmp_path / "nflverse" / "2024-10-15" / "player_stats" / "season=2024.parquet"
    before = first.read_bytes()

    monkeypatch.setattr(nflreadpy, "load_player_stats", FakeLoader(lambda s: stats_frame(s, value=9)))
    pull("player_stats", [2024], now=MIDSEASON + timedelta(days=2), root=tmp_path)
    assert first.read_bytes() == before


def test_load_raw_latest_and_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(nflreadpy, "load_player_stats", FakeLoader(lambda s: stats_frame(s, value=1)))
    pull("player_stats", [2023, 2024], now=MIDSEASON, root=tmp_path)
    monkeypatch.setattr(nflreadpy, "load_player_stats", FakeLoader(lambda s: stats_frame(s, value=9)))
    pull("player_stats", [2024], now=MIDSEASON + timedelta(days=2), root=tmp_path)

    latest = load_raw("player_stats", root=tmp_path).sort("season")
    assert latest["receptions"].to_list() == [1, 9]  # 2023 from first pull, 2024 refreshed

    pinned = load_raw("player_stats", pull_date=date(2024, 10, 15), root=tmp_path).sort("season")
    assert pinned["receptions"].to_list() == [1, 1]


def test_load_raw_missing_season_raises(tmp_path, stats_loader):
    pull("player_stats", [2024], now=MIDSEASON, root=tmp_path)
    with pytest.raises(FileNotFoundError, match="2019"):
        load_raw("player_stats", [2019], root=tmp_path)


def test_draft_pick_outcomes_dropped(tmp_path, monkeypatch):
    def picks(seasons):
        row = {c: 1 for c in DRAFT_PICK_COLUMNS} | {"season": seasons[0]}
        row |= {"car_av": 80, "w_av": 70, "games": 150, "to": 2020, "allpro": 3, "hof": True}
        return pl.DataFrame([row])

    monkeypatch.setattr(nflreadpy, "load_draft_picks", FakeLoader(picks))
    pull("draft_picks", [2010], now=MIDSEASON, root=tmp_path)
    cols = load_raw("draft_picks", root=tmp_path).columns
    assert set(cols) == set(DRAFT_PICK_COLUMNS)


def test_combine_draft_results_dropped(tmp_path, monkeypatch):
    def combine(seasons):
        return pl.DataFrame([{"season": seasons[0], "player_name": "X", "forty": 4.4,
                              "draft_team": "KC", "draft_round": 1, "draft_ovr": 20}])

    monkeypatch.setattr(nflreadpy, "load_combine", FakeLoader(combine))
    pull("combine", [2010], now=MIDSEASON, root=tmp_path)
    assert load_raw("combine", root=tmp_path).columns == ["season", "player_name", "forty"]


def test_schema_change_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(nflreadpy, "load_player_stats", FakeLoader(stats_frame))
    pull("player_stats", [2024], now=MIDSEASON, root=tmp_path)
    monkeypatch.setattr(nflreadpy, "load_player_stats",
                        FakeLoader(lambda s: stats_frame(s, extra_col=True)))
    later = MIDSEASON + timedelta(days=2)
    with pytest.raises(SchemaChangedError, match="new_col"):
        pull("player_stats", [2024], now=later, root=tmp_path)
    assert pull("player_stats", [2024], now=later, root=tmp_path, allow_schema_change=True)


def test_non_seasonal_dataset(tmp_path, monkeypatch):
    loader = FakeLoader(lambda _: pl.DataFrame({"gsis_id": ["00-1"], "birth_date": ["2000-01-01"]}))
    monkeypatch.setattr(nflreadpy, "load_players", loader)
    pull("players", now=MIDSEASON, root=tmp_path)
    assert pull("players", now=MIDSEASON + timedelta(hours=1), root=tmp_path) == []
    pull("players", now=MIDSEASON + timedelta(days=2), root=tmp_path)
    assert len(loader.calls) == 2
    assert load_raw("players", root=tmp_path).height == 1


def test_empty_season_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(nflreadpy, "load_snap_counts", FakeLoader(lambda s: stats_frame([2024])))
    written = pull("snap_counts", [2024, 2025], now=MIDSEASON, root=tmp_path)
    assert [p.name for p in written] == ["season=2024.parquet"]


def test_unknown_dataset():
    with pytest.raises(KeyError, match="Known"):
        pull("pbp_everything")


@pytest.mark.network
def test_live_pull_roundtrip(tmp_path):
    for name in nflverse.DATASETS:
        seasons = None if not nflverse.DATASETS[name].seasonal else [2023]
        pull(name, seasons, root=tmp_path)
        assert load_raw(name, seasons, root=tmp_path).height > 0
