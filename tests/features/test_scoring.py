from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from dwar.config import ScoringConfig, load_format, load_scoring
from dwar.features.scoring import STAT_COLUMNS, fantasy_points

ALL_STATS = sorted({c for cols in STAT_COLUMNS.values() for c in cols})


def week(**stats) -> dict:
    row = {c: 0 for c in ALL_STATS}
    row.update(stats)
    return row


# Hand-computed examples. Arithmetic is spelled out per row.
QB = week(
    position="QB",
    passing_yards=250,          # 250 * 0.04 = 10.0
    passing_tds=2,              # 2 * 4      =  8.0
    passing_interceptions=1,    # 1 * -2     = -2.0 (a pick-six costs the QB only the INT)
    rushing_yards=30,           # 30 * 0.1   =  3.0
    rushing_tds=1,              # 1 * 6      =  6.0
    sack_fumbles_lost=1,        # 1 * -2     = -2.0
)                               # total      = 23.0 in every format (no receptions)
TE = week(
    position="TE",
    receptions=6,               # PPR 6.0 | half 3.0 | standard 0 | TE premium 6 * 1.5 = 9.0
    receiving_yards=70,         # 7.0
)                               # PPR 13.0 | half 10.0 | standard 7.0 | TE premium 16.0
RB = week(
    position="RB",
    rushing_yards=85,           # 8.5
    rushing_tds=1,              # 6.0
    receptions=3,               # PPR 3.0 | half 1.5 | standard 0
    receiving_yards=22,         # 2.2
    rushing_fumbles_lost=1,     # -2.0
    rushing_2pt_conversions=1,  # 2.0
)                               # PPR 19.7 | half 18.2 | standard 16.7
WR = week(
    position="WR",
    receptions=4,               # PPR 4.0
    receiving_yards=51,         # 5.1
    special_teams_tds=1,        # 6.0 (kick return TD)
)                               # PPR 15.1 | half 13.1 | standard 11.1

EXPECTED = {
    "ppr": [23.0, 13.0, 19.7, 15.1],
    "half_ppr": [23.0, 10.0, 18.2, 13.1],
    "standard": [23.0, 7.0, 16.7, 11.1],
    "te_premium_ppr": [23.0, 16.0, 19.7, 15.1],
}


@pytest.fixture
def weekly() -> pl.DataFrame:
    return pl.DataFrame([QB, TE, RB, WR])


@pytest.mark.parametrize("fmt", sorted(EXPECTED))
def test_hand_computed(weekly, fmt):
    out = fantasy_points(weekly, load_scoring(fmt))
    assert out["fpts"].to_list() == pytest.approx(EXPECTED[fmt])


def test_format_deltas_are_receptions(weekly):
    ppr = fantasy_points(weekly, load_scoring("ppr"))["fpts"]
    half = fantasy_points(weekly, load_scoring("half_ppr"))["fpts"]
    std = fantasy_points(weekly, load_scoring("standard"))["fpts"]
    rec = weekly["receptions"].cast(pl.Float64)
    assert (ppr - std).to_list() == pytest.approx(rec.to_list())
    assert (ppr - half).to_list() == pytest.approx((rec * 0.5).to_list())


def test_te_premium_only_affects_tes(weekly):
    ppr = fantasy_points(weekly, load_scoring("ppr"))["fpts"]
    tep = fantasy_points(weekly, load_scoring("te_premium_ppr"))["fpts"]
    diff = (tep - ppr).to_list()
    assert diff == pytest.approx([0.0, 3.0, 0.0, 0.0])


def test_nulls_score_as_zero():
    df = pl.DataFrame([week(position="WR", receptions=2, receiving_yards=20)]).with_columns(
        pl.lit(None, dtype=pl.Int32).alias("receiving_tds")
    )
    assert fantasy_points(df, load_scoring("ppr"))["fpts"].item() == pytest.approx(4.0)


def test_input_columns_preserved(weekly):
    out = fantasy_points(weekly, load_scoring("ppr"))
    assert out.columns == [*weekly.columns, "fpts"]


def test_missing_stat_column_raises(weekly):
    with pytest.raises(ValueError, match="receiving_yards"):
        fantasy_points(weekly.drop("receiving_yards"), load_scoring("ppr"))


def test_first_down_bonus_opt_in(weekly):
    cfg = load_scoring("ppr").model_copy(update={"rec_fd": 0.5})
    df = weekly.with_columns(receiving_first_downs=pl.Series([0, 4, 1, 3]))
    base = fantasy_points(df, load_scoring("ppr"))["fpts"]
    bonus = fantasy_points(df, cfg)["fpts"]
    assert (bonus - base).to_list() == pytest.approx([0.0, 2.0, 0.5, 1.5])


def _scoring_yaml(tmp_path: Path, monkeypatch, body: str) -> None:
    (tmp_path / "scoring").mkdir()
    (tmp_path / "scoring" / "bad.yaml").write_text(body)
    monkeypatch.setenv("DWAR_CONFIG_DIR", str(tmp_path))


def test_unknown_key_raises(tmp_path, monkeypatch):
    good = (Path(__file__).parents[2] / "config" / "scoring" / "ppr.yaml").read_text()
    _scoring_yaml(tmp_path, monkeypatch, good + "pass_yds: 0.05\n")
    with pytest.raises(ValidationError, match="pass_yds"):
        load_scoring("bad")


def test_missing_required_key_raises(tmp_path, monkeypatch):
    good = (Path(__file__).parents[2] / "config" / "scoring" / "ppr.yaml").read_text()
    _scoring_yaml(tmp_path, monkeypatch, good.replace("rec_td: 6\n", ""))
    with pytest.raises(ValidationError, match="rec_td"):
        load_scoring("bad")


def test_config_is_frozen():
    with pytest.raises(ValidationError):
        load_scoring("ppr").rec = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("fmt", ["superflex_ppr", "1qb_ppr"])
def test_formats_reference_real_scoring(fmt):
    league = load_format(fmt)
    assert isinstance(load_scoring(league.scoring), ScoringConfig)


@pytest.mark.network
@pytest.mark.parametrize("season", [2012, 2024])
def test_matches_nflverse_fantasy_points(season):
    import nflreadpy

    stats = nflreadpy.load_player_stats([season], summary_level="week")
    ppr = fantasy_points(stats, load_scoring("ppr"))
    std = fantasy_points(stats, load_scoring("standard"))
    assert (ppr["fpts"] - ppr["fantasy_points_ppr"].fill_null(0)).abs().max() < 0.01
    assert (std["fpts"] - std["fantasy_points"].fill_null(0)).abs().max() < 0.01
