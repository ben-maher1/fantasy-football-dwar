import polars as pl

from dwar.features.fill_in import flag_fill_ins, lead_backs


def _weeks(rows):
    return pl.DataFrame(rows, schema=["player_id", "team", "season", "week", "position", "carries"], orient="row")


def test_backup_flagged_only_when_lead_back_misses():
    rows = []
    for wk in (1, 2, 3, 4, 5):
        if wk != 4:
            rows.append(("lead", "CHI", 2025, wk, "RB", 15))
        rows.append(("backup", "CHI", 2025, wk, "RB", 25 if wk == 4 else 5))
        rows.append(("wr", "CHI", 2025, wk, "WR", 0))
    out = flag_fill_ins(_weeks(rows))
    flagged = out.filter(pl.col("fill_in")).select("player_id", "week").rows()
    assert flagged == [("backup", 4)]


def test_no_lead_back_before_two_team_games_or_in_a_committee():
    rows = []
    for wk in (1, 2, 3):
        rows += [("a", "MIA", 2025, wk, "RB", 10), ("b", "MIA", 2025, wk, "RB", 9), ("c", "MIA", 2025, wk, "RB", 9),
                 ("wr", "MIA", 2025, wk, "WR", 0)]
    rows += [("b", "MIA", 2025, 4, "RB", 20), ("wr", "MIA", 2025, 4, "WR", 0)]
    # Week 1: no earlier games; weeks 2-4: 'a' has under 40% of RB carries.
    assert lead_backs(_weeks(rows)).height == 0
    assert not flag_fill_ins(_weeks(rows))["fill_in"].any()


def test_lead_back_uses_only_earlier_games():
    # The backup's big week makes him the season leader, but the lead back
    # going into week 3 is the one who led weeks 1-2.
    rows = [("lead", "BUF", 2025, 1, "RB", 20), ("backup", "BUF", 2025, 1, "RB", 2),
            ("lead", "BUF", 2025, 2, "RB", 20), ("backup", "BUF", 2025, 2, "RB", 2),
            ("backup", "BUF", 2025, 3, "RB", 60)]
    lead = lead_backs(_weeks(rows)).filter(pl.col("week") == 3).row(0, named=True)
    assert lead["lead"] == "lead" and not lead["lead_played"]


def test_no_flags_without_carries():
    w = _weeks([("x", "NYJ", 2025, 1, "RB", 5)]).drop("carries")
    assert not flag_fill_ins(w)["fill_in"].any()
