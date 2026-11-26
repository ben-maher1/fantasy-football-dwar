import polars as pl

from dwar.features.long_plays import long_play_counts, with_long_plays


def play(**kw) -> dict:
    row = dict(game_id="g1", two_point_attempt=0, yards_gained=0, complete_pass=0, pass_touchdown=0,
               rush_attempt=0, rush_touchdown=0, passer_player_id=None, receiver_player_id=None,
               rusher_player_id=None)
    row.update(kw)
    return row


PBP = pl.DataFrame([
    play(yards_gained=45, complete_pass=1, pass_touchdown=1, passer_player_id="qb", receiver_player_id="wr"),
    play(yards_gained=40, complete_pass=1, passer_player_id="qb", receiver_player_id="wr"),
    play(yards_gained=39, complete_pass=1, passer_player_id="qb", receiver_player_id="wr"),   # too short
    play(yards_gained=50, complete_pass=0, passer_player_id="qb", receiver_player_id="wr"),   # incomplete (penalty yards)
    play(yards_gained=60, rush_attempt=1, rush_touchdown=1, rusher_player_id="rb"),
    play(yards_gained=41, rush_attempt=1, rusher_player_id="qb"),                              # QB scramble
], schema_overrides={"passer_player_id": pl.String, "receiver_player_id": pl.String, "rusher_player_id": pl.String})


def test_counts():
    c = {r["player_id"]: r for r in long_play_counts(PBP).to_dicts()}
    assert (c["qb"]["passing_cmp_40p"], c["qb"]["passing_td_40p"], c["qb"]["rushing_40p"]) == (2, 1, 1)
    assert (c["wr"]["receiving_40p"], c["wr"]["receiving_td_40p"]) == (2, 1)
    assert (c["rb"]["rushing_40p"], c["rb"]["rushing_td_40p"]) == (1, 1)


def test_join_fills_zero():
    stats = pl.DataFrame({"game_id": ["g1", "g1", "g2"], "player_id": ["wr", "te", "wr"]})
    out = with_long_plays(stats, PBP)
    assert out["receiving_40p"].to_list() == [2, 0, 0]
