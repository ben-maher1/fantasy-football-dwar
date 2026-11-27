from dataclasses import replace

import numpy as np
import polars as pl

from dwar.ingest import sleeper as S
from dwar.value import league_odds as O

LEAGUE = {"season": "2026", "settings": {"draft_rounds": 1, "last_scored_leg": 1, "playoff_week_start": 4,
                                         "playoff_teams": 2}}
ROSTERS = [
    {"roster_id": 1, "owner_id": "u1", "players": ["a", "b"], "taxi": ["b"], "settings": {"wins": 1, "fpts": 120}},
    {"roster_id": 2, "owner_id": "u2", "players": ["c"], "settings": {"losses": 1, "fpts": 100}},
]
USERS = [{"user_id": "u1", "display_name": "ben"}, {"user_id": "u2", "display_name": "rival"}]
SCHED = [{"roster_id": 1, "matchup_id": 1, "points": 0.0}, {"roster_id": 2, "matchup_id": 1, "points": 0.0}]
MATCHUPS = {1: [{"roster_id": 1, "matchup_id": 1, "points": 120.0}, {"roster_id": 2, "matchup_id": 1, "points": 100.0}],
            2: SCHED, 3: SCHED}
TRADED = [{"season": "2027", "round": 1, "roster_id": 2, "owner_id": 1, "previous_owner_id": 2}]


def _ctx(qb_talent: list[float]) -> O.Context:
    D, J = 4, 2
    tal = np.broadcast_to(np.array(qb_talent)[None, :, None], (D, 3, J)).copy()
    rank = np.array([1, 2])
    rookies = {2027: (rank, np.array(["QB", "QB"], object), np.full((D, 2, J), 50.0), np.ones((D, 2, J)))}
    hurdle = {"sigma": 0.5, "nu": 30.0, "h0": 1.5, "h1": 1.0, "zero_value": 0.0}
    return O.Context(season0=2026, player_id=["ga", "gb", "gc"], position=np.array(["QB"] * 3, object),
                     talent=tal, share=np.ones((D, 3, J)), rookies=rookies, waiver={"QB": 0.0},
                     params={"QB": hurdle}, likelihood={"QB": "sqrt_hurdle"})


def _setup(ctx):
    lg = S.League(LEAGUE, ROSTERS, USERS, TRADED, MATCHUPS, {})
    xw = pl.DataFrame({"sleeper_id": ["a", "b", "c"], "gsis_id": ["ga", "gb", "gc"]})
    return O.league_setup(lg, ctx, xw, {"QB": 1})


def test_league_setup_reads_rosters_schedule_and_picks():
    ctx = _ctx([20.0, 40.0, 10.0])
    lg = _setup(ctx)
    assert lg.roster.tolist() == [0, 0, 1] and lg.taxi.tolist() == [False, True, False]
    assert lg.schedule_left.tolist() == [[1, 0], [1, 0]]
    assert lg.wins.tolist() == [1.0, 0.0] and lg.points.tolist() == [120.0, 100.0]
    assert (2027, 1, 1, 0) in lg.picks           # team 2's 2027 1st now belongs to team 1


def test_taxi_cannot_start_this_season_and_traded_pick_joins_next():
    # Team 1's starter is weak this year (its 40-PPG QB is on taxi), team 2's 10-PPG QB starts.
    ctx = _ctx([5.0, 40.0, 10.0])
    lg = _setup(ctx)
    rng = np.random.default_rng(1)
    paths = np.stack([O.simulate_path(ctx, lg, d % 4, 2, rng, ctx.talent.mean(0))[0] for d in range(60)])
    # This season team 2 is favoured (team 1 starts its 5-PPG QB); next season team 1's taxi QB plays.
    assert (paths[:, 0, 1] == 0).mean() > 0.6
    assert (paths[:, 1, 0] == 0).mean() > 0.9


def test_with_and_without_values_share_the_same_weeks():
    ctx = _ctx([15.0, 40.0, 20.0])
    r, rp, rt, rs = ctx.rookies[2027]
    ctx.rookies[2027] = (r, rp, np.full_like(rt, 1.0), rs)     # rookies who never start
    lg = _setup(ctx)
    assets = O.assets_for(lg, 0)
    kinds = {(a.kind, a.ref, a.team, a.mode) for a in assets}
    assert ("player", 2, 0, "add") in kinds and ("player", 1, 0, "remove") in kinds
    assert ("pick", lg.picks.index((2027, 1, 1, 0)), 0, "remove") in kinds
    rng = np.random.default_rng(3)
    d = np.stack([O.simulate_path(ctx, lg, i % 4, 2, rng, ctx.talent.mean(0), assets, {0: 0})[1] for i in range(80)])
    a_taxi = next(k for k, a in enumerate(assets) if (a.kind, a.ref, a.mode) == ("player", 1, "remove"))
    assert (d[:, 0, a_taxi] == 0).all()          # a taxi player cannot change this season
    assert d[:, 1, a_taxi].mean() > 0.3          # next season he is the team's QB
    assert set(np.unique(d)) <= {-1, 0, 1}


def test_a_trade_is_one_counterfactual():
    ctx = _ctx([15.0, 40.0, 20.0])
    r, rp, rt, rs = ctx.rookies[2027]
    ctx.rookies[2027] = (r, rp, np.full_like(rt, 1.0), rs)
    lg = _setup(ctx)
    # Team 0 sends its starter (QB 0) for team 1's QB 2: next season (taxi QB 1 plays) nothing changes for it.
    swap = O.Asset("trade", 0, 0, "trade", (("player", 0, "remove"), ("player", 2, "add")))
    same = O.Asset("trade", 1, 0, "trade", (("player", 0, "remove"), ("player", 0, "add")))
    rng = np.random.default_rng(5)
    d = np.stack([O.simulate_path(ctx, lg, i % 4, 2, rng, ctx.talent.mean(0), [swap, same])[1] for i in range(60)])
    assert d[:, 0, 0].mean() >= 0 and (d[:, :, 1] == 0).all()


def test_a_bye_every_week_takes_the_player_out():
    # Team 0's starter (QB 0, 30 PPG) is on a bye every week left: team 0 plays nobody and loses.
    ctx = _ctx([30.0, 40.0, 10.0])
    lg = _setup(ctx)
    weeks = range(lg.first_week, lg.first_week + 6)
    off = replace(ctx, week_mult={w: np.array([0.0, 1.0, 1.0]) for w in weeks})
    rng = np.random.default_rng(2)
    on = np.stack([O.simulate_path(ctx, lg, d % 4, 1, rng, ctx.talent.mean(0))[0] for d in range(60)])
    out = np.stack([O.simulate_path(off, lg, d % 4, 1, rng, ctx.talent.mean(0))[0] for d in range(60)])
    assert (on[:, 0, 0] == 0).mean() > 0.6 and (out[:, 0, 0] == 0).mean() < 0.1


def test_division_winners_take_the_top_seeds():
    # Division 1: teams 0-2, division 2: teams 3-5. Teams 3 and 4 have the two best records,
    # but team 0 wins division 1, so the seeds are 3, 0, then 4, 1, 5, 2 by record.
    wins = np.array([7.0, 6.0, 3.0, 10.0, 10.0, 6.0])
    pf = np.array([1100.0, 1340.0, 900.0, 1440.0, 1370.0, 1200.0])
    div = np.array([1, 1, 1, 2, 2, 2])
    assert O.seed_order(wins, pf, div).tolist() == [3, 0, 4, 1, 5, 2]
    assert O.seed_order(wins, pf, None).tolist() == [3, 4, 0, 1, 5, 2]
    # Batched over leading axes, ties on record go to points for.
    w = np.stack([wins, np.array([6.0, 6.0, 3.0, 10.0, 10.0, 6.0])])
    assert O.seed_order(w, pf, div)[1].tolist() == [3, 1, 4, 5, 0, 2]


def _ctx_fa() -> O.Context:
    """Four QBs: team 0 has a 5-PPG starter (and a taxi QB), team 1 a 10-PPG
    one, and a 40-PPG QB sits unrostered."""
    ctx = _ctx([5.0, 40.0, 10.0])
    D, J = ctx.talent.shape[0], ctx.talent.shape[2]
    tal = np.concatenate([ctx.talent, np.full((D, 1, J), 40.0)], 1)
    return replace(ctx, player_id=ctx.player_id + ["gd"], position=np.array(["QB"] * 4, object),
                   talent=tal, share=np.ones((D, 4, J)))


def test_redraft_values_are_the_first_season_of_the_full_run():
    ctx = _ctx([15.0, 40.0, 20.0])
    lg = _setup(ctx)
    assets = [a for a in O.assets_for(lg, 0) if a.kind == "player"]
    rctx, rlg = O.redraft_context(ctx), replace(lg, picks=[])
    known, rknown = ctx.talent.mean(0), rctx.talent.mean(0)
    for i in range(30):
        # Same random stream per path: this season is drawn first, so it matches exactly.
        full = O.simulate_path(ctx, lg, i % 4, 2, np.random.default_rng([9, i]), known, assets, {0: 0})
        one = O.simulate_path(rctx, rlg, i % 4, 1, np.random.default_rng([9, i]), rknown, assets, {0: 0})
        assert (full[0][0] == one[0][0]).all() and (full[1][0] == one[1][0]).all()


def test_redraft_free_agent_can_be_picked_up_and_sets_the_waiver_level():
    ctx = _ctx_fa()
    lg = _setup(ctx)
    assert lg.roster.tolist() == [0, 0, 1, -1]
    rctx, rlg, fa = O.redraft_setup(O.redraft_context(ctx), lg, 0, free_agents=5, rank=1)
    assert rctx.waiver["QB"] == 40.0 and rlg.picks == [] and rlg.waiver_slots["QB"] >= 1
    assert [(a.ref, a.team, a.mode) for a in fa] == [(3, 0, "add")]
    rng = np.random.default_rng(4)
    # Waiver refill off: adding the 40-PPG free agent (cutting QB 0) wins team 0 the title far more often.
    plain = replace(rlg, waiver_slots={"QB": 0})
    d = np.stack([O.simulate_path(rctx, plain, i % 4, 1, rng, rctx.talent.mean(0), fa, {0: 0})[1] for i in range(60)])
    assert d[:, 0, 0].mean() > 0.4
    # The free agent alone never scores for anyone.
    places = np.stack([O.simulate_path(rctx, plain, i % 4, 1, rng, rctx.talent.mean(0))[0] for i in range(60)])
    assert (places[:, 0, 1] == 0).mean() > 0.6     # team 1 (10 PPG) beats team 0 (5 PPG)
