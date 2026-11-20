# Title value prototype (2027 season, Your league tab)

Plan approved by ben, 2026-11-20 (Claude Doc "Plan: championship-based player value").
Code: `src/dwar/value/title.py` (simulator), `src/dwar/value/title_run.py` (runner).
Nothing on the board changes; DWAR output is unchanged.

**Result: title value tracks DWAR closely.** For 2027, a player is worth about 3.8
percentage points of title odds per DWAR win across the board. The top 10 earn about
10% more per win, and QBs and TEs gain on WRs. Nobody in DWAR's top 25 moves more than
6 places. The largest effect is team context: the same player adds about 12 times more
title odds on a contender than on a rebuilder.

## Setup

- Fits as of 2026-11-20 (data through week 11 of 2026). All four fits passed (no
  divergences, R-hat at most 1.0095). Refit took 95 minutes.
- Format: `1qb_3flex_half_ppr` (12 teams, 25 active: 3 QB, 8 RB, 10 WR, 4 TE).
  Assumed season until ben gives his: 14-week regular season, 6 playoff teams, 2 byes,
  weeks 15-17 playoffs, points-for tiebreak.
- 40,000 simulated leagues, one posterior draw each, about 3.3 minutes on 4 cores.
- Rosters are dealt by DWAR rank plus noise (0.5 on the rank normal score), snake per
  position. Missed weeks come in stretches averaging 2.1 weeks (measured, 2013-25).
- Each player's value is his team's title chance with him minus without him (spot to
  the waiver wire), on the same simulated weeks.

## Checks

| Check | Result |
| --- | --- |
| Weekly spread of a starter's score vs real (2022-25, 10+ games) | Too narrow at first: RB 23%, WR 15%, TE 17% below real; QB right. Fixed with spread factors RB 1.35, WR 1.20, TE 1.25 (`--spread`) |
| Team weekly score SD vs DWAR's measured SD | 21.6 vs 22.9 (within 10%) after the fix; 18.4 before |
| Title share of the best preseason roster | 36%; second best 21%; bottom four under 3% combined |
| Roster noise 0.5 vs 1.0 (uncalibrated spread) | Rank correlation with DWAR 0.986 vs 0.982: noise barely matters |
| Smoothing curve vs per-player values (top 30) | Curve is fine for RB/WR, but it underrates top QBs by about 0.5 points. Per-player values (SE about 0.13 at 40,000 leagues) are used below. The curve needs a narrower bandwidth at the top before the multi-year build |

## Title value against 2027 DWAR

Rank correlation, DWAR's top 150: 0.984. Players moving more than 10 places: 0 of the top
25, 3 of the top 50, 28 of the top 150 (mostly ranks 100-150, where values sit within a
few tenths of a point).

| DWAR rank band | Title odds per DWAR win (pct points) |
| --- | --- |
| 1-10 | 4.24 |
| 11-30 | 3.80 |
| 31-60 | 3.77 |
| 61-100 | 3.85 |
| 101-150 | 3.82 |

| Position (top 150) | Title odds per DWAR win |
| --- | --- |
| TE | 4.27 |
| QB | 4.18 |
| RB | 3.96 |
| WR | 3.63 |

Biggest movers in the top 50: Dak Prescott (DWAR 48 to title 33), Harold Fannin Jr.
(51 to 37) and C.J. Stroud (57 to 44) up; Zay Flowers (35 to 48) down. The pattern is
position-wide. A likely reason (not tested): QB and TE scores are a bigger part of what
separates teams week to week, while WR value is spread over deep WR and flex slots.

Contender vs rebuilder (top and bottom third of preseason rosters): top 50 players add
about 12 times more title odds on a contender. For example, Gibbs adds 14.4 points on a
contender and 1.0 on a rebuilder (10.3 on a typical team).

Full table: `top150_2027.csv` (`title_pct` per-player value, `title_curve` smoothed,
ranges from the curve over posterior draws). Run settings and checks: `run_2027_cal.json`.

## Limits

- No same-NFL-team links (QB with his receivers), no in-season trades, waivers or role
  changes. The last three would likely make titles less predictable than 36% for the best
  roster.
- Weekly spread is by position and talent level only. It cannot tell a boom-or-bust
  player from a steady one at the same average.
- 2027 only, and the 2027 rookie class is not in the pool.

## Reproduce

```bash
uv run python -c "from datetime import date; from dwar.pipeline import run; run(date(2026, 11, 20), '1qb_3flex_half_ppr')"
uv run python -m dwar.value.title_run --as-of 2026-11-20 --format 1qb_3flex_half_ppr \
    --leagues 40000 --noise 0.5 --spread RB=1.35,WR=1.2,TE=1.25 --tag _cal
```

## Update 2026-11-20 (later): DWAR check and TEs

**Simulator fix.** Lineups were set by each draw's true talent, so simulated managers knew
more than real ones. They now use posterior mean talent (`LeagueInputs.known`). Rerun
(`--tag _known`): rank correlation 0.983, top-25 max move 7, 3 of the top 50 move more than
10 (Kittle and Fannin up 11, Prescott up 10; Tyson down 11). Title odds per DWAR win: ranks
1-10 4.16, 11-30 3.79, 31-60 3.50, 61-150 2.81; by position QB 3.77, TE 3.87, RB 3.53,
WR 3.21. Contender vs rebuilder still about 11x. The star premium is larger than first
reported, and the position pattern holds.

**DWAR's with-and-without step, checked** (`decomp.py` in the thread's scratch work; per
starter, points lost per team week, 1,500 leagues each; starters = top 12 QB/TE, top 24 RB/WR):

| Setup | QB | TE | RB | WR |
| --- | --- | --- | --- | --- |
| DWAR's own grid | 3.13 | 2.83 | 4.50 | 4.75 |
| E0: simulator on DWAR's setup | 3.15 | 2.84 | 4.48 | 4.67 |
| E1a: 2027 projected pool (mean talent) | 3.32 | 2.94 | 4.24 | 4.44 |
| E1c: + true talent spread, lineups by mean | 3.69 | 3.24 | 4.85 | 5.09 |
| E2c: + each player's own games share | 4.90 | 3.29 | 5.30 | 5.52 |
| E4c: + injury stretches, rostered by DWAR with noise | 5.12 | 3.43 | 5.41 | 5.52 |

- The two engines agree on DWAR's setup (within 2%).
- **Main finding:** DWAR gives every rostered player at a position the same chance of
  playing a week (QB 72%, RB 74%, WR 79%, TE 82%). Backups really play less. Over 2014-25,
  players ranked by last season's PPG played this share of next season's games:
  QB 84% (top 12) vs 58% (13-36); RB 77% vs 55%; WR 83% vs 67%; TE 77% vs 73%.
  Using each player's own expected share raises a starter's value by QB +33%, RB +9%,
  WR +8%, TE +2%.
- Talent spread (true talent differs from its mean) adds 10-15% at every position.
  That changes the scale, not the order.
- Injury stretches change nothing.

**TEs.** In the title simulation TEs gain about 20% per DWAR win on WRs. About half of that
is points: the simulated league says a TE costs his team about 12% more points, relative to
DWAR, than a WR does. That comes from a thinner 2027 TE bench and from dynasty rostering,
not from backup playing time (TE backups play nearly as much as starters). The other half is
how points turn into titles: 4.3 title points per 100 points lost for TEs vs 4.0 for WRs.
That part is not explained yet.
