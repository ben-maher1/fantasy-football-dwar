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
