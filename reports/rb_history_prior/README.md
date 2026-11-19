# RB rookie prior and age curve from 1999+ history (2026-11-18/19)

Plan approved by ben 2026-11-18 ("Plan: RB rookie prior learned from longer draft
history", start year 1999, with the age curve tested as its own arm). Every test is
scored on five-year value: a forecast of each player's five-season discounted
half-PPR points (0.80 a season, washouts = 0) against what he actually scored.

**Result: nothing is kept.** The history-learned draft prior is worse at the top of
the draft, and the history-learned age curve ties. The config is unchanged
(`history_prior` off, `picks.history_first` stays 2006).

## Method (model/history.py, `history_prior` in config/model.yaml, off)

A season-level RB fit (the same likelihood, age curve and quadratic-in-log-pick
rookie prior with log-pick spread, but no game-by-game form, usage or situation
terms) on 1999-2012, as of 2013-03-01. It doesn't overlap the main fit's
2013+ seasons, so the hand-off is a clean sequential update. Players drafted
before 1999, or undrafted and already playing in 1999, count as veterans
(`censored_start`). That leaves 468 rookies, including 18 top-12 picks.

Handed to the main 2013+ RB fit as Normal(mean, sd) priors:
- "pick" arm: the rookie prior's shape by draft slot (alpha_pick, alpha_pick2,
  alpha_undrafted, sd_pick, sd_undrafted), converted draw by draw to the main fit's
  log-pick standardization (`convert_pick_terms`, tested exact).
- "age" arm: beta1 and beta2.

The average rookie's level and spread (alpha0, sigma_rookie) stay the main fit's
own. For the scale check, the 1999-2012 average rookie sits lower than the 2013+ one
(alpha0 1.64 vs 1.84 in the same basis), which is a real era difference that
should not leak in.

History fit: alpha_pick −0.67±0.08, alpha_pick2 −0.17±0.07, sd_pick
+0.03±0.05. The main fit alone had sd_pick +0.39, so over 1999-2012 the spread
barely tightens toward the top picks. With the hand-off, the main fit's sd_pick is
+0.09.

## Test 1: in sample (fits as of 2026-09-01, RB classes 1999-2021, `in_sample.csv`)

| | Current (quad + spread) | History prior | Paired difference |
|---|---|---|---|
| Picks 1-32 average miss (53) | 174.4 | 177.6 | +3.2±2.2 |
| Picks 1-32 log score | | | −0.067±0.019 |
| Picks 1-12 average miss (22) | 196.3 | 203.5 | +7.2±4.3 |
| Picks 1-12 bias | +90 | +101 | |
| 1999-2012 picks 1-12 bias (17) | +131 | +141 | |
| 2013-21 picks 1-12 bias (5) | −49 | −34 | |
| All picks average miss (451) | 121.7 | 121.5 | −0.3±0.3 |
| Top-32 50% / 90% coverage | 53% / 94% | 55% / 100% | |

The wider spread at the top doesn't lower top picks' forecasts. On the square-root
scale, a wider spread raises the expected points, so top picks are forecast even
higher, and their 90% ranges cover everything (too wide). The current model already
over-forecasts the 1999-2012 top-12 backs by about 131 five-year points while
under-forecasting 2013-21's by 49. The eras differ in the level of top picks, not
in the spread.

## Test 2: out of time (fits as of 2019-04-01, 2019-21 classes, `out_of_sample.csv`)

All 62: average miss 117.3 → 114.7 (−2.5±0.8), log score +0.027±0.012, mostly from
picks 101+. Picks 1-32 (4): −3.3±5.3, log −0.043±0.032.

## Test 4: age curve (fits as of 2019-04-01, every RB's 2019-23 five-year value, `veterans_2019.csv`)

| | Current | Age curve from history |
|---|---|---|
| Age 25+ average miss (112) | 75.3 | 76.9 |
| Age 25+ log score | | +0.029±0.027 |
| Under 25 (55) | 124.8 | 126.6 |

This is a tie. The main fit's own curve was already close to the history one
(beta1/beta2 −0.044/−0.023 vs −0.032/−0.026).

## Rookie-season models from 1999 (`rookie_season_models_1999.csv`)

The rookie-year games and survival models for pick values were trained from 1999
instead of 2006, scored on the same 1999-2021 draft-day check. All picks: −0.3±0.2.
Picks 1-32: +1.0±1.1. This is a tie, so history_first stays 2006.

Fits: the RB main fits needed 7000 draws (as of 2026-09-01) and tune 8000 / 12000
draws (as of 2019-04-01) with the history prior. Fewer draws just missed R-hat or
tail ESS.
