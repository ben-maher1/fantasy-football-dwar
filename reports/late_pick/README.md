# Late-drafted young RB pull-back (2026-11-10)

Cases: RBs drafted in round 3 or later, or undrafted, whose last full season
(8+ games) was in career years 1-3 with 8+ half-PPR PPG; the model's
season-start forecasts of the next 1-3 seasons (outcomes with 4+ games), from the
step-through tests (rb_usage; rb_career for 2014-16). 276 cases. Study script:
scratchpad `study/latepick.py`.

Actual / forecast, by seasons after the last full season (each period's factors
measured on the other three periods):

| Test period | 1 | 2 | 3 | MAE model -> scaled |
|---|---|---|---|---|
| 2014-16 | 0.996 | 0.877 | 0.805 | 3.17 -> 3.08 |
| 2017-19 | 1.014 | 0.832 | 0.811 | 3.08 -> 2.83 |
| 2020-22 | 0.996 | 0.840 | 0.772 | 3.23 -> 3.30 |
| 2023-25 | 1.011 | 0.862 | 0.828 | 3.33 -> 3.15 |

All periods: 1.004 / 0.854 / 0.806; config uses 1.0 / 0.85 / 0.80 (seasons beyond
three keep 0.80). Blending toward last PPG x the historical keep-rate (as for old
star RBs) helped less (best: 25% blend, MAE 3.19 -> 3.14), since a young
player's last season is noisy. Rounds 1-2 need no change (forecast bias -0.2).

## Follow-up and decision: off (2026-11-10)

ben flagged Achane (round 3; 16.1 / 15.3 / 18.1 PPG in 2023-25) as cut far too
hard, and asked why young RBs are over-forecast at all, without a round line.
Cases widened to career years 1-4 off a 5+ PPG season, each period fitted on
the other three (scratchpad `study/anchor*.py`, `study/fade*.py`):

- The model's random-walk talent has no pull back toward a typical level, so
  seasons two or three out run high; for young RBs the gap is mostly in rounds
  3-7 (actual / forecast about 0.88 two and three seasons out), not rounds 1-2
  (1.02 / 0.95). Undrafted backs fall in between (about 80 cases).
- An anchor (draft-slot level mixed with the player's own career PPG, weighted by
  games, with the forecast drifting toward it) did not work: the games weight
  ran to infinity, the fit was unstable, and 2023-25 got worse in every version.
- Smooth fades: one fade for all young RBs (x0.94 / x0.895) had the best MAE
  (3.44 -> 3.37; this rule 3.39) but left rounds 1-2 about 0.75 PPG low; a log
  draft-pick term and a production-level term added nothing. A milder fitted
  round 3+ fade (x0.89 / x0.855) was about unbiased in every group (this rule:
  late-round 14+ PPG backs about 1 PPG low), MAE 3.39.

All versions were within about 0.07 PPG MAE of each other. ben chose to turn the
rule off (`late_pick: {}` in config/value.yaml); the code and tests stay.
