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
