# Old star RB pull-back (2026-11-10)

ben flagged that the board kept aging star RBs (McCaffrey) too high. Cases: RBs
aged 29+ (Sept 1) coming off a season of 12+ half-PPR PPG in 8+ games, forecast
one to three seasons out (outcome seasons with 4+ games), from the step-through
forecasts at season start (rb_usage; rb_career for 2014-16). 34 cases.

| Forecast | MAE (PPG) |
|---|---|
| Model | 3.49 |
| Model, small cut (x0.98 one season out, x0.87 after) | 3.36 |
| Last PPG x historical keep-rate (rates from seasons before each test period) | 2.82 |
| 75% rule + 25% model (kept) | 2.93 |
| Last PPG unchanged | 7.01 |

By period, model -> 75% blend: 2014-16 2.83 -> 1.82, 2017-19 2.54 -> 1.84,
2020-22 4.45 -> 3.79, 2023-25 5.04 -> 5.48 (7 cases; Derrick Henry never
declined).

Keep-rates since 1999 (total PPG kept, played 4+ games): 71% one season out,
54% two, 45% three. The model's one-season pull-back already matched (72%); it
kept 61% two seasons out.

In the value layer (value/old_star.py, config/value.yaml `old_star`), the
keep-rates are measured from completed seasons as of the run date, and each
future season after the current one moves 75% of the way from the model's mean
toward last full season's PPG x keep[h]. Survival and games played follow the
lower talent.
