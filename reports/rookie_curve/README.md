# RB rookie prior on a draft-slot curve (2026-11-15/16)

Plan approved by ben 2026-11-15 ("Plan: curved draft-slot prior, tested on five-year
value"). ben's rule: every test is scored on five-year value, meaning a draft-day
forecast of each rookie's five-season discounted half-PPR points (0.80 a season,
washouts = 0) against what he actually scored.

**Change.** The rookie prior's draft term was a straight line in log pick. For RBs
it is now `alpha_pick * curve`, where curve = exp(-(pick - 1) / k), standardized
over drafted players. The curve falls fast over the first picks and then flattens
toward a floor. The spread also tightens toward the top picks:
`sigma_rookie * exp(sd_curve * curve + sd_undrafted * undrafted)`. Config is
`rookie_curve: {RB: 105}` and `rookie_curve_sd: [RB]` (model/talent.py `curve_k`,
`curve_sd`; model/data.py `pick_curve`; value/picks.py rebuilds the curve from the
saved `curve_k`). WR and TE stay on the line, per ben's choice on 2026-11-15.

**k.** k was chosen leave-one-class-out on five-year value, fitting (a + b·curve)²
to the 2013-21 classes. The surface is flat from about 90 to 200 at every
position. The rule was the k nearest zero top-10 bias among k within 1% of the
best RMSE, which gave RB 105, WR 200 and TE 90.

## Check 1: in sample (fits as of 2026-09-01, 2013-21 classes, `in_sample.csv`)

Draft-day futures came from the picks machinery (`value_rookies`), with
prior_shape and talent_scale off in every arm. All differences are paired, with ±1
standard error.

| | Line | Curve | Curve + spread |
|---|---|---|---|
| Picks 1-32 average miss | 164 | 151 (−13±14) | 151 (−13±14) |
| Picks 1-10 bias | +117 | −55 | −62 |
| Picks 1-32 log score | | +0.010±0.041 | +0.054±0.042 |
| All picks log score | | +0.016±0.008 | +0.018±0.008 |
| RB 1-32 log score | | +0.069±0.078 | +0.265±0.090 |

No band was worse by more than one standard error. The spread term's fitted
coefficient was RB −0.33, WR about 0 and TE +0.11, and TE scored worse with it.

## Check 2: out of sample (fits as of 2019-04-01, 2019-21 classes, `out_of_sample*.csv`)

| | Line | Curve | Curve + spread |
|---|---|---|---|
| RB average miss (62) | 119 | 112 (−7.4±2.7) | 114 (−5.6±2.1) |
| RB log score | | +0.048±0.014 | +0.041±0.014 |
| RB 1-32 log score (4) | | +0.060±0.072 | +0.142±0.031 |
| WR/TE picks 1-32 average miss (16) | 185 | 208 (+23±15) | 206 |
| WR 1-32 log score (13) | | −0.080±0.091 | −0.067±0.094 |

The 2019 fits see only 2013-18 RB/WR data. The RB fits needed tune 8000 and
draws 10000 to pass the gate; the default settings failed on form_ratio and
rho_form in both arms. The WR curve under-forecast the strong 2019-21 first-round
receivers.

**Decision (ben, 2026-11-15):** RBs only, with the curve and the spread.

## Board (fits as of 2026-11-14; RB line vs RB curve on the same data, QB/WR/TE fits shared)

The board moves are small. On the league tab the mean |change| is RB 0.016 DWAR and
WR 0.010 (the WR change comes through with-and-without). The largest moves are
Jadarian Price +0.33 (#97 to #79), Love +0.14 (#29 to #24) and Jonathon Brooks
+0.11. Love's fitted talent goes from 10.0 to 10.5 PPG: the tighter top-pick prior
keeps more weight on his draft slot after a slow start. The 2027 1.01 pick goes
from 3.87 to 3.37 and the 2027 1st round from 1.52 to 1.46. The talent_scale RB
factor of 1.17 was kept and not recalibrated. Under the plan's preseason view (as
of 2026-09-01), Love goes from 10.8 to 5.7 DWAR on the league tab, and from #1 to
#2 behind Nacua.

Also in this change: the value layer now takes draw indices from the fit with the
fewest draws (it assumed the first fit, QB). The QB fit as of 2026-11-14/15 just
missed R-hat (1.011) with 2000 draws and passed with 4000.

## Switch to a quadratic in log pick (ben, 2026-11-16/17)

ben asked for a quadratic in log pick for the RB prior and chose it over the curve
(decision card, 2026-11-17). Config: `rookie_pick_quad: [RB]` (alpha_pick2 times the
standardized squared log-pick z) plus `pick_sd: [RB]` (spread
`sigma_rookie * exp(sd_pick * z + sd_undrafted * undrafted)`). The curve option stays
in the code, switched off. Five-year value, RBs, scored the same way as above
(`quad_*_rb.csv`):

| | Line | Curve + spread | Quadratic + spread |
|---|---|---|---|
| In sample, average miss (192) | 122.5 | 118.8 | 118.5 (vs curve −0.2±0.8) |
| In sample, picks 1-32 miss (12) | 170 | 128 | 126 (vs curve −2.7±10.7) |
| In sample, top-10 bias (5) | +187 | −115 | −58 |
| Out of sample, average miss (62) | 119.4 | 113.7 | 116.9 (vs curve +3.1±1.5) |
| Out of sample, log score vs line | | +0.041±0.014 | +0.034±0.013 |
| Out of sample, picks 1-32 log score vs curve (4) | | | −0.075±0.047 |

Fitted values (2026-09-01): alpha_pick −0.85, alpha_pick2 −0.23, sd_pick +0.39. At
pick 3 (z ≈ −3.9) the prior spread is about 0.2 times the average rookie's, so
early-round backs lean much harder on draft slot than before.

Board (as of 2026-11-14, RB only refit, QB/WR/TE shared), league tab, line → curve →
quadratic: Love 2.19 → 2.33 → 2.80 DWAR (#29 → #24 → #15; fitted talent 10.0 → 10.5
→ 11.4 PPG), Price 0.82 → 1.15 → 1.12 (#97 → #79 → #82), Kyren Williams 2.55 → 2.44.
Everyone else moves less than 0.11; the mean |change| is RB 0.015 and WR 0.009. 2027
1.01 goes 3.87 → 3.37 → 3.42 and the 2027 1st 1.52 → 1.46 → 1.45. Preseason view
(as of 2026-09-01): Love goes 10.76 → 5.70 → 6.97 and stays #1 with the quadratic.
