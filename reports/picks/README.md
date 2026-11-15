# Rookie draft pick values (2026-11-14/15)

Plan approved by ben 2026-11-14 (5 rounds x 12, rookie drafts 2027-2031, 12 teams,
model draft-day order instead of ADP, same window as players). `value/picks.py`.

Each NFL class 2013-2026 (drafted QB/RB/WR/TE) is put in draft-day order by the
model's draft-time view. Each rookie gets the full distribution of DWAR futures
from the fit's rookie prior plus the board's projection and availability models
(rookie-season games/survival models learned from classes since 2006). Slot k =
the k-th rookie's draws pooled over classes. Busts count as zero.

## Check: model vs real 5-season points, 2013-21 classes (`check.json`, Your league)

| Draft-day band | n | Model | Real |
|---|---|---|---|
| 1.01-1.04 | 36 | 766 | 769 |
| 1.05-1.12 | 72 | 496 | 541 |
| Round 2 | 108 | 393 | 448 |
| Round 3 | 108 | 248 | 262 |
| Rounds 4-5 | 216 | 164 | 146 |

All: 311 vs 324; 50% / 90% ranges cover 56% / 94%. By position QB 423/435,
RB 306/307, WR 288/312, TE 263/274.

First version (straight-line prior, rookie games independent of talent, no RB
factor) was 15-22% low from 1.05 to round 3 (RB 264 vs 326, 3.4 se). Causes found:

1. The rookie prior is linear in log pick; the fit's own drafted rookies sit below
   it for the top 8 NFL picks (RB -0.41, WR -0.19 latent) and above it for picks
   17-100. `picks.prior_shape`: quadratic in log pick fitted per draw to the
   rookies' posterior deviations (understates the bend, since low-data rookies'
   deviations are shrunk).
2. Rookie-year games were drawn from pick alone. Given pick, fit talent and games
   correlate ~0.3 (QB .30, RB .30, WR .33, TE .20). `picks.games_talent_corr`:
   Gaussian copula re-pairing each rookie's games and talent draws (marginals kept).
3. RBs who start hot win the job: real points 326 vs real games x fit talent 289.
   The model counts this for no player. ben chose (2026-11-15) one RB factor from
   history: `picks.talent_scale: {RB: 1.17}` (1.175 overall; 1.13-1.22 leaving each
   class out).

Draft terms in survival/games for all positions were also tried for picks: barely
moved the check, not used (board stays as validated).
