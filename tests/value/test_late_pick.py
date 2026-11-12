import numpy as np
import polars as pl

from dwar.value.late_pick import pull_back, qualifying


def test_late_picks_in_early_years_with_production_are_scaled_after_the_current_season():
    last = pl.DataFrame({"player_id": ["L", "E", "U", "O", "S"], "season": [2025, 2025, 2025, 2025, 2025],
                         "ppg": [10.0, 12.0, 9.0, 10.0, 5.0]})
    static = pl.DataFrame({"player_id": ["L", "E", "U", "O", "S"], "draft_round": [7, 1, None, 5, 6],
                           "draft_year": [2025, 2024, None, 2019, 2024], "first_season": [2025, 2024, 2024, 2019, 2024]})
    hit = qualifying(last, static, 3, 3, 8.0)
    # L: 7th round, year 1. E: 1st round. U: undrafted, year 2. O: year 7. S: too few PPG.
    assert hit.tolist() == [True, False, True, False, False]
    talent = np.full((2, 5, 4), 10.0)
    out = pull_back(talent, np.array([2026, 2027, 2028, 2029]), 2026, np.full(5, 2025.0), hit, (1.0, 0.85, 0.8))
    assert np.allclose(out[:, 0, :], [10.0, 8.5, 8.0, 8.0])   # current season untouched; h=2, 3, 3+
    assert np.allclose(out[:, 1, :], 10.0)
