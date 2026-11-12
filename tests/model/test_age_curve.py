import numpy as np

from dwar.model.talent import LATE_KNOT, age_curve_effect


def test_late_curve_is_quadratic_until_the_knot_then_adds_a_straight_decline():
    age_c = np.array([-4.0, 0.0, LATE_KNOT - 26.0, LATE_KNOT - 26.0 + 3.0])
    quad = age_curve_effect(age_c, 0.05, -0.01)
    late = age_curve_effect(age_c, 0.05, -0.01, "late", -0.1)
    assert np.allclose(late[:3], quad[:3])
    assert np.isclose(late[3] - quad[3], -0.3)
