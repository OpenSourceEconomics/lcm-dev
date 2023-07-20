"""Test the utility function of the analytical solution."""

import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    utility,
)
from numpy.testing import assert_almost_equal as aae

UTILTIY_TEST_CASES = [
    (1.0, False, 0.0, 0.0),
    (1.0, True, 0.0, 0.0),
    (1.0, True, 0.5, -0.5),
    (5.0, False, 0.5, np.log(5.0)),
    (5.0, True, 0.5, np.log(5.0) - 0.5),
    (-1.0, False, 0.5, -np.inf),
]


@pytest.mark.parametrize(
    ("consumption", "work_decision", "delta", "expected"),
    UTILTIY_TEST_CASES,
)
def test_utility(consumption, work_decision, delta, expected):
    util = utility(
        consumption=consumption,
        work_decision=work_decision,
        delta=delta,
    )
    aae(util, expected)
