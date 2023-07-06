"""Test the utility function of the analytical solution."""
# ruff: noqa: FBT003

import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    utility,
)
from numpy.testing import assert_almost_equal as aae

utility_test_cases = [
    (1.0, False, 0.0, 0.0),
    (1.0, True, 0.0, 0.0),
    (1.0, True, 0.5, -0.5),
    (5.0, False, 0.5, np.log(5.0)),
    (5.0, True, 0.5, np.log(5.0) - 0.5),
    (-1.0, False, 0.5, -np.inf),
]


@pytest.mark.parametrize(
    ("consumption", "work_dec", "delta", "expected"),
    utility_test_cases,
)
def test_utility(consumption, work_dec, delta, expected):
    util = utility(
        consumption=consumption,
        work_dec=work_dec,
        delta=delta,
    )
    aae(util, expected)
