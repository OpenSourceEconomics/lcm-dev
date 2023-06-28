"""Test the utility function of the analytical solution."""

import numpy as np
import pytest
from lcm_dev import analytical_solution
from numpy.testing import assert_almost_equal as aae

utility_test_cases = [
    (
        {
            "consumption": 1.0,
            "work_dec": False,
            "delta": 0.0,
        },
        0.0,
    ),
    (
        {
            "consumption": 1.0,
            "work_dec": True,
            "delta": 0.0,
        },
        0.0,
    ),
    (
        {
            "consumption": 1.0,
            "work_dec": True,
            "delta": 0.5,
        },
        -0.5,
    ),
    (
        {
            "consumption": 5.0,
            "work_dec": False,
            "delta": 0.5,
        },
        np.log(5.0),
    ),
    (
        {
            "consumption": 5.0,
            "work_dec": True,
            "delta": 0.5,
        },
        np.log(5.0) - 0.5,
    ),
    (
        {
            "consumption": -1.0,
            "work_dec": False,
            "delta": 0.5,
        },
        -np.inf,
    ),
]


@pytest.mark.parametrize(("inputs", "expected"), utility_test_cases)
def test_utility(inputs, expected):
    aae(analytical_solution.utility(**inputs), expected)
