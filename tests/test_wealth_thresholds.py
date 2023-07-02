"""Test the wealth threshold computation of the analytical solution."""

import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    _compute_wealth_tresholds,
    _evaluate_piecewise_conditions,
    retirement_threshold,
    root_function,
    wealth_thresholds_kinks_discs,
)
from numpy.testing import assert_almost_equal as aae
from numpy.testing import assert_array_almost_equal as aaae

test_cases_ret_threshold = [
    # 1 period before end of life
    (
        3.0,
        0.0,
        0.95,
        0.1,
        1,
        3 * np.exp(-0.1 * (1.95) ** (-1)) / (1 - np.exp(-0.1 * (1.95) ** (-1))),
    ),
    # 2 periods before end of life
    (
        3.0,
        0.1,
        0.95,
        0.1,
        2,
        (3 / 1.1 * np.exp(-0.1 * (1.95 + 0.95**2) ** (-1)))
        / (1 - np.exp(-0.1 * (1.95 + 0.95**2) ** (-1))),
    ),
]


test_cases_root_fct = [
    # Value of root function without v_prime
    {
        "inputs": {
            "wealth": 10,
            "consumption_lb": lambda x: 10,  # noqa: ARG005
            "consumption_ub": lambda x: 20,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: 1,  # noqa: ARG005
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "delta": 0.1,
        },
        "expected": np.log(1 / 2),
    },
    # Value of root function with non-trivial v_prime
    {
        "inputs": {
            "wealth": 50,
            "consumption_lb": lambda x: 10,  # noqa: ARG005
            "consumption_ub": lambda x: 20,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: -(wealth**2),  # noqa: ARG005
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "delta": 0.1,
        },
        "expected": np.log(1 / 2)
        - 0.95 * ((50 - 10) + 3) ** 2
        + 0.95 * ((50 - 20) + 3) ** 2,
    },
]

test_cases_wealth_thresholds_kinks_discs = [
    # Test root finding for trivial value function
    {
        "inputs": {
            "v_prime": lambda wealth, work_status: 0,  # noqa: ARG005
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "delta": 0.1,
            "bracket": 0,
            "consumption_policy": [
                lambda wealth: None,  # noqa: ARG005
                lambda wealth: wealth,
                lambda wealth: np.exp(0),  # noqa: ARG005
            ],
            "ret_threshold": 10,
            "wealth_thresholds": [-10, -10],
        },
        "expected": 1.0,
    },
    # Test root finding for non-trivial value function
    {
        "inputs": {
            "v_prime": lambda wealth, work_status: wealth,  # noqa: ARG005
            "wage": 0.0,
            "interest_rate": 0,
            "beta": 0.95,
            "delta": 0.0,
            "bracket": 0,
            "consumption_policy": [
                None,
                lambda wealth: wealth + 1,
                lambda wealth: -wealth + 1,
            ],
            "ret_threshold": 10,
            "wealth_thresholds": [None, -10],
        },
        "expected": 0.0,
    },
]

test_cases_wealth_threshold = [
    # Check analytical calculation of thresholds
    {
        "inputs": {
            "v_prime": None,
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "delta": 0.1,
            "tau": 1,
            "consumption_policy": [],
        },
        "expected": {
            "len_array": 4,
            "wealth_thresholds": [
                -np.inf,
                3 / 0.95,
                3 * np.exp(-0.1 / 1.95) / (1 - np.exp(-0.1 / 1.95)),
                np.inf,
            ],
        },
    },
    # Check threshold calculation via root finding
    {
        "inputs": {
            "v_prime": lambda wealth, work_status: wealth,  # noqa: ARG005
            "wage": 0.0,
            "interest_rate": 0,
            "beta": 0.95,
            "delta": 0.1,
            "tau": 2,
            "consumption_policy": [
                None,
                lambda wealth: -wealth + 1.0,
                lambda wealth: wealth + 1.0,
                lambda wealth: -wealth + 1.0,
                None,
            ],
        },
        "expected": {
            "len_array": 6,
            "wealth_thresholds": [
                -np.inf,
                0.0,
                0.0,
                0.0,
                0.0,
                np.inf,
            ],
        },
    },
]

test_cases_piecewise_conditions = [
    # wealth, thresholds, expected
    (0.5, [0, 1, 2, 3], [True, False, False]),
    (10, [0, 10, 20, 30], [False, True, False]),
]


@pytest.mark.parametrize(
    ("wage", "interest_rate", "beta", "delta", "tau", "expected"),
    test_cases_ret_threshold,
)
def test_retirement_threshold(wage, interest_rate, beta, delta, tau, expected):
    """Test the retirement threshold function."""
    retirement_threshold_solution = retirement_threshold(
        wage=wage,
        interest_rate=interest_rate,
        beta=beta,
        delta=delta,
        tau=tau,
    )
    aae(retirement_threshold_solution, expected)


@pytest.mark.parametrize("test", test_cases_root_fct)
def test_root_fct(test):
    """Test the root function."""
    sol_root_fct = root_function(**test["inputs"])
    aae(sol_root_fct, test["expected"])


@pytest.mark.parametrize("test", test_cases_wealth_thresholds_kinks_discs)
def test_wealth_thresholds_kinks_discs(test):
    """Test the wealth thresholds, kinks and discontinuities function."""
    aae(
        wealth_thresholds_kinks_discs(**test["inputs"]),
        test["expected"],
    )


@pytest.mark.parametrize("test", test_cases_wealth_threshold)
def test_compute_wealth_thresholds_length(test):
    """Test the wealth thresholds function."""
    wt = _compute_wealth_tresholds(**test["inputs"])
    assert len(wt) == test["expected"]["len_array"]
    aae(wt, test["expected"]["wealth_thresholds"])


@pytest.mark.parametrize(
    ("wealth", "thresholds", "expected"),
    test_cases_piecewise_conditions,
)
def test_evaluate_piecewise_conditions(wealth, thresholds, expected):
    """Test the piecewise conditions function."""
    aaae(
        _evaluate_piecewise_conditions(
            wealth=wealth,
            wealth_thresholds=thresholds,
        ),
        expected,
    )
