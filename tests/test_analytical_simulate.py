"""Testing the simulate function of the analytical solution."""

import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    simulate,
    simulate_cons_work_response,
)
from numpy.testing import assert_array_almost_equal as aaae

# ruff: noqa: FBT003

TEST_CASES_SIMULATE_CONS_WORK_RESPONSE = [
    # First period
    {
        "inputs": {
            "period": 0,
            "wealth_levels": np.linspace(1, 100, 12),
            "wage": 1.0,
            "interest_rate": 0.0,
            "work_decision_function": lambda wealth, work_status: True,  # noqa: ARG005
            "consumption_function": lambda wealth, work_status: wealth,  # noqa: ARG005
            "work_status_last_period": np.repeat(True, 12),
        },
        "expected": {
            "consumption": np.linspace(1, 100, 12),
            "work_decision": np.repeat(True, 12),
            "wealth_next_period": np.repeat(1.0, 12),
        },
    },
    # Second period
    {
        "inputs": {
            "period": 1,
            "wealth_levels": np.linspace(1, 100, 12),
            "wage": 1.0,
            "interest_rate": 0.0,
            "work_decision_function": lambda wealth, work_status: True,  # noqa: ARG005
            "consumption_function": lambda wealth, work_status: wealth,  # noqa: ARG005
            "work_status_last_period": np.repeat(True, 12),
        },
        "expected": {
            "consumption": np.linspace(1, 100, 12),
            "work_decision": np.repeat(True, 12),
            "wealth_next_period": np.repeat(1.0, 12),
        },
    },
    # Second period, nontrivial consumption and work function
    {
        "inputs": {
            "period": 1,
            "wealth_levels": np.linspace(1, 100, 12),
            "wage": 2.0,
            "interest_rate": 0.1,
            "work_decision_function": lambda wealth, work_status: wealth  # noqa: ARG005
            < 50,
            "consumption_function": lambda wealth, work_status: wealth  # noqa: ARG005
            ** 0.5,
            "work_status_last_period": np.repeat(True, 12),
        },
        "expected": {
            "consumption": np.linspace(1, 100, 12) ** 0.5,
            "work_decision": np.repeat(True, 6) + np.repeat(False, 6),
            "wealth_next_period": (
                np.linspace(1, 100, 12) - np.linspace(1, 100, 12) ** 0.5
            )
            * 1.1
            + np.multiply([True] * 6 + [False] * 6, 2.0),
        },
    },
]

TEST_CASES_SIMULATE = (
    (
        1.0,
        1.0,
        np.linspace(1, 100, 12),
        1,
        2.0,
        0.0,
        np.linspace(1, 100, 12),
        np.repeat(False, 12),
    ),
)

TEST_CASES_WORK_DECISION = [
    {
        "kwargs": {
            "beta": 0.95,
            "delta": 1.0,
            "initial_wealth_levels": np.linspace(1, 100, 12),
            "n_periods": 3,
            "wage": 10.0,
            "interest_rate": 0.0,
        },
        "expected": 2,
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES_SIMULATE_CONS_WORK_RESPONSE)
def test_simulate_cons_work_resp(test_case):
    """Test the simulate_cons_work_response function."""
    expected = test_case["expected"]
    inputs = test_case["inputs"]
    c, work_decision, wealth_next_period = simulate_cons_work_response(
        **inputs,
    )
    aaae(c, expected["consumption"])
    aaae(work_decision, expected["work_decision"])
    aaae(wealth_next_period, expected["wealth_next_period"])


@pytest.mark.parametrize(
    (
        "beta",
        "delta",
        "initial_wealth_levels",
        "n_periods",
        "wage",
        "interest_rate",
        "expected_consumption",
        "expected_work_decision",
    ),
    TEST_CASES_SIMULATE,
)
def test_simulate(
    beta,
    delta,
    initial_wealth_levels,
    n_periods,
    wage,
    interest_rate,
    expected_consumption,
    expected_work_decision,
):
    """Test the simulate function."""
    consumption, work_decision = simulate(
        beta=beta,
        delta=delta,
        initial_wealth_levels=initial_wealth_levels,
        n_periods=n_periods,
        wage=wage,
        interest_rate=interest_rate,
    )
    aaae(consumption[0], expected_consumption)
    aaae(work_decision[0], expected_work_decision)


@pytest.mark.parametrize("test_case", TEST_CASES_WORK_DECISION)
def test_simulate_work_dec(test_case):
    """Test that work decision is not False in all periods for lowest wealth level."""
    _, work_decision = simulate(**test_case["kwargs"])
    assert np.sum(work_decision.T[0]) == test_case["expected"]
