"""Testing the simulate function of the analytical solution."""

import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    analytical_solution,
    simulate,
    simulate_cons_work_response,
)
from numpy.testing import assert_array_almost_equal as aaae

test_cases_simulate_cons_work_resp = [
    # First period
    {
        "inputs": {
            "period": 0,
            "wealth_levels": np.linspace(1, 100, 12),
            "wage": 1.0,
            "interest_rate": 0.0,
            "work_decision_function": lambda wealth, work_status: True,  # noqa: ARG005
            "consumption_function": lambda wealth, work_status: wealth,  # noqa: ARG005
            "work_status_last_period": [True] * 12,
        },
        "expected": {
            "consumption": np.linspace(1, 100, 12),
            "work_dec_vec": np.ones(12, dtype=bool),
            "wealth_next_period": np.ones(12, dtype=float),
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
            "work_status_last_period": [True] * 12,
        },
        "expected": {
            "consumption": np.linspace(1, 100, 12),
            "work_dec_vec": np.ones(12, dtype=bool),
            "wealth_next_period": np.ones(12, dtype=float),
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
            "work_status_last_period": [True] * 12,
        },
        "expected": {
            "consumption": np.linspace(1, 100, 12) ** 0.5,
            "work_dec_vec": [True] * 6 + [False] * 6,
            "wealth_next_period": (
                np.linspace(1, 100, 12) - np.linspace(1, 100, 12) ** 0.5
            )
            * 1.1
            + np.multiply([True] * 6 + [False] * 6, 2.0),
        },
    },
]

test_cases_simulate = (
    (
        1.0,
        1.0,
        np.linspace(1, 100, 12),
        1,
        2.0,
        0.0,
        np.linspace(1, 100, 12),
        np.repeat(False, 12),  # noqa: FBT003
    ),
)

test_cases_analytical_solution = [
    {
        "inputs": {
            "beta": 0.9,
            "wage": 1.0,
            "interest_rate": 0.0,
            "delta": 0.0,
            "num_periods": 1,
        },
        "expected": {
            "values": np.log(np.linspace(1, 100, 12)),
            "consumption": np.linspace(1, 100, 12),
            "work_decision": np.repeat(False, 12),  # noqa: FBT003
        },
        "wealth_grid": np.linspace(1, 100, 12),
    },
]

test_cases_analytical_solution_work_decision = [
    {
        "kwargs": {
            "beta": 0.95,
            "delta": 1.0,
            "initial_wealth_levels": np.linspace(1, 100, 12),
            "num_periods": 3,
            "wage": 10.0,
            "interest_rate": 0.0,
        },
        "expected": 2,
    },
]


@pytest.mark.parametrize("test_case", test_cases_simulate_cons_work_resp)
def test_simulate_cons_work_resp(test_case):
    """Test the simulate_cons_work_response function."""
    expected = test_case["expected"]
    inputs = test_case["inputs"]
    c, work_dec, wealth_next_period = simulate_cons_work_response(
        **inputs,
    )
    aaae(c, expected["consumption"])
    aaae(work_dec, expected["work_dec_vec"])
    aaae(wealth_next_period, expected["wealth_next_period"])


@pytest.mark.parametrize(
    (
        "beta",
        "delta",
        "initial_wealth_levels",
        "num_periods",
        "wage",
        "interest_rate",
        "expected_consumption",
        "expected_work_decision",
    ),
    test_cases_simulate,
)
def test_simulate(
    beta,
    delta,
    initial_wealth_levels,
    num_periods,
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
        num_periods=num_periods,
        wage=wage,
        interest_rate=interest_rate,
    )
    aaae(consumption[0], expected_consumption)
    aaae(work_decision[0], expected_work_decision)


@pytest.mark.parametrize("test_case", test_cases_analytical_solution)
def test_analytical_solution(test_case):
    """Test the analytical solution function."""
    grid = test_case["wealth_grid"]
    expected = test_case["expected"]
    inputs = test_case["inputs"]
    value_function, consumption_function, work_decision_function = analytical_solution(
        **inputs,
    )
    values_worker = [value_function[0](x, work_status=True) for x in grid]
    values_retired = [value_function[0](x, work_status=False) for x in grid]
    consumption = [consumption_function[0](x, work_status=True) for x in grid]
    work_decision = [work_decision_function[0](x, work_status=True) for x in grid]

    aaae(expected["values"], values_worker)
    aaae(expected["values"], values_retired)
    aaae(expected["consumption"], consumption)
    aaae(expected["work_decision"], work_decision)


@pytest.mark.parametrize("test_case", test_cases_analytical_solution_work_decision)
def test_simulate_work_dec(test_case):
    """Test that work decision is not False in all periods for lowest wealth level."""
    _, work_decision = simulate(**test_case["kwargs"])
    assert np.sum(work_decision.T[0]) == test_case["expected"]
