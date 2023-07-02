"""Testing the simulate function of the analytical solution."""

import numpy as np
import pytest
from lcm_dev import analytical_solution
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

test_cases_simulate = [
    {
        "inputs": {
            "c_func_vec": [
                lambda wealth, work_status: wealth,  # noqa: ARG005
                lambda wealth, work_status: wealth**2,  # noqa: ARG005
            ],
            "work_dec_func_vec": [
                lambda wealth, work_status: True,  # noqa: ARG005
                lambda wealth, work_status: False,  # noqa: ARG005
            ],
            "wealth_levels": np.linspace(1, 100, 12),
            "num_periods": 2,
            "wage": 2.0,
            "interest_rate": 0.0,
        },
        "expected": {
            "consumption": np.array(
                [
                    np.linspace(1, 100, 12),
                    (np.ones(12, dtype=float) + 1) ** 2,
                ],
            ),
            "work_dec_vec": np.array(
                [
                    np.ones(12, dtype=bool),
                    np.zeros(12, dtype=bool),
                ],
            ),
        },
    },
]

test_cases_analytical_solution = [
    {
        "inputs": {
            "wealth_grid": np.linspace(1, 100, 12),
            "simulation_grid": np.linspace(1, 100, 12),
            "beta": 0.9,
            "wage": 1.0,
            "interest_rate": 0.0,
            "delta": 0.0,
            "num_periods": 1,
        },
        "expected": {
            "v": {
                "worker": [np.log(np.linspace(1, 100, 12))],
                "retired": [np.log(np.linspace(1, 100, 12))],
            },
            "c": [np.linspace(1, 100, 12)],
            "work_dec_vec": [np.zeros(12, dtype=bool)],
        },
    },
]

test_cases_analytical_solution_work_decision = [
    {
        "kwargs": {
            "wealth_grid": np.linspace(1, 100, 12),
            "simulation_grid": np.linspace(1, 100, 12),
            "beta": 0.95,
            "wage": 10.0,
            "interest_rate": 0.0,
            "delta": 1.0,
            "num_periods": 3,
        },
        "expected": 2,
    },
]


@pytest.mark.parametrize("test_case", test_cases_simulate_cons_work_resp)
def test_simulate_cons_work_resp(test_case):
    """Test the simulate_cons_work_response function."""
    expected = test_case["expected"]
    inputs = test_case["inputs"]
    c, work_dec, wealth_next_period = analytical_solution.simulate_cons_work_response(
        **inputs,
    )
    aaae(c, expected["consumption"])
    aaae(work_dec, expected["work_dec_vec"])
    aaae(wealth_next_period, expected["wealth_next_period"])


@pytest.mark.parametrize("test_case", test_cases_simulate)
def test_simulate(test_case):
    """Test the simulate function."""
    expected = test_case["expected"]
    inputs = test_case["inputs"]
    cons, work_dec = analytical_solution.simulate(**inputs)
    aaae(cons, expected["consumption"])
    aaae(work_dec, expected["work_dec_vec"])


@pytest.mark.parametrize("test_case", test_cases_analytical_solution)
def test_analytical_solution(test_case):
    """Test the simulate function."""
    expected = test_case["expected"]
    inputs = test_case["inputs"]
    v, cons, work_dec = analytical_solution.analytical_solution(**inputs)
    aaae(v["worker"], expected["v"]["worker"])
    aaae(v["retired"], expected["v"]["retired"])
    aaae(cons, expected["c"])
    aaae(work_dec, expected["work_dec_vec"])


@pytest.mark.parametrize("test_case", test_cases_analytical_solution_work_decision)
def test_analytical_solution_work_dec(test_case):
    """Test that work decision is not False in all periods for lowest wealth level."""
    v, cons, work_dec = analytical_solution.analytical_solution(**test_case["kwargs"])
    assert np.sum(work_dec.T[0]) == test_case["expected"]
