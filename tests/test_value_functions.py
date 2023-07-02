"""Test the value functions of the analytical solution."""
# ruff: noqa: FBT003
import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    _construct_model,
    _value_function,
    value_function_last_period,
    value_function_retirees,
    value_function_workers,
)
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_almost_equal as aae

test_cases_value_func_retirees = [
    (10.0, 1.0, 0, 0.0, np.log(10.0)),
    (20.0, 1.0, 0, 0.0, np.log(20.0)),
    (10.0, 0.95, 1, 0.0, np.log(10.0 / 1.95) + 0.95 * np.log(10.0 - (10.0 / 1.95))),
    (20.0, 0.95, 1, 0.0, np.log(20.0 / 1.95) + 0.95 * np.log(20.0 - (20.0 / 1.95))),
    (
        10.0,
        0.95,
        2,
        0.1,
        np.log(10.0) * (1 + 0.95 + 0.95**2)
        - np.log(1 + 0.95 + 0.95**2) * (1 + 0.95 + 0.95**2)
        + 0.95 * (np.log(0.95) + np.log(1.1)) * (1 + 0.95 + 0.95),
    ),
    (
        20.0,
        0.95,
        2,
        0.1,
        np.log(20.0) * (1 + 0.95 + 0.95**2)
        - np.log(1 + 0.95 + 0.95**2) * (1 + 0.95 + 0.95**2)
        + 0.95 * (np.log(0.95) + np.log(1.1)) * (1 + 0.95 + 0.95),
    ),
]


test_cases_value_func_workers = [
    # Value function without continuation value
    {
        "inputs": {
            "wealth": 10.0,
            "beta": 1.0,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": np.bool_(True),
            "work_dec_func": lambda wealth, work_status: np.bool_(True),  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: 0.0,  # noqa: ARG005
        },
        "expected": np.log(10.0) - 0.1,
    },
    # Value function with continuation value
    {
        "inputs": {
            "wealth": 10.0,
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": np.bool_(True),
            "work_dec_func": lambda wealth, work_status: np.bool_(True),  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
        },
        "expected": np.log(10.0) - 0.1 + 0.95 * np.log(1.0),
    },
    # Value function with continuation value and nontrivial consumption
    {
        "inputs": {
            "wealth": 10.0,
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": np.bool_(True),
            "work_dec_func": lambda wealth, work_status: np.bool_(True),  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth / 2,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
        },
        "expected": np.log(10.0 / 2) - 0.1 + 0.95 * np.log(1.0 + 10.0 / 2),
    },
]

test_cases_value_func = [
    # Value function retirees
    {
        "inputs": {
            "wealth": 10.0,
            "work_status": np.bool_(False),
            "work_dec_func": None,
            "c_pol": None,
            "v_prime": None,
            "beta": 0.95,
            "delta": None,
            "tau": 2,
            "interest_rate": 0.1,
            "wage": None,
        },
        "expected": np.log(10.0) * (1 + 0.95 + 0.95**2)
        - np.log(1 + 0.95 + 0.95**2) * (1 + 0.95 + 0.95**2)
        + 0.95 * (np.log(0.95) + np.log(1.1)) * (1 + 0.95 + 0.95),
    },
    # Value function workers
    {
        "inputs": {
            "wealth": 10.0,
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": np.bool_(True),
            "work_dec_func": lambda wealth, work_status: np.bool_(True),  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth / 2,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
            "tau": None,
        },
        "expected": np.log(10.0 / 2) - 0.1 + 0.95 * np.log(1.0 + 10.0 / 2),
    },
]

test_cases_value_func_last_period = [
    (10, np.bool_(False), np.log(10)),
    (10, np.bool_(True), np.log(10)),
    (-1, np.bool_(False), -np.inf),
    (-1, np.bool_(True), -np.inf),
]

test_cases_construct_model = [
    {
        "wealth": 10,
        "inputs": {
            "delta": None,
            "num_periods": 1,
            "param_dict": {
                "beta": None,
                "wage": None,
                "interest_rate": None,
                "tau": None,
            },
        },
        "expected": ([np.log(10)], [10], [np.bool_(False)]),
    },
    {
        "wealth": 10.0,
        "inputs": {
            "delta": 0.1,
            "num_periods": 2,
            "param_dict": {
                "beta": 0.95,
                "wage": 1.0,
                "interest_rate": 0.0,
                "tau": None,
            },
        },
        "expected": (
            [
                np.log(11 / 1.95) - 0.1 + 0.95 * np.log(11 - 11 / 1.95),
                np.log(10),
            ],
            [
                11 / 1.95,
                10,
            ],
            [
                np.bool_(True),
                np.bool_(False),
            ],
        ),
    },
]


@pytest.mark.parametrize(
    ("wealth", "beta", "tau", "interest_rate", "expected"),
    test_cases_value_func_retirees,
)
def test_value_func_retirees(wealth, beta, tau, interest_rate, expected):
    solution = value_function_retirees(
        wealth=wealth,
        beta=beta,
        tau=tau,
        interest_rate=interest_rate,
    )
    aaae(expected, solution)


@pytest.mark.parametrize("test_case", test_cases_value_func_workers)
def test_value_func_workers(test_case):
    solution = value_function_workers(**test_case["inputs"])
    aaae(test_case["expected"], solution)


@pytest.mark.parametrize("test_case", test_cases_value_func)
def test_value_func(test_case):
    solution = _value_function(
        **test_case["inputs"],
    )
    aaae(test_case["expected"], solution)


@pytest.mark.parametrize(
    ("wealth", "work_status", "expected"),
    test_cases_value_func_last_period,
)
def test_value_func_last_period(wealth, work_status, expected):
    solution = value_function_last_period(
        wealth=wealth,
        work_status=work_status,
    )
    aae(expected, solution)


@pytest.mark.parametrize("test_case", test_cases_construct_model)
def test_construct_model(test_case):
    """Test fully constructed model.

    First period value function, consumption policy and work decision function.

    """
    v_vec, c_vec, work_dec_vec = _construct_model(
        **test_case["inputs"],
    )
    wealth = test_case["wealth"]

    sol_work_dec = [
        work_dec(wealth, work_status=np.bool_(True)) for work_dec in work_dec_vec
    ]
    sol_v = [v(wealth, work_status=np.bool_(True)) for v in v_vec]
    sol_c = [c(wealth, work_status=np.bool_(True)) for c in c_vec]

    aaae(test_case["expected"], (sol_v, sol_c, sol_work_dec))
