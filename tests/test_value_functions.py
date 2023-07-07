"""Test the value functions of the analytical solution."""
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

TEST_CASES_VALUES_RETIREES = [
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


TEST_CASES_VALUES_WORKERS = [
    # Value function without continuation value
    {
        "kwargs": {
            "wealth": 10.0,
            "beta": 1.0,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: 0.0,  # noqa: ARG005
        },
        "expected": np.log(10.0) - 0.1,
    },
    # Value function with continuation value
    {
        "kwargs": {
            "wealth": 10.0,
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
        },
        "expected": np.log(10.0) - 0.1 + 0.95 * np.log(1.0),
    },
    # Value function with continuation value and nontrivial consumption
    {
        "kwargs": {
            "wealth": 10.0,
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth / 2,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
        },
        "expected": np.log(10.0 / 2) - 0.1 + 0.95 * np.log(1.0 + 10.0 / 2),
    },
]

TEST_CASES_VALUE_FUNC = [
    # Value function retirees
    {
        "kwargs": {
            "wealth": 10.0,
            "work_status": False,
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
        "kwargs": {
            "wealth": 10.0,
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth / 2,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
            "tau": None,
        },
        "expected": np.log(10.0 / 2) - 0.1 + 0.95 * np.log(1.0 + 10.0 / 2),
    },
]

TEST_CASES_VALUE_FUNC_LAST_PERIOD = [
    (10, False, np.log(10)),
    (10, True, np.log(10)),
    (-1, False, -np.inf),
    (-1, True, -np.inf),
]

TEST_CASES_CONSTRUCT_MODEL = [
    {
        "wealth": 10,
        "kwargs": {
            "delta": -1,
            "n_periods": 1,
            "beta": -1,
            "wage": -1,
            "interest_rate": -1,
        },
        "expected": ([np.log(10)], [10], [False]),
    },
    {
        "wealth": 10.0,
        "kwargs": {
            "delta": 0.1,
            "n_periods": 2,
            "beta": 0.95,
            "wage": 1.0,
            "interest_rate": 0.0,
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
                True,
                False,
            ],
        ),
    },
]


@pytest.mark.parametrize(
    ("wealth", "beta", "tau", "interest_rate", "expected"),
    TEST_CASES_VALUES_RETIREES,
)
def test_value_func_retirees(wealth, beta, tau, interest_rate, expected):
    solution = value_function_retirees(
        wealth=wealth,
        beta=beta,
        tau=tau,
        interest_rate=interest_rate,
    )
    aaae(expected, solution)


@pytest.mark.parametrize("test_case", TEST_CASES_VALUES_WORKERS)
def test_value_func_workers(test_case):
    solution = value_function_workers(**test_case["kwargs"])
    aaae(test_case["expected"], solution)


@pytest.mark.parametrize("test_case", TEST_CASES_VALUE_FUNC)
def test_value_func(test_case):
    solution = _value_function(
        **test_case["kwargs"],
    )
    aaae(test_case["expected"], solution)


@pytest.mark.parametrize(
    ("wealth", "work_status", "expected"),
    TEST_CASES_VALUE_FUNC_LAST_PERIOD,
)
def test_value_func_last_period(wealth, work_status, expected):
    solution = value_function_last_period(
        wealth=wealth,
        work_status=work_status,
    )
    aae(expected, solution)


@pytest.mark.parametrize("test_case", TEST_CASES_CONSTRUCT_MODEL)
def test_construct_model(test_case):
    """Test fully constructed model.

    First period value function, consumption policy and work decision function.

    """
    value_functions, consumption_function, work_decision_functions = _construct_model(
        **test_case["kwargs"],
    )
    wealth = test_case["wealth"]

    sol_work_dec = [
        work_dec(wealth, work_status=True) for work_dec in work_decision_functions
    ]
    sol_v = [v(wealth, work_status=True) for v in value_functions]
    sol_c = [c(wealth, work_status=True) for c in consumption_function]

    aaae(test_case["expected"], (sol_v, sol_c, sol_work_dec))
