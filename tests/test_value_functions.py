"""Test the value functions of the analytical solution."""

import numpy as np
import pytest
from lcm_dev import analytical_solution
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_almost_equal as aae

test_cases_value_func_retirees = [
    {
        "inputs": {
            "beta": 1.0,
            "tau": 0,
            "interest_rate": 0.0,
        },
        "expected_func": lambda wealth: np.log(wealth),
    },
    {
        "inputs": {
            "beta": 0.95,
            "tau": 1,
            "interest_rate": 0.0,
        },
        "expected_func": lambda wealth: np.log(wealth / 1.95)
        + 0.95 * np.log(wealth - (wealth / 1.95)),
    },
    {
        "inputs": {
            "beta": 0.95,
            "tau": 2,
            "interest_rate": 0.1,
        },
        "expected_func": lambda wealth: np.log(wealth) * (1 + 0.95 + 0.95**2)
        - np.log(1 + 0.95 + 0.95**2) * (1 + 0.95 + 0.95**2)
        + 0.95 * (np.log(0.95) + np.log(1.1)) * (1 + 0.95 + 0.95),
    },
]

test_cases_value_func_workers = [
    # Value function without continuation value
    {
        "inputs": {
            "beta": 1.0,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: 0.0,  # noqa: ARG005
        },
        "expected_func": lambda wealth: np.log(wealth) - 0.1,
    },
    # Value function with continuation value
    {
        "inputs": {
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
        },
        "expected_func": lambda wealth: np.log(wealth) - 0.1 + 0.95 * np.log(1.0),
    },
    # Value function with continuation value and nontrivial consumption
    {
        "inputs": {
            "beta": 0.95,
            "delta": 0.1,
            "interest_rate": 0.0,
            "wage": 1.0,
            "work_status": True,
            "work_dec_func": lambda wealth, work_status: True,  # noqa: ARG005
            "c_pol": lambda wealth, work_status: wealth / 2,  # noqa: ARG005
            "v_prime": lambda wealth, work_status: np.log(wealth),  # noqa: ARG005
        },
        "expected_func": lambda wealth: np.log(wealth / 2)
        - 0.1
        + 0.95 * np.log(1.0 + wealth / 2),
    },
]

test_cases_value_func = [
    # Value function retirees
    {
        "inputs": {
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
        "expected_func": lambda wealth: np.log(wealth) * (1 + 0.95 + 0.95**2)
        - np.log(1 + 0.95 + 0.95**2) * (1 + 0.95 + 0.95**2)
        + 0.95 * (np.log(0.95) + np.log(1.1)) * (1 + 0.95 + 0.95),
    },
    # Value function workers
    {
        "inputs": {
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
        "expected_func": lambda wealth: np.log(wealth / 2)
        - 0.1
        + 0.95 * np.log(1.0 + wealth / 2),
    },
]

test_cases_value_func_last_period = [
    ((10, False), np.log(10)),
    ((10, True), np.log(10)),
    ((-1, False), -np.inf),
    ((-1, True), -np.inf),
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
        "expected": ([np.log(10)], [10], [False]),
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
                True,
                False,
            ],
        ),
    },
]


@pytest.mark.parametrize("test_case", test_cases_value_func_retirees)
def test_value_func_retirees(test_case):
    wealth_levels = np.linspace(1, 100, 12)
    expected = [test_case["expected_func"](wealth) for wealth in wealth_levels]
    solution = [
        analytical_solution.value_function_retirees(wealth, **test_case["inputs"])
        for wealth in wealth_levels
    ]
    aaae(expected, solution)


@pytest.mark.parametrize("test_case", test_cases_value_func_workers)
def test_value_func_workers(test_case):
    wealth_levels = np.linspace(1, 100, 12)
    expected = [test_case["expected_func"](wealth) for wealth in wealth_levels]
    solution = [
        analytical_solution.value_function_workers(wealth, **test_case["inputs"])
        for wealth in wealth_levels
    ]
    aaae(expected, solution)


@pytest.mark.parametrize("test_case", test_cases_value_func)
def test_value_func(test_case):
    wealth_levels = np.linspace(1, 100, 12)
    expected = [test_case["expected_func"](wealth) for wealth in wealth_levels]
    solution = [
        analytical_solution._value_function(  # noqa: SLF001
            wealth,
            **test_case["inputs"],
        )
        for wealth in wealth_levels
    ]
    aaae(expected, solution)


@pytest.mark.parametrize(("inputs", "expected"), test_cases_value_func_last_period)
def test_value_func_last_period(inputs, expected):
    solution = analytical_solution.value_function_last_period(*inputs)
    aae(expected, solution)


@pytest.mark.parametrize("test_case", test_cases_construct_model)
def test_construct_model(test_case):
    """Test fully constructed model.

    First period value function, consumption policy and work decision function.

    """
    v_vec, c_vec, work_dec_vec = analytical_solution._construct_model(  # noqa: SLF001
        **test_case["inputs"],
    )
    wealth = test_case["wealth"]

    sol_work_dec = [work_dec(wealth, work_status=True) for work_dec in work_dec_vec]
    sol_v = [v(wealth, work_status=True) for v in v_vec]
    sol_c = [c(wealth, work_status=True) for c in c_vec]

    aaae(test_case["expected"], (sol_v, sol_c, sol_work_dec))