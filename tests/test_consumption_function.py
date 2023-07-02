"""Test the consumptions functions of the analytical solution."""
# ruff: noqa: FBT003
import numpy as np
import pytest
from lcm_dev import analytical_solution
from numpy.testing import assert_array_almost_equal as aaae

test_cases_liquidity = [
    # Liquidity constraint binding
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "constraint_timing": 0,
        },
        "expected": np.array((10, 50)),
    },
    # Liquidity constraint binding in 1 period
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "constraint_timing": 1,
        },
        "expected": np.array((13 / 1.95, 53 / 1.95)),
    },
    # Liquidity constraint binding in 2 periods
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "constraint_timing": 2,
        },
        "expected": np.array(
            (
                (10 + 3 * 2) / (1 + 0.95 + 0.95**2),
                (50 + 3 * 2) / (1 + 0.95 + 0.95**2),
            ),
        ),
    },
    # Liquidity constraint binding in 1 period, positive interest rate
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0.1,
            "beta": 0.95,
            "constraint_timing": 1,
        },
        "expected": np.array(((10 + 3 / 1.1) / (1.95), (50 + 3 / 1.1) / (1.95))),
    },
]

test_cases_retirement = [
    # Retirement this period, end of life today
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 0,
            "retirement_timing": 0,
        },
        "expected": (10, 50),
    },
    # Retirement this period, end of life in 1 period
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 1,
            "retirement_timing": 0,
        },
        "expected": (10 / 1.95, 50 / 1.95),
    },
    # Retirement in this period, end of life in 2 periods
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 2,
            "retirement_timing": 0,
        },
        "expected": (10 / (1.95 + 0.95**2), 50 / (1.95 + 0.95**2)),
    },
    # Retirement in 1 period, end of life in 1 period
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 1,
            "retirement_timing": 1,
        },
        "expected": (13 / 1.95, 53 / 1.95),
    },
    # Retirement in 1 period, end of life in 1 period, positive interest rate
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "wage": 3.0,
            "interest_rate": 0.1,
            "beta": 0.95,
            "tau": 1,
            "retirement_timing": 1,
        },
        "expected": ((10 + 3 / 1.1) / 1.95, (50 + 3 / 1.1) / 1.95),
    },
]

test_cases_retiree_consumption = [
    # Retiree consumption, end of life today
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "tau": 0,
            "beta": 0.95,
        },
        "expected": np.array(
            (10, 50),
        ),
    },
    # Retiree consumption, end of life in 1 period
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "tau": 1,
            "beta": 0.95,
        },
        "expected": np.array(
            (10 / 1.95, 50 / 1.95),
        ),
    },
    # Retiree consumption, end of life in 2 periods
    {
        "inputs": {
            "wealth": np.array(
                (10, 50),
            ),
            "tau": 2,
            "beta": 0.95,
        },
        "expected": np.array(
            (10 / (1 + 0.95 + 0.95**2), 50 / (1 + 0.95 + 0.95**2)),
        ),
    },
]

test_cases_policy_func_vector = [
    # Consumption policies, end of life today
    {
        "inputs": {
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 0,
        },
        "expected": {
            "retired": np.array(
                (10, 50),
            ),
            "worker": [
                np.array(
                    (10, 50),
                ),
            ],
        },
    },
    # Consumption policies, end of life next period
    {
        "inputs": {
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 1,
        },
        "expected": {
            "retired": np.array(
                (10 / 1.95, 50 / 1.95),
            ),
            "worker": [
                np.array(
                    (10, 50),
                ),
                np.array(
                    (13 / 1.95, 53 / 1.95),
                ),
                np.array(
                    (10 / 1.95, 50 / 1.95),
                ),
            ],
        },
    },
    # Consumption policies, end of life in 2 periods
    {
        "inputs": {
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 2,
        },
        "expected": {
            "retired": np.array(
                (10 / (1.95 + 0.95**2), 50 / (1.95 + 0.95**2)),
            ),
            "worker": [
                np.array(
                    (10, 50),
                ),
                np.array(
                    (13 / 1.95, 53 / 1.95),
                ),
                np.array(
                    (
                        (10 + 3 * 2) / (1.95 + 0.95**2),
                        (50 + 3 * 2) / (1.95 + 0.95**2),
                    ),
                ),
                np.array(
                    (13 / (1.95 + 0.95**2), 53 / (1.95 + 0.95**2)),
                ),
                np.array(
                    (10 / (1.95 + 0.95**2), 50 / (1.95 + 0.95**2)),
                ),
            ],
        },
    },
]

test_cases_consumption = [
    {
        "inputs": {
            "work_status": np.bool_(False),
            "policy_dict": {
                "retired": lambda wealth: wealth**2,
                "worker": lambda wealth: wealth**3,
            },
            "wt": [-np.inf, np.inf],
        },
        "expected_func": lambda wealth: wealth**2,
    },
    {
        "inputs": {
            "work_status": np.bool_(True),
            "policy_dict": {
                "retired": lambda wealth: wealth**2,
                "worker": [
                    lambda wealth: wealth**3,
                    lambda wealth: wealth**4,
                ],
            },
            "wt": [-np.inf, 0, np.inf],
        },
        "expected_func": lambda wealth: wealth**3 if wealth < 0 else wealth**4,
    },
]


@pytest.mark.parametrize("test", test_cases_liquidity)
def test_liquidity_constrained_consumption(test):
    """Test consumption when liquidity constrained."""
    aaae(
        analytical_solution.liquidity_constrained_consumption(**test["inputs"]),
        test["expected"],
    )


@pytest.mark.parametrize("test", test_cases_retirement)
def test_retirement_discontinuity_consumption(test):
    """Test consumption at retirement discontinuities."""
    aaae(
        analytical_solution.retirement_discontinuity_consumption(**test["inputs"]),
        test["expected"],
    )


@pytest.mark.parametrize("test", test_cases_retiree_consumption)
def test_retiree_consumption(test):
    """Test consumption for retirees."""
    aaae(analytical_solution.retirees_consumption(**test["inputs"]), test["expected"])


@pytest.mark.parametrize("test", test_cases_policy_func_vector)
def test_policy_func_vector(test):
    """Test whole policy function vector."""
    policy_vec = analytical_solution._generate_policy_function_vector(  # noqa: SLF001
        **test["inputs"],
    )
    solution_ret = list(map(policy_vec["retired"], [10, 50]))
    solution_wrk = [list(map(func, [10, 50])) for func in policy_vec["worker"]]
    aaae(solution_ret, test["expected"]["retired"])
    for sol, exp in zip(solution_wrk, test["expected"]["worker"]):
        aaae(sol, exp)


@pytest.mark.parametrize("test", test_cases_consumption)
def test_consumption(test):
    """Test final consumption function."""
    wealth_level = np.linspace(0, 100, 12)
    expected_solution = [test["expected_func"](wealth) for wealth in wealth_level]
    solution = [
        analytical_solution._consumption(wealth, **test["inputs"])  # noqa: SLF001
        for wealth in wealth_level
    ]
    aaae(solution, expected_solution)
