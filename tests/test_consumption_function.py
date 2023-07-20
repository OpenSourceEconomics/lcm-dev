"""Test the consumptions functions of the analytical solution."""
import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    _consumption,
    _generate_policy_function_vector,
    liquidity_constrained_consumption,
    retirees_consumption,
    retirement_discontinuity_consumption,
)
from numpy.testing import assert_array_almost_equal as aaae

TEST_CASES_LIQUIDITY = [
    # Liquidity constraint binding
    (10.0, 3.0, 0.0, 0.95, 0, 10.0),
    (50.0, 3.0, 0.0, 0.95, 0, 50.0),
    # Liquidity constraint binding in 1 period
    (10.0, 3.0, 0.0, 0.95, 1, 13 / 1.95),
    (50.0, 3.0, 0.0, 0.95, 1, 53 / 1.95),
    # Liquidity constraint binding in 2 periods
    (10.0, 3.0, 0.0, 0.95, 2, (10 + 3 * 2) / (1 + 0.95 + 0.95**2)),
    (50.0, 3.0, 0.0, 0.95, 2, (50 + 3 * 2) / (1 + 0.95 + 0.95**2)),
    # Liquidity constraint binding in 1 period, positive interest rate
    (10.0, 3.0, 0.1, 0.95, 1, (10 + 3 / 1.1) / (1.95)),
    (50.0, 3.0, 0.1, 0.95, 1, (50 + 3 / 1.1) / (1.95)),
]

TEST_CASES_RETIREMENT = [
    # Retirement this period, end of life today
    (10.0, 3.0, 0.0, 0.95, 0, 0, 10.0),
    (50.0, 3.0, 0.0, 0.95, 0, 0, 50.0),
    # Retirement this period, end of life in 1 period
    (10.0, 3.0, 0.0, 0.95, 1, 0, 10 / 1.95),
    (50.0, 3.0, 0.0, 0.95, 1, 0, 50 / 1.95),
    # Retirement in this period, end of life in 2 periods
    (10.0, 3.0, 0.0, 0.95, 2, 0, 10 / (1.95 + 0.95**2)),
    (50.0, 3.0, 0.0, 0.95, 2, 0, 50 / (1.95 + 0.95**2)),
    # Retirement in 1 period, end of life in 1 period
    (10.0, 3.0, 0.0, 0.95, 1, 1, 13 / 1.95),
    (50.0, 3.0, 0.0, 0.95, 1, 1, 53 / 1.95),
    # Retirement in 1 period, end of life in 1 period, positive interest rate
    (10.0, 3.0, 0.1, 0.95, 1, 1, (10 + 3 / 1.1) / 1.95),
    (50.0, 3.0, 0.1, 0.95, 1, 1, (50 + 3 / 1.1) / 1.95),
]

TEST_CASES_RETIREE_CONSUMPTION = [
    # Retiree consumption, end of life today
    (10.0, 0, 0.95, 10.0),
    (50.0, 0, 0.95, 50.0),
    # Retiree consumption, end of life in 1 period
    (10.0, 1, 0.95, 10 / 1.95),
    (50.0, 1, 0.95, 50 / 1.95),
    # Retiree consumption, end of life in 2 periods
    (10.0, 2, 0.95, 10 / (1 + 0.95 + 0.95**2)),
    (50.0, 2, 0.95, 50 / (1 + 0.95 + 0.95**2)),
]

TEST_CASES_POLICY_FUNC_VECTOR = [
    # Consumption policies, end of life today
    {
        "kwargs": {
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 0,
        },
        "expected": {
            "retired": np.array(
                [10, 50],
            ),
            "worker": [
                np.array(
                    [10, 50],
                ),
            ],
        },
    },
    # Consumption policies, end of life next period
    {
        "kwargs": {
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 1,
        },
        "expected": {
            "retired": np.array(
                [10 / 1.95, 50 / 1.95],
            ),
            "worker": [
                np.array(
                    [10, 50],
                ),
                np.array(
                    [13 / 1.95, 53 / 1.95],
                ),
                np.array(
                    [10 / 1.95, 50 / 1.95],
                ),
            ],
        },
    },
    # Consumption policies, end of life in 2 periods
    {
        "kwargs": {
            "wage": 3.0,
            "interest_rate": 0,
            "beta": 0.95,
            "tau": 2,
        },
        "expected": {
            "retired": np.array(
                [10 / (1.95 + 0.95**2), 50 / (1.95 + 0.95**2)],
            ),
            "worker": [
                np.array(
                    [10, 50],
                ),
                np.array(
                    [13 / 1.95, 53 / 1.95],
                ),
                np.array(
                    [
                        (10 + 3 * 2) / (1.95 + 0.95**2),
                        (50 + 3 * 2) / (1.95 + 0.95**2),
                    ],
                ),
                np.array(
                    [13 / (1.95 + 0.95**2), 53 / (1.95 + 0.95**2)],
                ),
                np.array(
                    [10 / (1.95 + 0.95**2), 50 / (1.95 + 0.95**2)],
                ),
            ],
        },
    },
]

TEST_CASES_CONSUMPTION = [
    {
        "kwargs": {
            "work_status": False,
            "policy_dict": {
                "retired": lambda wealth: wealth**2,
                "worker": lambda wealth: wealth**3,
            },
            "wt": [-np.inf, np.inf],
        },
        "expected_func": lambda wealth: wealth**2,
    },
    {
        "kwargs": {
            "work_status": True,
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


@pytest.mark.parametrize(
    ("wealth", "wage", "interest_rate", "beta", "constraint_timing", "expected"),
    TEST_CASES_LIQUIDITY,
)
def test_liquidity_constrained_consumption(
    wealth,
    wage,
    interest_rate,
    beta,
    constraint_timing,
    expected,
):
    """Test consumption when liquidity constrained."""
    sol = liquidity_constrained_consumption(
        wealth=wealth,
        wage=wage,
        interest_rate=interest_rate,
        beta=beta,
        constraint_timing=constraint_timing,
    )
    aaae(
        sol,
        expected,
    )


@pytest.mark.parametrize(
    ("wealth", "wage", "interest_rate", "beta", "tau", "retirement_timing", "expected"),
    TEST_CASES_RETIREMENT,
)
def test_retirement_discontinuity_consumption(
    wealth,
    wage,
    interest_rate,
    beta,
    tau,
    retirement_timing,
    expected,
):
    """Test consumption at retirement discontinuities."""
    sol = retirement_discontinuity_consumption(
        wealth=wealth,
        wage=wage,
        interest_rate=interest_rate,
        beta=beta,
        tau=tau,
        retirement_timing=retirement_timing,
    )
    aaae(sol, expected)


@pytest.mark.parametrize(
    ("wealth", "tau", "beta", "expected"),
    TEST_CASES_RETIREE_CONSUMPTION,
)
def test_retiree_consumption(wealth, tau, beta, expected):
    """Test consumption for retirees."""
    sol = retirees_consumption(
        wealth=wealth,
        tau=tau,
        beta=beta,
    )
    aaae(sol, expected)


@pytest.mark.parametrize("test", TEST_CASES_POLICY_FUNC_VECTOR)
def test_policy_func_vector(test):
    """Test whole policy function vector."""
    policy_vec = _generate_policy_function_vector(
        **test["kwargs"],
    )
    solution_ret = list(map(policy_vec["retired"], [10, 50]))
    solution_wrk = [list(map(func, [10, 50])) for func in policy_vec["worker"]]
    aaae(solution_ret, test["expected"]["retired"])
    for sol, exp in zip(solution_wrk, test["expected"]["worker"]):
        aaae(sol, exp)


@pytest.mark.parametrize("test", TEST_CASES_CONSUMPTION)
def test_consumption(test):
    """Test final consumption function."""
    wealth_level = np.linspace(0, 100, 12)
    expected_solution = [test["expected_func"](wealth) for wealth in wealth_level]
    solution = [_consumption(wealth, **test["kwargs"]) for wealth in wealth_level]
    aaae(solution, expected_solution)
