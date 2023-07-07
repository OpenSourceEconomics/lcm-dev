"""Compare results of analytical solution with a hard-coded two period model."""
# ruff: noqa: FBT003
import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    compute_value_function,
    simulate,
)
from numpy.testing import assert_array_almost_equal as aaae


def utility(consumption, work_dec, delta):
    """Utility function.

    Args:
        consumption (float): consumption
        work_dec (bool): work indicator
        delta (float): disutility of work
    Returns:
        float: utility

    """
    return np.log(consumption) - work_dec * delta if consumption > 0 else -np.inf


def consumption_second_to_last(wealth, wage, interest_rate, beta, delta):
    """Consumption in the second to last period.

    Args:
        wealth (float): wealth
        wage (float): wage
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
    Returns:
        float: consumption

    """
    k = delta / (1 + beta)
    m_bar = (wage / (1 + interest_rate) * np.exp(-k)) / (1 - np.exp(-k))
    if wealth <= wage / ((1 + interest_rate) * beta):
        out = wealth
    elif wealth > wage / ((1 + interest_rate) * beta) and wealth <= m_bar:
        out = (wealth + wage / (1 + interest_rate)) / (1 + beta)
    else:
        out = wealth / (1 + beta)
    return out


def work_dec_second_to_last(wealth, wage, interest_rate, beta, delta):
    """Work decision in the second to last period.

    Args:
        wealth (float): wealth
        wage (float): wage
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
    Returns:
        bool: work decision

    """
    k = delta / (1 + beta)
    m_bar = ((wage / (1 + interest_rate)) * np.exp(-k)) / (1 - np.exp(-k))
    return wealth <= m_bar


def value_function_last(wealth, delta):
    """Value function in the last period.

    Args:
        wealth (float): wealth
        wage (float): wage
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
    Returns:
        float

    """
    consumption = wealth
    work_dec = False
    return utility(consumption=consumption, work_dec=work_dec, delta=delta)


def value_function_second_to_last(wealth, worker, wage, interest_rate, beta, delta):
    """Value function in the second to last period.

    Args:
        wealth (float): wealth
        worker (bool): indicator for worker
        wage (float): wage
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
    Returns:
        float

    """
    if worker:
        consumption = consumption_second_to_last(
            wealth=wealth,
            wage=wage,
            interest_rate=interest_rate,
            beta=beta,
            delta=delta,
        )
        work_dec = work_dec_second_to_last(
            wealth=wealth,
            wage=wage,
            interest_rate=interest_rate,
            beta=beta,
            delta=delta,
        )
    else:
        consumption = wealth / (1 + beta)
        work_dec = False

    u = utility(consumption=consumption, work_dec=work_dec, delta=delta)
    v = value_function_last(
        wealth=(wealth - consumption) * (1 + interest_rate) + wage * work_dec,
        delta=delta,
    )
    return u + beta * v


def create_solution(wealth_grid, simulation_grid, params):
    """Create solution of 2-period model.

    Args:
        wealth_grid (np.array): grid of wealth
        simulation_grid (np.array): grid of simulation
        params (dict): parameters
    Returns:
        np.array: value function
        np.array: consumption
        np.array: work decision

    """
    v_worker = np.zeros((2, len(wealth_grid)))
    v_retiree = np.zeros((2, len(wealth_grid)))
    cons = np.zeros((2, len(simulation_grid)))
    work_dec = np.zeros((2, len(simulation_grid)))

    for i, wealth in enumerate(wealth_grid):
        v_worker[0, i] = value_function_second_to_last(
            wealth=wealth,
            worker=True,
            **params,
        )
        v_retiree[0, i] = value_function_second_to_last(
            wealth=wealth,
            worker=False,
            **params,
        )
        v_worker[1, i] = value_function_last(
            wealth=wealth,
            delta=params["delta"],
        )
        v_retiree[1, i] = np.log(wealth)

    for i, wealth in enumerate(simulation_grid):
        cons[0, i] = consumption_second_to_last(
            wealth=wealth,
            **params,
        )
        work_dec[0, i] = work_dec_second_to_last(
            wealth=wealth,
            **params,
        )
        wealth_next_period = (wealth - cons[0, i]) * (
            1 + params["interest_rate"]
        ) + params["wage"] * work_dec[0, i]

        cons[1, i] = wealth_next_period
        work_dec[1, i] = False

    v = {
        "worker": v_worker,
        "retired": v_retiree,
    }

    return v, cons, work_dec


@pytest.fixture()
def params():
    """Parameters for the model."""
    return {
        "wage": 20.0,
        "interest_rate": 0.1,
        "beta": 0.98,
        "delta": 1.0,
    }


def test_analytical_solution(params):
    """Test analytical solution against simple two-period model."""
    grid = np.linspace(1, 100, 12)

    values = compute_value_function(
        grid=grid,
        n_periods=2,
        **params,
    )

    consumption, work_decision = simulate(
        n_periods=2,
        initial_wealth_levels=grid,
        **params,
    )

    v_exp, cons_exp, work_dec_exp = create_solution(
        wealth_grid=grid,
        simulation_grid=grid,
        params=params,
    )
    aaae(values["worker"], v_exp["worker"])
    aaae(values["retired"], v_exp["retired"])
    aaae(consumption, cons_exp)
    aaae(work_decision, work_dec_exp)
