"""Temporary module until PHELPS_DEATON_NO_BORROWING and.

simulation are on the same lcm branch.

"""

import jax.numpy as jnp


def phelps_deaton_utility_with_filter(
    consumption,
    working,
    delta,
    lagged_retirement,  # noqa: ARG001
):
    return jnp.log(consumption) - delta * working


def next_wealth(wealth, consumption, working, wage, interest_rate):
    return (1 + interest_rate) * (wealth - consumption) + wage * working


def consumption_constraint(consumption, wealth):
    return wealth >= consumption


def absorbing_retirement_filter(retirement, lagged_retirement):
    return jnp.logical_or(retirement == 1, lagged_retirement == 0)


def working(retirement):
    return 1 - retirement


PHELPS_DEATON_NO_BORROWING = {
    "functions": {
        "utility": phelps_deaton_utility_with_filter,
        "next_wealth": next_wealth,
        "consumption_constraint": consumption_constraint,
        "working": working,
        "absorbing_retirement_filter": absorbing_retirement_filter,
        "next_lagged_retirement": lambda retirement: retirement,
    },
    "choices": {
        "retirement": {"options": [0, 1]},
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 100,
            "n_points": 11,
        },
    },
    "states": {
        "wealth": {"grid_type": "linspace", "start": 0, "stop": 100, "n_points": 11},
        "lagged_retirement": {"options": [0, 1]},
    },
    "n_periods": 20,
}
