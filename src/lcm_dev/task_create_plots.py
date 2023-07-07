import pytask

# temporary until lcm branch is merged
from lcm.entry_point import get_lcm_function
from pybaum import tree_update

from lcm_dev.analytical_solution import _construct_model
from lcm_dev.config import BLD
from lcm_dev.create_plots import plot_consumption_function
from lcm_dev.models import (
    PHELPS_DEATON_NO_BORROWING,
)

UPDATE_CONFIG = {
    "states": {
        "wealth": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": 1_000,
        },
    },
    "n_periods": 5,
    "choices": {
        "consumption": {
            "grid_type": "linspace",
            "start": 1,
            "stop": 400,
            "n_points": 1_000,
        },
    },
}

PARAMS = {
    "beta": 0.98,
    "next_wealth": {
        "interest_rate": 0.0,
        "wage": 20.0,
    },
    "utility": {
        "delta": 1.0,
    },
}

PERIOD = 0


@pytask.mark.produces(BLD.joinpath("plots", "consumption_function.html"))
def task_create_consumption_function_plots(produces):
    """Create plots of consumption function."""
    # LCM solution
    model = tree_update(PHELPS_DEATON_NO_BORROWING, UPDATE_CONFIG)
    solve_model, param_template = get_lcm_function(model=model, targets="solve")
    model_params = tree_update(param_template, PARAMS)

    model_solution = solve_model(model_params)

    # Analytical consumption function
    _, analytical_consumption_fct, _ = _construct_model(
        n_periods=UPDATE_CONFIG["n_periods"],
        delta=PARAMS["utility"]["delta"],
        beta=PARAMS["beta"],
        interest_rate=PARAMS["next_wealth"]["interest_rate"],
        wage=PARAMS["next_wealth"]["wage"],
    )

    # Create plots
    fig = plot_consumption_function(
        model=model,
        analytical_consumption_fct=analytical_consumption_fct,
        lcm_solution=model_solution,
        model_params=model_params,
        period=PERIOD,
    )
    fig.write_html(produces)
