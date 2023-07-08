import jax.numpy as jnp
import numpy as np
import pytask
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model

from lcm_dev.analytical_solution import _construct_model, simulate
from lcm_dev.config import BLD
from lcm_dev.create_plots import plot_consumption_function, plot_consumption_profiles

PERIOD = 0


@pytask.mark.produces(
    BLD.joinpath("plots", f"consumption_function_period_{PERIOD}.html"),
)
def task_create_consumption_function_plots(produces):
    """Create plots of consumption function."""
    # LCM solution
    model = get_model("iskhakov_2017_five_periods")
    solve_model, _ = get_lcm_function(model=model.model, targets="solve")
    model_solution = solve_model(model.params)
    # Analytical consumption function
    _, analytical_consumption_fct, _ = _construct_model(
        n_periods=model.model["n_periods"],
        delta=model.params["utility"]["delta"],
        beta=model.params["beta"],
        interest_rate=model.params["next_wealth"]["interest_rate"],
        wage=model.params["next_wealth"]["wage"],
    )

    # Create plots
    fig = plot_consumption_function(
        model=model.model,
        analytical_consumption_fct=analytical_consumption_fct,
        lcm_solution=model_solution,
        model_params=model.params,
        period=PERIOD,
    )
    fig.write_html(produces)


@pytask.mark.produces(BLD.joinpath("plots", "consumption_profiles.html"))
def task_create_consumption_profile_plots(produces):
    grid = np.linspace(1, 200, 100)
    """Create plots of consumption profiles over the life-cycle."""

    # LCM solution
    model = get_model("iskhakov_2017_five_periods")
    solve_model, _ = get_lcm_function(model=model.model, targets="solve")
    model_solution = solve_model(model.params)
    model_solution = solve_model(model.params)
    simulate_model, _ = get_lcm_function(model=model.model, targets="simulate")
    simulation_results = simulate_model(
        model.params,
        vf_arr_list=model_solution,
        initial_states={
            "wealth": jnp.array(grid),
            "lagged_retirement": jnp.repeat(0, repeats=len(grid)),
        },
    )
    numerical_consumption = [
        result.get("choices", {}).get("consumption", [])
        for result in simulation_results
    ]

    # Simulate analytical consumption
    analytical_consumption, _ = simulate(
        beta=model.params["beta"],
        delta=model.params["utility"]["delta"],
        interest_rate=model.params["next_wealth"]["interest_rate"],
        wage=model.params["next_wealth"]["wage"],
        n_periods=model.model["n_periods"],
        initial_wealth_levels=grid,
    )

    # Create plots
    fig = plot_consumption_profiles(
        analytical_consumption=analytical_consumption,
        numerical_consumption=numerical_consumption,
        grid=grid,
    )
    fig.write_html(produces)
