import jax.numpy as jnp
import numpy as np
import pytask
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model

from lcm_dev.analytical_solution import (
    _construct_model,
    compute_value_function,
    simulate,
)
from lcm_dev.config import BLD
from lcm_dev.create_plots import (
    plot_consumption_function,
    plot_consumption_profiles,
    plot_value_functions,
)

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


@pytask.mark.produces(
    {
        "worker": BLD.joinpath(
            "plots",
            "value_functions_worker.html",
        ),
        "retired": BLD.joinpath("plots", "value_functions_retired.html"),
    },
)
def task_create_value_function_plots(produces):
    """Create plots of value functions over the life-cycle."""
    # LCM solution
    model = get_model("iskhakov_2017_five_periods")
    solve_model, _ = get_lcm_function(model=model.model, targets="solve")
    vf_arr_list = solve_model(model.params)
    numerical_solution = np.stack(vf_arr_list)
    numerical_solution = {
        "worker": numerical_solution[:, 0, :],
        "retired": numerical_solution[:, 1, :],
    }

    # Analytical solution
    grid_start = model.model["states"]["wealth"]["start"]
    grid_end = model.model["states"]["wealth"]["stop"]
    grid_size = model.model["states"]["wealth"]["n_points"]
    grid = np.linspace(grid_start, grid_end, grid_size)

    analytical_solution = compute_value_function(
        grid=grid,
        beta=model.params["beta"],
        delta=model.params["utility"]["delta"],
        interest_rate=model.params["next_wealth"]["interest_rate"],
        wage=model.params["next_wealth"]["wage"],
        n_periods=model.model["n_periods"],
    )

    # Create plots
    fig_worker, fig_retired = plot_value_functions(
        analytical_solution=analytical_solution,
        numerical_solution=numerical_solution,
        grid=grid,
        periods=model.model["n_periods"],
    )
    fig_worker.write_html(produces["worker"])
    fig_retired.write_html(produces["retired"])
