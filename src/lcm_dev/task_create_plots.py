import pytask
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model

from lcm_dev.analytical_solution import _construct_model
from lcm_dev.config import BLD
from lcm_dev.create_plots import plot_consumption_function

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
