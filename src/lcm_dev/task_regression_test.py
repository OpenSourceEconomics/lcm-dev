"""Save LCM solution and simulation output for regression testing.

This task needs to run again when the example model `PHELPS_DEATON` changes.

"""
import json

import jax.numpy as jnp
import pytask
from lcm.entry_point import get_lcm_function
from lcm.get_model import get_model
from pybaum import tree_map

from lcm_dev.config import BLD


@pytask.mark.produces(BLD.joinpath("regression_tests", "solution_and_simulation.json"))
def task_save_lcm_output(produces):
    """Produce output for lcm regression testing."""
    model_config = get_model("phelps_deaton_regression_test")

    solve_model, _ = get_lcm_function(model=model_config.model, targets="solve")
    simulate_model, _ = get_lcm_function(model=model_config.model, targets="simulate")

    solution = solve_model(model_config.params)

    simulation = simulate_model(
        params=model_config.params,
        vf_arr_list=solution,
        initial_states={
            "wealth": jnp.array([1.0, 20, 40, 70]),
        },
    )

    # Convert output to JSON serializable format
    solution = [sol.tolist() for sol in solution]
    simulation = tree_map(lambda x: x.tolist(), simulation)

    out = {
        "solution": solution,
        "simulation": simulation,
    }

    with produces.open("w") as file:
        file.write(json.dumps(out, indent=4))
