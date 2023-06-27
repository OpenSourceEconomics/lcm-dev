"""Save LCM solution and simulation output for regression testing.

This task needs to run again when the example model `PHELPS_DEATON` changes.

"""
import json

import jax.numpy as jnp
import pytask
from lcm.entry_point import get_lcm_function
from lcm.example_models import PHELPS_DEATON
from pybaum import tree_map

from lcm_dev.config import BLD


@pytask.mark.produces(BLD.joinpath("regression_tests", "solution_and_simulation.json"))
def task_save_lcm_output(produces):
    model = {**PHELPS_DEATON, "n_periods": 5}

    solve_model, _ = get_lcm_function(model=model, targets="solve")
    simulate_model, _ = get_lcm_function(model=model, targets="simulate")

    params = {
        "beta": 1.0,
        "utility": {"delta": 1.0},
        "next_wealth": {
            "interest_rate": 0.05,
            "wage": 1.0,
        },
    }

    solution = solve_model(params)

    simulation = simulate_model(
        params,
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
