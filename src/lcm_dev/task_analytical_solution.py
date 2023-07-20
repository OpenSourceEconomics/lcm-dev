"""Task creating the analytical solution."""

import pickle

import numpy as np
import pytask
from lcm.get_model import get_model

from lcm_dev.analytical_solution import compute_value_function, simulate
from lcm_dev.config import BLD

MODELS = {
    "iskhakov_2017_five_periods": get_model("iskhakov_2017_five_periods"),
    "iskhakov_2017_low_delta": get_model("iskhakov_2017_low_delta"),
}

INITIAL_WEALTH_LEVELS = np.linspace(1, 100, 20)

for model_name, model in MODELS.items():

    @pytask.mark.task(
        id=model_name,
        kwargs={
            "produces": {
                "values": BLD / "analytical_solution" / f"{model_name}_v.pkl",
                "consumption": BLD / "analytical_solution" / f"{model_name}_c.pkl",
                "work_decision": BLD / "analytical_solution" / f"{model_name}_work.pkl",
            },
            "model": model,
        },
    )
    def task_create_analytical_solution(produces, model):
        """Store analytical solution in a pickle file."""
        wealth_grid_start = model.model["states"]["wealth"]["start"]
        wealth_grid_stop = model.model["states"]["wealth"]["stop"]
        wealth_grid_size = model.model["states"]["wealth"]["n_points"]
        wealth_grid = np.linspace(wealth_grid_start, wealth_grid_stop, wealth_grid_size)

        values = compute_value_function(
            grid=wealth_grid,
            beta=model.params["beta"],
            delta=model.params["utility"]["delta"],
            interest_rate=model.params["next_wealth"]["interest_rate"],
            wage=model.params["next_wealth"]["wage"],
            n_periods=model.model["n_periods"],
        )
        consumption, work_decision = simulate(
            initial_wealth_levels=INITIAL_WEALTH_LEVELS,
            beta=model.params["beta"],
            delta=model.params["utility"]["delta"],
            interest_rate=model.params["next_wealth"]["interest_rate"],
            wage=model.params["next_wealth"]["wage"],
            n_periods=model.model["n_periods"],
        )
        pickle.dump(
            values,
            produces["values"].open("wb"),
        )
        pickle.dump(
            consumption,
            produces["consumption"].open("wb"),
        )
        pickle.dump(
            work_decision,
            produces["work_decision"].open("wb"),
        )
