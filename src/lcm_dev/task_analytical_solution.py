"""Task creating the analytical solution."""

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
                "values_worker": BLD.joinpath(
                    "analytical_solution",
                    f"{model_name}__values_worker.csv",
                ),
                "values_retired": BLD.joinpath(
                    "analytical_solution",
                    f"{model_name}__values_retired.csv",
                ),
                "consumption": BLD.joinpath(
                    "analytical_solution",
                    f"{model_name}__consumption.csv",
                ),
                "work_decision": BLD.joinpath(
                    "analytical_solution",
                    f"{model_name}__work_decision.csv",
                ),
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

        # save consumption
        np.savetxt(produces["consumption"], consumption, delimiter=",")

        # save work decision
        np.savetxt(produces["work_decision"], work_decision, delimiter=",")

        # save value function
        for _type in ["worker", "retired"]:
            np.savetxt(produces[f"values_{_type}"], values[_type], delimiter=",")
