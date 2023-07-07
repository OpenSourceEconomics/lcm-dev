"""Task creating the analytical solution."""

import pickle

import numpy as np
import pytask

from lcm_dev.analytical_solution import compute_value_function, simulate
from lcm_dev.config import BLD

MODELS = {
    "iskhakov_2017": {
        "beta": 0.98,
        "delta": 1.0,
        "wage": float(20),
        "interest_rate": 0.0,
        "n_periods": 5,
    },
    "low_delta": {
        "beta": 0.98,
        "delta": 0.1,
        "wage": float(20),
        "interest_rate": 0.0,
        "n_periods": 3,
    },
}

WEALTH_GRID = np.linspace(1, 100, 10_000)
INITIAL_WEALTH_LEVELS = np.linspace(1, 100, 20)

for model, params in MODELS.items():

    @pytask.mark.task(
        id=model,
        kwargs={
            "produces": {
                "values": BLD / "analytical_solution" / f"{model}_v.pkl",
                "consumption": BLD / "analytical_solution" / f"{model}_c.pkl",
                "work_decision": BLD / "analytical_solution" / f"{model}_work.pkl",
            },
            "params": params,
        },
    )
    def task_create_analytical_solution(produces, params):
        """Store analytical solution in a pickle file."""
        values = compute_value_function(grid=WEALTH_GRID, **params)
        consumption, work_decision = simulate(
            initial_wealth_levels=INITIAL_WEALTH_LEVELS,
            **params,
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
