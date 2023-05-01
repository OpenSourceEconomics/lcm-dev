"""Task creating the analytical solution."""

import pickle

import numpy as np
import pytask

from lcm_dev.analytical_solution import analytical_solution
from lcm_dev.config import BLD

models = {
    "iskhakov_2017": {
        "beta": 0.98,
        "delta": 1.0,
        "wage": float(20),
        "r": 0.0,
        "num_periods": 5,
    },
    "low_delta": {
        "beta": 0.98,
        "delta": 0.1,
        "wage": float(20),
        "r": 0.0,
        "num_periods": 3,
    },
    "high_wage": {
        "beta": 0.98,
        "delta": 1.0,
        "wage": float(100),
        "r": 0.0,
        "num_periods": 5,
    },
}

wealth_grid = np.linspace(1, 100, 10_000)

for model, params in models.items():

    @pytask.mark.task(
        id=model,
        kwargs={
            "produces": BLD / "analytical_solution" / f"{model}.p",
            "params": params,
        },
    )
    def task_create_analytical_solution(produces, params):
        """Store analytical solution in a pickle file."""
        pickle.dump(
            analytical_solution(grid=wealth_grid, **params),
            produces.open("wb"),
        )
