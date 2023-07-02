"""Test the work decision function."""
# ruff: noqa: FBT003
import numpy as np
import pytest
from lcm_dev.analytical_solution import (
    _work_decision,
)

test_cases_work_dec = [
    ((10, np.bool_(False), [0, 10, 20, 30]), np.bool_(False)),
    ((40, np.bool_(False), [0, 10, 20, 30]), np.bool_(False)),
    ((40, np.bool_(True), [0, 10, 20, 30]), np.bool_(False)),
    ((10, np.bool_(True), [0, 10, 20, 30]), np.bool_(True)),
    ((19, np.bool_(True), [0, 10, 20, 30]), np.bool_(True)),
]


@pytest.mark.parametrize(("inputs", "expected"), test_cases_work_dec)
def test_evaluate_piecewise_conditions(inputs, expected):
    """Test the piecewise conditions function."""
    assert _work_decision(*inputs) == expected
