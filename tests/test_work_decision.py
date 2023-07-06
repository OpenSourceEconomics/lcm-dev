"""Test the work decision function."""
# ruff: noqa: FBT003
import pytest
from lcm_dev.analytical_solution import (
    _work_decision,
)

test_cases_work_dec = [
    ((10, False, [0, 10, 20, 30]), False),
    ((40, False, [0, 10, 20, 30]), False),
    ((40, True, [0, 10, 20, 30]), False),
    ((10, True, [0, 10, 20, 30]), True),
    ((19, True, [0, 10, 20, 30]), True),
]


@pytest.mark.parametrize(("inputs", "expected"), test_cases_work_dec)
def test_evaluate_piecewise_conditions(inputs, expected):
    """Test the piecewise conditions function."""
    assert _work_decision(*inputs) == expected
