# Developer Documentation


## `create_params.py`

The module contains the `create_params` function, which extracts the estimation
parameters from a model specification. These are the parameters that are estimated
outside of `lcm`.


## `distributions.py`

This module contains distributions and params-getter functions. The distributions can
be used whenever a model requires randomness, and the params-getter function tell LCM
which parameter of the distribution (e.g. scale) needs to be estimated.


## `function_evaluator.py`

The module provides the `get_function_evaluator` function. The resulting evaluator can
be used to create a function out of values that are defined on a grid. The resulting
function can be handled as if it is an analytical function. This is useful for defining
a value function that acts as if it is an analytical function, even though we calculated
the exact values only on a grid.


## `grids.py`

The module provides function that generate and work with different kind of grids.
Those are either grid-making functions like `linspace` and `logspace`, or
coordinate-getters, that generalizes the coordinate of a value in a grid using
interpolation. That is, in the grid [0, 1, 2] the value 0.5 has coordinate 0.5.


## `interfaces.py`

The module provides internal container objects. All of these objects inherit from
`NamedTuple`.

### `Model`

Internal representation of the user model.


## `interpolation.py`

The module provides functionality to interpolate a point in a grid. Currently
implemented is `linear_interpolation`. It takes the grid information, i.e. the type,
start, stop and number of grid points; it takes a point between the start and stop of
the grid; and it takes the values a function attains on the grid points. The function
then computes the general coordinate value of the point in the grid, and used
interpolation to find the corresponding value at this coordinate. As an example, if
- `grid_info = ('linspace', (0, 1, 3))`
- `point = np.array([0.25])`
- `values = np.array([1, 2, 3])`

Then the interpolated value is 1.5.


## `model_functions.py`

The module contains the `get_utility_and_feasibility_function`, which uses a `Model`
object and additional information to create the utility and feasibility functions
of a given period.


## `process_model.py`

The module contains the `process_model` function, which takes a model specified by
the user and returns a `lcm.interfaces.Model` object.
