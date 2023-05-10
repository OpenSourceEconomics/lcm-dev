# Developer Documentation


## `create_params.py`

The module contains the `create_params` function, which extracts the esitmation
parameters from a model specification. These are the paramaters that are estimated
outside of `lcm`.


## `distributions.py`

This module contains distributions and params-getter functions. The distributions can
be used whenever a model requires randomness, and the params-getter function tell LCM
which parameter of the distribution (e.g. scale) needs to be estimated.


## `grids.py`

The module provides function that generate and work with different kind of grids.
Those are either grid-making functions like `linspace` and `logspace`, or
coordinate-getters, that generalizes the coordinate of a value in a grid using
interpolation. That is, in the grid [0, 1, 2] the value 0.5 has coordinate 0.5.


## `interfaces.py`

The module provides internal container objects. All of these objects inherit from
`NamedTuple`.

### `Model`

Internal represenation of the user model.


## `model_functions.py`

The module contains the `get_utility_and_feasibility_function`, which uses a `Model`
object and additional information to create the utility and feasibility functions
of a given period.


## `process_model.py`

The module contains the `process_model` function, which takes a model specified by
the user and returns a `lcm.interfaces.Model` object.
