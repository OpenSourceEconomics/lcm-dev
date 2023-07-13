# Hyperstates


## Motivating examples

### Stochastic survival

Assume a model with
- two dense state variables (wealth and human capital)
- Stochastic survival with probability \alpha_t
- Bequest motives

While making choices, agents do not know whether they will survive or not. With probability \alpha_t, their continuation value will be given by the standard Value Function that depends on wealth and human capital. With probability (1 - \alpha_t) the continuation value is given by a bequest Value Function that only depends on wealth.

Being alive is a hyperstate in this model.


### Singles and couples

Assume a model with
- Singles and couples
- The utility of couples depends on the duration of marriage

Thus the state space for couples contains the duration of marriage, whereas this variables is irrelevant/undefined for singles

Being married is a hyperstate in this model

### Life stages

Assume a model with mandatory retirement at age 65.

Before age 65, consumption and thus utility depend on human capital.

After age 65, human capital is irrelevant and not part of the state space.

Being retired is a hyperstate in this model.


## A heuristic definition

Hyperstates are the values of a hypothetical discrete state variable. For each hyperstate, the state space has a different dimensionality.



## There are workarounds but they are not good

### Extra state dimension and filters

The hyperstate is an additional state variable and all differences between the state spaces in different hyperstates are expressed via filters.

This approach can be inefficient if the model has many dense variables that get rendered sparse by the filters that are only needed in one hyperstate.

We currently suggest this approach for mandatory retirement, but there are better ways.

### Special values of a state variable and filters

The hypestate is marked by a special value of one state variable. Example: health == 0 means dead.

This keeps the state space smaller, but puts a lot of complexity (probably many if conditions) into the stochastic transition of a particular state variable. It is also not clear whether all hyperstates have natural special values. (e.g. stochastic survival in a model without health).

## Stochasticity in and between hyperstates

- Stochasticity within hyperstates is usually integrated out via a vmap
- Stochasticity between hyperstates can not be integrated out via a vmap since the state continuation values in the different hyperstates might have very different dimensioalities.


## Potential dangers

- While hyperstates can be a very efficient way to represent complex state spaces, sometimes filters are better. For speed it will be important to have a hand full of large state space chunks, i.e. not to have too many hyperstates.
- The interface can become very complex, ideally the special case of models with only one hyperstate look the same as our current model specification.







