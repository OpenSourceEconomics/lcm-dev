"""Implementation of analytical solution by Iskhakov et al (2017)."""
from functools import partial

import numpy as np
from scipy.optimize import root_scalar


def utility(consumption, work_decision, delta):
    """Utility function.

    Args:
        consumption (float): consumption
        work_decision (float): work indicator (True or False)
        delta (float): disutility of work
    Returns:
        float: utility

    """
    return np.log(consumption) - work_decision * delta if consumption > 0 else -np.inf


def liquidity_constrained_consumption(
    wealth,
    wage,
    interest_rate,
    beta,
    constraint_timing,
):
    """Compute consumption function in liquidity constrained range.

    Args:
        wealth (float): wealth
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        constraint_timing (int): periods left until liquidity constraint is binding
    Returns:
        float: consumption

    """
    return (
        wealth
        + wage
        * np.sum([(1 + interest_rate) ** (-j) for j in range(1, constraint_timing + 1)])
    ) / (np.sum([beta**j for j in range(0, constraint_timing + 1)]))


def retirement_discontinuity_consumption(
    wealth,
    wage,
    interest_rate,
    beta,
    tau,
    retirement_timing,
):
    """Compute consumption function in retirement discontinuity range.

    Args:
        wealth (float): wealth
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        tau (int): periods left until end of life
        retirement_timing (int): periods left until retirement
    Returns:
        float: consumption

    """
    return (
        wealth
        + wage
        * np.sum([(1 + interest_rate) ** (-j) for j in range(1, retirement_timing + 1)])
    ) / (np.sum([beta**j for j in range(0, tau + 1)]))


def retirees_consumption(wealth, tau, beta):
    """Compute consumption function for retirees.

    Args:
        wealth (float): wealth
        tau (int): periods left until end of life
        beta (float): discount factor
    Returns:
        float: consumption

    """
    return wealth / (np.sum([beta**j for j in range(0, tau + 1)]))


def _generate_policy_function_vector(wage, interest_rate, beta, tau):
    """Gererate consumption policy function vector given tau.

    This function returns the functions that are used in the
    piecewise consumption function.

    Args:
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        tau (int): periods left until end of life

    Returns:
        dict: consumption policy dict

    """
    policy_vec_worker = [lambda wealth: wealth]

    # Generate liquidity constraint kink functions
    for constraint_timing in range(1, tau + 1):
        policy_vec_worker.append(
            partial(
                liquidity_constrained_consumption,
                wage=wage,
                interest_rate=interest_rate,
                beta=beta,
                constraint_timing=constraint_timing,
            ),
        )

    # Generate retirement discontinuity functions
    for retirement_timing in reversed(range(0, tau)):
        policy_vec_worker.append(
            partial(
                retirement_discontinuity_consumption,
                wage=wage,
                interest_rate=interest_rate,
                beta=beta,
                tau=tau,
                retirement_timing=retirement_timing,
            ),
        )

    # Generate function for retirees
    policy_retiree = partial(
        retirees_consumption,
        tau=tau,
        beta=beta,
    )

    return {"worker": policy_vec_worker, "retired": policy_retiree}


def retirement_threshold(wage, interest_rate, beta, delta, tau):
    """Compute retirement threshold.

    Args:
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
    Returns:
        float: retirement threshold

    """
    k = delta * np.sum([beta**j for j in range(0, tau + 1)]) ** (-1)
    ret_threshold = ((wage / (1 + interest_rate)) * np.exp(-k)) / (1 - np.exp(-k))

    return ret_threshold


def root_function(
    wealth,
    consumption_lb,
    consumption_ub,
    v_prime,
    wage,
    interest_rate,
    beta,
    delta,
):
    """Root function for root finding algorithm.

    Used to compute position of kinks and notches in the
    consumption function. At the kinks and notches, the
    agents are indifferent between the consumption levels
    implied by the two adjacent parts of the consumption function.

    Args:
        wealth (float): wealth
        consumption_lb (float): consumption function at lower bound
        consumption_ub (float): consumption function at upper bound
        v_prime (function): continuation value of value function
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
    Returns:
        float: root function value

    """
    return (
        utility(
            consumption=consumption_lb(wealth),
            work_decision=True,
            delta=delta,
        )
        - utility(
            consumption=consumption_ub(wealth),
            work_decision=True,
            delta=delta,
        )
        + beta
        * v_prime(
            wealth=(1 + interest_rate) * (wealth - consumption_lb(wealth)) + wage,
            work_status=True,
        )
        - beta
        * v_prime(
            wealth=(1 + interest_rate) * (wealth - consumption_ub(wealth)) + wage,
            work_status=True,
        )
    )


def wealth_thresholds_kinks_discs(
    v_prime,
    wage,
    interest_rate,
    beta,
    delta,
    bracket,
    consumption_policy,
    ret_threshold,
    wealth_thresholds,
):
    """Compute wealth treshold for piecewise consumption function.

    Args:
        v_prime (function): continuation value of value function
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
        bracket (int): current bracket to be calculated
        consumption_policy (list): consumption policy vector
        ret_threshold (float): retirement threshold
        wealth_thresholds (list): wealth thresholds up to `bracket`

    Returns:
        list: list of wealth thresholds

    """
    consumption_lb = consumption_policy[bracket + 1]
    consumption_ub = consumption_policy[bracket + 2]

    root_fct = partial(
        root_function,
        consumption_lb=consumption_lb,
        consumption_ub=consumption_ub,
        v_prime=v_prime,
        wage=wage,
        interest_rate=interest_rate,
        beta=beta,
        delta=delta,
    )

    sol = root_scalar(
        root_fct,
        method="brentq",
        bracket=[wealth_thresholds[bracket + 1], ret_threshold],
        xtol=1e-10,
        rtol=1e-10,
        maxiter=1000,
    )
    assert sol.converged
    return sol.root


def _compute_wealth_tresholds(
    v_prime,
    wage,
    interest_rate,
    beta,
    delta,
    tau,
    consumption_policy,
):
    """Compute wealth treshold for piecewise consumption function.

    Args:
        v_prime (function): continuation value of value function
        wage (float): labor income
        interest_rate (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
        consumption_policy (list): consumption policy vector

    Returns:
        list: list of wealth thresholds

    """
    # Liquidity constraint threshold
    wealth_thresholds = [-np.inf, wage / ((1 + interest_rate) * beta)]

    # Retirement threshold
    ret_threshold = retirement_threshold(
        wage=wage,
        interest_rate=interest_rate,
        beta=beta,
        delta=delta,
        tau=tau,
    )

    # Other kinks and discontinuities: Root finding
    for bracket in range(0, (tau - 1) * 2):
        wealth_thresholds.append(
            wealth_thresholds_kinks_discs(
                v_prime=v_prime,
                wage=wage,
                interest_rate=interest_rate,
                beta=beta,
                delta=delta,
                bracket=bracket,
                consumption_policy=consumption_policy,
                ret_threshold=ret_threshold,
                wealth_thresholds=wealth_thresholds,
            ),
        )

    # Add retirement threshold
    wealth_thresholds.append(ret_threshold)

    # Add upper bound
    wealth_thresholds.append(np.inf)

    return wealth_thresholds


def _evaluate_piecewise_conditions(wealth, wealth_thresholds):
    """Determine correct sub-function of policy function given wealth wealth.

    Args:
        wealth (float): current wealth level
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        list: list of booleans

    """
    cond_list = [
        wealth >= lb and wealth < ub
        for lb, ub in zip(wealth_thresholds[:-1], wealth_thresholds[1:])
    ]
    return cond_list


def _work_decision(wealth, work_status, wealth_thresholds):
    """Determine work decision given current wealth level.

    Args:
        wealth (float): current wealth level
        work_status (np.bool_): work status from last period
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        np.bool_: work decision

    """
    ret_threshold = wealth_thresholds[-2]
    return np.bool_(wealth < ret_threshold) if work_status else False


def _consumption(wealth, work_status, policy_dict, wt):
    """Determine consumption given current wealth level.

    Args:
        wealth (float): current wealth level
        work_status (np.bool_): work status from last period
        policy_dict (dict): dictionary of consumption policy functions
        wt (list): list of wealth thresholds
    Returns:
        float: consumption

    """
    if work_status:
        condlist = _evaluate_piecewise_conditions(wealth, wealth_thresholds=wt)
        cons = np.piecewise(x=wealth, condlist=condlist, funclist=policy_dict["worker"])

    else:
        cons = policy_dict["retired"](wealth)

    return cons


def value_function_retirees(wealth, beta, tau, interest_rate):
    """Determine value function for retirees.

    Args:
        wealth (float): current wealth level
        beta (float): discount factor
        tau (int): periods left until end of life
        interest_rate (float): interest rate
    Returns:
        float: value function

    """
    a = np.log(wealth) * np.sum([beta**j for j in range(0, tau + 1)])
    b = -np.log(np.sum([beta**j for j in range(0, tau + 1)]))
    c = np.sum([beta**j for j in range(0, tau + 1)])
    d = beta * (np.log(beta) + np.log(1 + interest_rate))
    e = np.sum(
        [
            beta**j * np.sum([beta**i for i in range(0, tau - j)])
            for j in range(0, tau)
        ],
    )
    return a + b * c + d * e


def value_function_workers(
    wealth,
    beta,
    delta,
    interest_rate,
    wage,
    work_status,
    work_dec_func,
    c_pol,
    v_prime,
):
    """Determine value function for workers.

    Args:
        wealth (float): current wealth level
        beta (float): discount factor
        delta (float): disutility of work
        interest_rate (float): interest rate
        wage (float): labor income
        work_status (np.bool_): work status from last period
        work_dec_func (function): work decision function
        c_pol (function): consumption policy function
        v_prime (function): continuation value of value function
    Returns:
        float: value function

    """
    work_decision = work_dec_func(
        wealth=wealth,
        work_status=work_status,
    )
    cons = c_pol(
        wealth=wealth,
        work_status=work_status,
    )

    inst_util = utility(
        consumption=cons,
        work_decision=work_decision,
        delta=delta,
    )
    cont_val = v_prime(
        wealth=(1 + interest_rate) * (wealth - cons) + wage * work_decision,
        work_status=work_decision,
    )

    return inst_util + beta * cont_val


def _value_function(
    wealth,
    work_status,
    work_dec_func,
    c_pol,
    v_prime,
    beta,
    delta,
    tau,
    interest_rate,
    wage,
):
    """Determine value function given current wealth level and retirement status.

    Args:
        wealth (float): current wealth level
        work_status (np.bool_): work decision from last period
        work_dec_func (function): work decision function
        c_pol (function): consumption policy function
        v_prime (function): continuation value of value function
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
        interest_rate (float): interest rate
        wage (float): labor income
    Returns:
        float: value function

    """
    if wealth == 0:
        value = -np.inf
    elif work_status:
        value = value_function_workers(
            wealth=wealth,
            beta=beta,
            delta=delta,
            interest_rate=interest_rate,
            wage=wage,
            work_status=work_status,
            work_dec_func=work_dec_func,
            c_pol=c_pol,
            v_prime=v_prime,
        )
    else:
        value = value_function_retirees(
            wealth=wealth,
            beta=beta,
            tau=tau,
            interest_rate=interest_rate,
        )

    return value


def value_function_last_period(wealth, work_status):  # noqa: ARG001
    """Determine value function in last period.

    Args:
        wealth (float): current wealth level
        work_status (np.bool_): work status from last period
    Returns:
        float: value function

    """
    return np.log(wealth) if wealth > 0 else -np.inf


def consumption_last_period(wealth, work_status):  # noqa: ARG001
    """Determine consumption in last period.

    Args:
        wealth (float): current wealth level
        work_status (np.bool_): work status from last period
    Returns:
        float: consumption

    """
    return wealth


def work_dec_last_period(wealth, work_status):  # noqa: ARG001
    """Determine work decision in last period.

    Args:
        wealth (float): current wealth level
        work_status (np.bool_): work status from last period
    Returns:
        np.bool_: work decision

    """
    return False


def _construct_model(delta, n_periods, beta, wage, interest_rate):
    """Construct model given parameters via backward inducton.

    Args:
        delta (float): disutility of work
        n_periods (int): length of life
        beta (float): discount factor
        wage (float): labor income
        interest_rate (float): interest rate
    Returns:
        list: list of value functions

    """
    param_dict = {
        "beta": float(beta),
        "wage": float(wage),
        "interest_rate": float(interest_rate),
    }
    c_pol = [None] * n_periods
    value_func = [None] * n_periods
    work_dec_func = [None] * n_periods

    for t in reversed(range(0, n_periods)):
        if t == n_periods - 1:
            value_func[t] = value_function_last_period
            c_pol[t] = consumption_last_period
            work_dec_func[t] = work_dec_last_period
        else:
            # Time left until retirement
            param_dict["tau"] = n_periods - t - 1

            # Generate consumption function
            policy_dict = _generate_policy_function_vector(**param_dict)

            wt = _compute_wealth_tresholds(
                v_prime=value_func[t + 1],
                consumption_policy=policy_dict["worker"],
                delta=delta,
                **param_dict,
            )

            c_pol[t] = partial(_consumption, policy_dict=policy_dict, wt=wt)

            # Generate work decision function
            work_dec_func[t] = partial(
                _work_decision,
                wealth_thresholds=wt,
            )

            # Calculate Value Function
            value_func[t] = partial(
                _value_function,
                work_dec_func=work_dec_func[t],
                c_pol=c_pol[t],
                v_prime=value_func[t + 1],
                delta=delta,
                **param_dict,
            )

    return value_func, c_pol, work_dec_func


def simulate_cons_work_response(
    period,
    wealth_levels,
    wage,
    interest_rate,
    work_decision_function,
    consumption_function,
    work_status_last_period,
):
    """Simulate consumption and work response to a change in wealth.

    Args:
        period (int): current period
        wealth_levels (list): initial wealth levels
        wage (float): labor income
        interest_rate (float): interest rate
        work_decision_function (function): work decision function
        consumption_function (function): consumption function
        work_status_last_period (np.bool_): work status from last period
    Returns:
        numpy array: consumption levels
        numpy array: work decision
        numpy array: wealth levels next period

    """
    work_decision = np.zeros(len(wealth_levels))
    consumption = np.zeros(len(wealth_levels))
    wealth_next_period = np.zeros(len(wealth_levels))

    if period == 0:
        work_decision_function = partial(
            work_decision_function,
            work_status=True,
        )
        consumption_function = partial(consumption_function, work_status=True)
        work_decision = list(map(work_decision_function, wealth_levels))
        consumption = list(map(consumption_function, wealth_levels))
        wealth_next_period = (1 + interest_rate) * (
            wealth_levels - consumption
        ) + np.multiply(wage, work_decision)

    else:
        for grid_id, wealth in enumerate(wealth_levels):
            work_decision[grid_id] = work_decision_function(
                wealth=wealth,
                work_status=work_status_last_period[grid_id],
            )
            consumption[grid_id] = consumption_function(
                wealth=wealth,
                work_status=work_decision[grid_id],
            )
            wealth_next_period[grid_id] = (1 + interest_rate) * (
                wealth - consumption[grid_id]
            ) + wage * work_decision[grid_id]

    return consumption, work_decision, wealth_next_period


def simulate(
    beta,
    delta,
    initial_wealth_levels,
    n_periods,
    wage,
    interest_rate,
):
    """Simulate consumption and retirement decision for different initial wealth levels.

    Args:
        beta (float): discount factor
        delta (float): disutility of work
        initial_wealth_levels (list): initial wealth levels
        n_periods (int): length of life
        wage (float): labor income
        interest_rate (float): interest rate
    Returns:
        numpy array: simulated consumption levels
        numpy array: simulated work decision

    """
    _, consumption_function, work_decision_function = _construct_model(
        beta=beta,
        wage=wage,
        interest_rate=interest_rate,
        delta=delta,
        n_periods=n_periods,
    )

    grid_size = len(initial_wealth_levels)
    c_mat = np.zeros((n_periods, grid_size))
    work_dec_mat = np.zeros((n_periods, grid_size), dtype=np.bool_)
    wealth_mat = np.zeros((n_periods + 1, grid_size))
    wealth_mat[0, :] = initial_wealth_levels  # initial wealth levels

    for period in range(n_periods):
        (
            c_mat[period, :],
            work_dec_mat[period, :],
            wealth_mat[period + 1, :],
        ) = simulate_cons_work_response(
            period=period,
            wealth_levels=wealth_mat[period, :],
            wage=wage,
            interest_rate=interest_rate,
            work_decision_function=work_decision_function[period],
            consumption_function=consumption_function[period],
            work_status_last_period=work_dec_mat[period - 1, :],
        )

    return c_mat, work_dec_mat


def compute_value_function(grid, beta, wage, interest_rate, delta, n_periods):
    """Compute value function analytically on a grid.

    Args:
        grid (list): grid of wealth levels
        beta (float): discount factor
        wage (float): labor income
        interest_rate (float): interest rate
        delta (float): disutility of work
        n_periods (int): length of life
    Returns:
        list: values of value function

    """
    value_function, _, _ = _construct_model(
        beta=beta,
        wage=wage,
        interest_rate=interest_rate,
        delta=delta,
        n_periods=n_periods,
    )

    values = {
        worker_type: [
            list(map(value_function[t], grid, np.repeat(work_status, len(grid))))
            for t in range(0, n_periods)
        ]
        for (worker_type, work_status) in [
            ["worker", True],
            ["retired", False],
        ]
    }

    return values
