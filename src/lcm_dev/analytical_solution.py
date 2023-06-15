"""Implementation of analytical solution by Iskhakov et al (2017)."""
from functools import partial

import numpy as np
from scipy.optimize import root_scalar


def _u(c, work_dec, delta):
    """Utility function.

    Args:
        c (float): consumption
        work_dec (float): work indicator (True or False)
        delta (float): disutility of work
    Returns:
        float: utility

    """
    u = np.log(c) - work_dec * delta if c > 0 else -np.inf

    return u


def _generate_policy_function_vector(wage, r, beta, tau):
    """Gererate consumption policy function vector given tau.

    This function returns the functions that are used in the
    piecewise consumption function.

    Args:
        wage (float): income
        r (float): interest rate
        beta (float): discount factor
        tau (int): periods left until end of life

    Returns:
        dict: consumption policy dict

    """
    policy_vec_worker = [lambda m: m]

    # Generate liquidity constraint kink functions
    for i in range(1, tau + 1):
        policy_vec_worker.append(
            lambda m, i=i: (
                m + wage * (np.sum([(1 + r) ** (-j) for j in range(1, i + 1)]))
            )
            / (np.sum([beta**j for j in range(0, i + 1)])),
        )

    # Generate retirement discontinuity functions
    for i in reversed(range(1, tau)):
        policy_vec_worker.append(
            lambda m, i=i, tau=tau: (
                m + wage * (np.sum([(1 + r) ** (-j) for j in range(1, i + 1)]))
            )
            / (np.sum([beta**j for j in range(0, tau + 1)])),
        )
    policy_vec_worker.append(
        lambda m, tau=tau: m / (np.sum([beta**j for j in range(0, tau + 1)])),
    )

    # Generate function for retirees
    policy_retiree = lambda m, tau=tau: m / (  # noqa: E731
        np.sum([beta**j for j in range(0, tau + 1)])
    )

    return {"worker": policy_vec_worker, "retired": policy_retiree}


def _compute_wealth_tresholds(v_prime, wage, r, beta, delta, tau, consumption_policy):
    """Compute wealth treshold for piecewise consumption function.

    Args:
        v_prime (function): continuation value of value function
        wage (float): labor income
        r (float): interest rate
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
        consumption_policy (list): consumption policy vector

    Returns:
        list: list of wealth thresholds

    """
    # Liquidity constraint threshold
    wealth_thresholds = [-np.inf, wage / ((1 + r) * beta)]

    # Retirement threshold
    k = delta * np.sum([beta**j for j in range(0, tau + 1)]) ** (-1)
    ret_threshold = ((wage / (1 + r)) * np.exp(-k)) / (1 - np.exp(-k))

    # Other kinks and discontinuities: Root finding
    for i in range(0, (tau - 1) * 2):
        c_l = consumption_policy[i + 1]
        c_u = consumption_policy[i + 2]

        def root_fct(m, c_l=c_l, c_u=c_u):
            return (
                _u(c=c_l(m), work_dec=True, delta=delta)
                - _u(c=c_u(m), work_dec=True, delta=delta)
                + beta * v_prime((1 + r) * (m - c_l(m)) + wage, work_status=True)
                - beta * v_prime((1 + r) * (m - c_u(m)) + wage, work_status=True)
            )

        sol = root_scalar(
            root_fct,
            method="brentq",
            bracket=[wealth_thresholds[i + 1], ret_threshold],
            xtol=1e-10,
            rtol=1e-10,
            maxiter=1000,
        )
        assert sol.converged
        wealth_thresholds.append(sol.root)

    # Add retirement threshold
    wealth_thresholds.append(ret_threshold)

    # Add upper bound
    wealth_thresholds.append(np.inf)

    return wealth_thresholds


def _evaluate_piecewise_conditions(m, wealth_thresholds):
    """Determine correct sub-function of policy function given wealth m.

    Args:
        m (float): current wealth level
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        list: list of booleans

    """
    cond_list = [
        m >= lb and m < ub
        for lb, ub in zip(wealth_thresholds[:-1], wealth_thresholds[1:])
    ]
    return cond_list


def _work_decision(m, work_status, wealth_thresholds):
    """Determine work decision given current wealth level.

    Args:
        m (float): current wealth level
        work_status (bool): work status from last period
        wealth_thresholds (list): list of wealth thresholds
    Returns:
        bool: work decision

    """
    return m < wealth_thresholds[-2] if work_status is not False else False


def _consumption(m, work_status, policy_dict, wt):
    """Determine consumption given current wealth level.

    Args:
        m (float): current wealth level
        work_status (bool): work status from last period
        policy_dict (dict): dictionary of consumption policy functions
        wt (list): list of wealth thresholds
    Returns:
        float: consumption

    """
    if work_status is False:
        cons = policy_dict["retired"](m)

    else:
        condlist = _evaluate_piecewise_conditions(m, wealth_thresholds=wt)
        cons = np.piecewise(x=m, condlist=condlist, funclist=policy_dict["worker"])
    return cons


def _value_function(
    m,
    work_status,
    work_dec_func,
    c_pol,
    v_prime,
    beta,
    delta,
    tau,
    r,
    wage,
):
    """Determine value function given current wealth level and retirement status.

    Args:
        m (float): current wealth level
        work_status (bool): work decision from last period
        work_dec_func (function): work decision function
        c_pol (function): consumption policy function
        v_prime (function): continuation value of value function
        beta (float): discount factor
        delta (float): disutility of work
        tau (int): periods left until end of life
        r (float): interest rate
        wage (float): labor income
    Returns:
        float: value function

    """
    if m == 0:
        v = -np.inf
    elif work_status is False:
        a = np.log(m) * np.sum([beta**j for j in range(0, tau + 1)])
        b = -np.log(np.sum([beta**j for j in range(0, tau + 1)]))
        c = np.sum([beta**j for j in range(0, tau + 1)])
        d = beta * (np.log(beta) + np.log(1 + r))
        e = np.sum(
            [
                beta**j * np.sum([beta**i for i in range(0, tau - j)])
                for j in range(0, tau)
            ],
        )
        v = a + b * c + d * e
    else:
        work_dec = work_dec_func(m=m, work_status=work_status)
        cons = c_pol(m=m, work_status=work_status)

        inst_util = _u(c=cons, work_dec=work_dec, delta=delta)
        cont_val = v_prime((1 + r) * (m - cons) + wage * work_dec, work_status=work_dec)

        v = inst_util + beta * cont_val

    return v


def _construct_model(delta, num_periods, param_dict):
    """Construct model given parameters via backward inducton.

    Args:
        delta (float): disutility of work
        num_periods (int): length of life
        param_dict (dict): dictionary of parameters
    Returns:
        list: list of value functions

    """
    c_pol = [None] * num_periods
    v = [None] * num_periods
    work_dec_func = [None] * num_periods

    for t in reversed(range(0, num_periods)):
        if t == num_periods - 1:
            v[t] = (
                lambda m, work_status: np.log(m) if m > 0 else -np.inf  # noqa: ARG005
            )
            c_pol[t] = lambda m, work_status: m  # noqa: ARG005
            work_dec_func[t] = lambda m, work_status: False  # noqa: ARG005
        else:
            # Time left until retirement
            param_dict["tau"] = num_periods - t - 1

            # Generate consumption function
            policy_dict = _generate_policy_function_vector(**param_dict)

            wt = _compute_wealth_tresholds(
                v_prime=v[t + 1],
                consumption_policy=policy_dict["worker"],
                delta=delta,
                **param_dict,
            )

            c_pol[t] = partial(_consumption, policy_dict=policy_dict, wt=wt)

            # Determine retirement status
            work_dec_func[t] = partial(
                _work_decision,
                wealth_thresholds=wt,
            )

            # Calculate V
            v[t] = partial(
                _value_function,
                work_dec_func=work_dec_func[t],
                c_pol=c_pol[t],
                v_prime=v[t + 1],
                delta=delta,
                **param_dict,
            )
    return v, c_pol, work_dec_func


def simulate_c(c_fct, work_dec_func, wealth_levels, num_periods, wage, r):
    """Simulate consumption for different initial wealth levels.

    Args:
        c_fct (list): consumption functions
        work_dec_func (list): work decision functions
        wealth_levels (list): initial wealth levels
        num_periods (int): length of life
        wage (float): labor income
        r (float): interest rate
    Returns:
        numpy array: consumption levels

    """
    c_vec = np.zeros((num_periods, len(wealth_levels)))
    work_dec_vec = np.zeros((num_periods, len(wealth_levels)))
    m_vec = np.zeros((num_periods, len(wealth_levels)))
    m_vec[0, :] = wealth_levels

    for t in range(num_periods):
        for i, m in enumerate(m_vec[t, :]):
            work_dec_vec[t, i] = work_dec_func[t](m=m, work_status=True)
            if t == 0:
                c_vec[t, i] = c_fct[t](m=m, work_status=True)
                m_vec[t + 1, i] = (1 + r) * (m - c_vec[t, i]) + wage * work_dec_vec[
                    t,
                    i,
                ]
            elif t > 0 and t < num_periods - 1:
                c_vec[t, i] = c_fct[t](m=m, work_status=work_dec_vec[t - 1, i])
                m_vec[t + 1, i] = (1 + r) * (m - c_vec[t, i]) + wage * work_dec_vec[
                    t,
                    i,
                ]
            else:
                c_vec[t, i] = c_fct[t](m=m, work_status=work_dec_vec[t - 1, i])

    return c_vec


def analytical_solution(
    wealth_grid,
    simulation_grid,
    beta,
    wage,
    r,
    delta,
    num_periods,
):
    """Compute value function analytically on a grid.

    Args:
        wealth_grid (list): grid of wealth levels
        simulation_grid (list): grid of wealth levels for simulation
        beta (float): discount factor
        wage (float): labor income
        r (float): interest rate
        delta (float): disutility of work
        num_periods (int): length of life
    Returns:
        list: values of value function

    """
    # Unpack parameters

    param_dict = {
        "beta": beta,
        "wage": wage,
        "r": r,
        "tau": None,
    }

    v_fct, c_pol, work_dec = _construct_model(
        delta=delta,
        num_periods=num_periods,
        param_dict=param_dict,
    )

    v = {
        k: [
            list(map(v_fct[t], wealth_grid, [v] * len(wealth_grid)))
            for t in range(0, num_periods)
        ]
        for (k, v) in [["worker", True], ["retired", False]]
    }

    c_vec = simulate_c(
        c_fct=c_pol,
        work_dec_func=work_dec,
        wealth_levels=simulation_grid,
        num_periods=num_periods,
        r=r,
        wage=wage,
    )

    return v, c_vec
