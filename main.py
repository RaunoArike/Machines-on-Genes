"""Main file used to compute (and save) the super selectivity using the ODE formulation."""
from helper_functions import run_and_save
from c_manipulations import *


def performance_parameters():
    # n0:       number of individual nano_stars; should not be changed
    # crs:      number of simulations executed at the same time
    return {"n0": 10000, "crs": None}


def investigation_parameters():
    nb_arms = 10
    compute_settling_time = False  # computing settling time runs much slower than without

    nr_range = np.logspace(-3, 5, 1000)  # 1000 is important for convergence
    k_sol_range = np.logspace(-10, -7, 7)  # -10; -7
    k_sur_range = np.logspace(-7, -3, 9)  # -7; -3
    ranges = k_sol_range, k_sur_range, nr_range

    k_on_scaling, a_on = power_law, 1  # linear, power_law or exponential; function parameter
    k_off_scaling, a_off = power_law, 1  # linear, power_law or exponential; function parameter
    scaling = k_on_scaling, a_on, k_off_scaling, a_off
    return nb_arms, compute_settling_time, ranges, scaling


if __name__ == '__main__':
    # filename to save to
    run_and_save("k10", performance_parameters(), investigation_parameters())
    pass
