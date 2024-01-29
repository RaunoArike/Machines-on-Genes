"""All kinds of functions used to produce the results of main"""
import time
import pickle
from multiprocessing import Pool
from scipy.optimize import root
from scipy.integrate import solve_ivp
from model import *


def all_variable_combinations(iv, k_off=1):
    nb_arms, _, (k_sol_range, k_sur_range, nr_range), (k_on_sc, a_on, k_off_sc, a_off) = iv
    all_variables = []
    for i, k_sol in enumerate(k_sol_range):
        for j, k_sur in enumerate(k_sur_range):
            c = np.array([k_sol] + (2*nb_arms-2)*[0] + [k_off])
            for m in range(1, nb_arms):
                c[m] = k_sur * k_on_sc(nb_arms-m, a_on)  # c(m)(m+1)
                c[-1-m] = k_off * k_off_sc(m+1, a_off)  # c(m+1)(m)
                pass
            all_variables.append((c, (i, j)))
            pass
        pass

    return all_variables


def settling_time(nb_arms, n0, nr, c, x):
    base = 1.1
    y = np.array([n0] + nb_arms * [0] + [n0 * nr])
    for j in range(round(np.log1p(1e10 * (base - 1)) / np.log(base))):
        if np.allclose(y, x, rtol=1e-2, atol=1e-1):
            # condition for testing if ODE close to steady state, adapt tolerances if needed
            return (base ** j - 1) / (base - 1)
        y = solve_ivp(time_evolution, (0, base ** j), y, args=(nb_arms, c), method='Radau').y[:, -1]  # stiff problem
        pass
    return -1


def run_model(nr_range, nb_arms, n0, cst, c):
    result = np.zeros((len(nr_range), nb_arms+2))
    settling_times = np.zeros(len(nr_range))
    x = np.array([n0]+nb_arms*[0]+[n0*nr_range[0]])

    for i, nr in enumerate(nr_range):
        x = root(steady_state, x, args=(nb_arms, n0, nr, c[0])).x
        result[i] = x.copy()/n0
        result[i, -1] /= nr
        if cst:
            settling_times[i] = settling_time(nb_arms, n0, nr, c[0], x)
        pass
    return c[1], result, settling_times


def run_and_save(name, pf, iv):
    nb_arms, cst, (_, _, nr_range), _ = iv
    vr_list = [(nr_range, nb_arms, pf["n0"], cst, v) for v in all_variable_combinations(iv)]
    print("Simulations Start")
    t0 = time.time()
    with Pool(pf["crs"]) as p:
        results = p.starmap(run_model, vr_list)
        pass
    print(f"main loop run time: {time.time() - t0}")

    with open(name + '.pickle', 'wb') as f:
        pickle.dump((results, iv[:-1]), f)
        pass
    pass
