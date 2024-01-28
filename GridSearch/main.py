from scipy.optimize import root
from model import *
from sklearn.model_selection import ParameterGrid
from c_manipulations import *
from multiprocessing import Pool
import os
import time


def variable_grid():
    common_grid = {
        'k_on_sol': np.logspace(-10, -7, 7), # 16
        'k_on_sur': np.logspace(-7, -3, 9), # 16
        'num_arms': np.linspace(1, 30, 30, dtype=int) # 30
    }

    power_law_specific = {
        'a_on': np.linspace(0, 2, 11), # 21
        'a_off': np.linspace(0, 2, 11), # 21
        'func': [power_law]
    }

    exponential_specific = {
        'a_on': np.linspace(-0.5, 2.5, 16), # 36
        'a_off': np.linspace(-0.5, 2.5, 16),
        'func': [exponential]
    }

    combined_grid = [
        {**common_grid, **power_law_specific},
        {**common_grid, **exponential_specific}
    ]

    combined_grid = ParameterGrid(combined_grid)

    return combined_grid


def run_model(nr_range, nb_arms, c, n0):
    result = np.zeros((len(nr_range), nb_arms+2))
    x = np.array([n0]+nb_arms*[0]+[n0*nr_range[0]])

    for i, nr in enumerate(nr_range):
        x = root(steady_state, x, args=(nb_arms, n0, nr, c)).x
        result[i] = x.copy()/n0
        result[i, -1] /= nr
    return result


def alpha_comp(thetas, spacing):
    return np.log(thetas[1:] / thetas[:-1]) / spacing


def max_alpha_comp(alphas):
    return np.nanmax(alphas), np.nanargmax(alphas)


def single_run(params, nr_range=np.logspace(-3, 5, 1000), n0=10000, k_off=1):
    func = params["func"]
    k_sur = params["k_on_sur"]
    k_sol = params["k_on_sol"]
    num_arms = params["num_arms"]
    a_on = params["a_on"]
    a_off = params["a_off"]

    c = np.array([k_sol] + (2*num_arms.item()-2)*[0] + [k_off])
    for m in range(1, num_arms):
        c[m] = k_sur * func(num_arms-m, a_on)  # c(m)(m+1)
        c[-1-m] = 1 * func(m+1, a_off)  # c(m+1)(m)

    results = run_model(nr_range, num_arms, c, n0)

    thetas = 1 - results[..., 0]
    alphas = alpha_comp(thetas, np.log(nr_range[1]/nr_range[0]))
    max_alpha, max_idx = max_alpha_comp(alphas)
    max_theta = thetas[max_idx]

    fn = 0 if func == exponential else 1

    return np.array([k_sol, k_sur, num_arms, fn, a_on, a_off, max_alpha, max_theta])


if __name__ == '__main__':
    start = time.time()
    
    param_grid = variable_grid()

    print(len(param_grid))
    print(os.cpu_count())

    with Pool() as pool:
        results = pool.map(single_run, param_grid)

    header = np.array([["K_on_sol", "K_on_sur", "Number of arms", "Scaling function", "A_on", "A_off", "Max alpha", "Max theta"]])
    with open('data.csv', 'w') as f:
        np.savetxt(f, header, fmt='%s', delimiter=',')
        np.savetxt(f, results, delimiter=',', fmt='%.18e')

    end = time.time()
    print(end - start)
    