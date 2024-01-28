import numpy as np


def steady_state(x, nb_arms, n0, nr, c):
    k = c * np.hstack((x[-1] * x[:-2], np.flip(x[1:-1])))

    dx_dt = np.zeros(nb_arms)
    dx_dt[0] = k[-1] - k[0]

    for i in range(1, nb_arms):
        dx_dt[i] = k[i-1] - k[-i] + k[-i-1] - k[i]
        pass

    constraint1 = x[:-1].sum() - n0
    constraint2 = x[-1] + np.dot(np.arange(nb_arms+1), x[:-1]) - n0*nr
    return np.append(dx_dt, (constraint1, constraint2))
