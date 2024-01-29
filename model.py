"""This file contains the different models describing the nanostar system."""
import numpy as np
from gillespy2 import Model, Species, Parameter, Reaction, TimeSpan


def steady_state(x, nb_arms, n0, nr, c):
    """Computes the steady state of the ODE formulation using root-finding."""
    k = c * np.hstack((x[-1] * x[:-2], np.flip(x[1:-1])))

    dx_dt = np.zeros(nb_arms)
    dx_dt[0] = k[-1] - k[0]
    for i in range(1, nb_arms):
        dx_dt[i] = k[i - 1] - k[-i] + k[-i - 1] - k[i]
        pass

    constraint1 = x[:-1].sum() - n0
    constraint2 = x[-1] + np.dot(np.arange(nb_arms+1), x[:-1]) - n0*nr
    return np.append(dx_dt, (constraint1, constraint2))


def time_evolution(t, x, nb_arms, c):
    """Computes the time evolution of the ODE system"""
    k = c * np.hstack((x[-1] * x[:-2], np.flip(x[1:-1])))

    dx_dt = np.zeros(nb_arms+2)
    dx_dt[0] = k[-1] - k[0]
    for i in range(1, nb_arms):
        dx_dt[i] = k[i-1] - k[-i] + k[-i-1] - k[i]
        pass
    dx_dt[-2] = k[nb_arms-1] - k[nb_arms]
    dx_dt[-1] = k[nb_arms:].sum() - k[:nb_arms].sum()
    return dx_dt


def k_nanostar(k, n0, nr, t, c):
    """Models the nanostar system in the Gillespy2 package."""
    model = Model(name="Nanostar")

    # Species
    model.add_species(Species(name="n0", initial_value=n0))
    for i in range(1, k+1):
        model.add_species(Species(f"n{i}"))
        pass
    model.add_species(Species(name="rec", initial_value=round(n0*nr)))

    # Parameters
    for i in range(k):
        model.add_parameter([Parameter(f"c{i}{i+1}", c[i]), Parameter(f"c{i+1}{i}", c[-1-i])])
        pass

    # Reactions
    for i in range(k):
        model.add_reaction(Reaction(reactants={f"n{i}": 1, "rec": 1}, products={f"n{i+1}": 1}, rate=f"c{i}{i+1}"))
        model.add_reaction(Reaction(reactants={f"n{i+1}": 1}, products={f"n{i}": 1, "rec": 1}, rate=f"c{i+1}{i}"))
        pass

    # Timespan
    model.timespan(TimeSpan.linspace(t=t, num_points=1001))

    return model
