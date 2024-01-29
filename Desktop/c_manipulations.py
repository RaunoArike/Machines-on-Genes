"""This file provides functions with a consistent parameter layout,
which can be applied to model competitive or cooperative binding."""
import numpy as np


def linear(x, a=None):
    return x


def power_law(x, a):
    # a: [0; inf) - [0; 2]
    return x**a


def exponential(x, a):
    # a: (-inf; inf) - [-0.5; 3]
    return x*np.exp(a*x)
