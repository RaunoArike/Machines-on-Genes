import numpy as np

def linear(x, a=None):
    return x

def power_law(x, a):
    return x**a

def exponential(x, a):
    return x*np.exp(a*x)