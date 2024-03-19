import numpy as np
from scipy.integrate import tplquad

# Define the function to be integrated
def integrand(l, y, x):
    return x / (x**2 + (y - l)**2) - x / (x**2 + (y + l)**2)

# Define the function that calculates the triple integral
def f(a, b, d, xi, L):
    # Constants
    mu_0 = 1.25e-6
    pi = np.pi
    # Integration limits
    x_lower = d
    x_upper = d + b
    y_lower = lambda x: xi
    y_upper = lambda x: xi + a
    l_lower = lambda x, y: 0
    l_upper = lambda x, y: L
    # Triple integral
    result, error = tplquad(integrand, x_lower, x_upper, y_lower, y_upper, l_lower, l_upper, args=(xi, L))
    return (mu_0 / (4 * pi)) * result