import numpy as np
import matplotlib.pyplot as plt
from linear_system_rbf import *

def gaussian(x, u, s):
    N = 1 / (s * np.sqrt(2 * np.pi))
    exp = np.e ** (-(x - u)**2 / (2 * s**2))
    return N * exp


def gaussian_norm(s):
    return 1 / (s * np.sqrt(2 * np.pi))


def unit_gaussian(x, u, s):
    exp = np.e ** (-(x - u)**2 / (2 * s**2))
    return exp

def get_std(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def create_fwhm1_gaussian(x):
    u = 0
    s = get_std(1)

    g = gaussian(x, u, s)

    return g


def get_n3_approximation():
    r = 0.375
    o = get_std(1)
    s = get_std(0.6)
    n = 3

    a = coefficients(r, s, o, n)

    x = np.linspace(-2, 2, 401)
    u = 0
    g_target = gaussian(x, u, o)
    c = gaussian_norm(o)

    g_components = [None for i in range(2*n - 1)]

    g_components[0] = c * a[0] * unit_gaussian(x, u, s)
    for i in range(1, n):
        g_components[i] = c * a[i] * unit_gaussian(x, u + i * r, s)
        g_components[-i] = c * a[i] * unit_gaussian(x, u - i * r, s)

    for g_component in g_components:
        plt.plot(x, g_component, color='k', linestyle=':')
    g_approx = sum(g_components)

    plt.plot(x, g_approx, color='b')

    sep = 5
    plt.scatter(x[::sep], g_target[::sep], color='r', s=5)

    error = g_approx - g_target
    plt.plot(x, error, color='r')

    print(max(error))

    plt.show()


if __name__ == "__main__":
    get_n3_approximation()
