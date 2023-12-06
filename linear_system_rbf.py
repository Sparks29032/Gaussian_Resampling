import numpy as np
import matplotlib.pyplot as plt


def matrix(r, s, o, n):
    def gaussian(r, sd):
        return np.e ** (-r ** 2 / (2 * sd ** 2))

    M = []
    for i in range(n):
        row = [gaussian(i * r, s)]
        for j in range(1, n):
            row.append(gaussian((i - j) * r, s) + gaussian((i + j) * r, s))
        row.append(gaussian(i * r, o))
        M.append(row)
    return M


def system(r, s, o, n):
    def gaussian(r, sd):
        return np.e ** (-r ** 2 / (2 * sd ** 2))

    A = []
    b = []
    for i in range(n):
        row = [gaussian(i * r, s)]
        for j in range(1, n):
            row.append(gaussian((i - j) * r, s) + gaussian((i + j) * r, s))
        b.append(gaussian(i * r, o))
        A.append(row)
    print(A, b)
    return A, b


def coefficients(r, s, o, n):
    def gaussian(r, sd):
        return np.e ** (-r ** 2 / (2 * sd ** 2))

    A = []
    b = []
    for i in range(n):
        row = [gaussian(i * r, s)]
        for j in range(1, n):
            row.append(gaussian((i - j) * r, s) + gaussian((i + j) * r, s))
        b.append(gaussian(i * r, o))
        A.append(row)
    return np.linalg.solve(A, b)

def general_approximation(u, fwhm_target, fwhm_approximator, n, r, r_min, r_max, dr):
    def gaussian(x, u, s):
        N = 1 / (s * np.sqrt(2 * np.pi))
        exp = np.e ** (-(x - u) ** 2 / (2 * s ** 2))
        return N * exp
    def gaussian_norm(s):
        return 1 / (s * np.sqrt(2 * np.pi))
    def unit_gaussian(x, u, s):
        exp = np.e ** (-(x - u) ** 2 / (2 * s ** 2))
        return exp
    def get_std(fwhm):
        return fwhm / (2 * np.sqrt(2 * np.log(2)))

    o = get_std(fwhm_target)
    s = get_std(fwhm_approximator)

    a = coefficients(r, s, o, n)

    r_n = int(((r_max - r_min) / dr) + 1)
    x = np.linspace(r_min, r_max, r_n)
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

    plt.show()

    return max(error), max(error/g_target)


if __name__=='__main__':
    abs_error, rel_error = general_approximation(0, 2, 1.8, 3, 0.3 * 2, -4, 4, 0.01)
    print(abs_error * 100, rel_error * 100)
