import numpy as np
import matplotlib.pyplot as plt
import scipy


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

def general_approximation(u, fwhm_target, fwhm_approximator, n, r, r_min, r_max, dr, plot=True):
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

    g_approx = sum(g_components)
    error = g_approx - g_target

    if plot:
        for g_component in g_components:
            plt.plot(x, g_component, color='k', linestyle=':')

        plt.plot(x, g_approx, color='b')

        sep = 5
        plt.scatter(x[::sep], g_target[::sep], color='r', s=5)

        plt.plot(x, error, color='r')

        plt.show()

    return max(error), max(error/g_target)


def error_plots():
    s = np.linspace(0.1, 0.9, 9)
    fig = plt.figure(figsize=(9, 9))
    axs = []
    for i in range(9):
        axs.append(fig.add_subplot(3, 3, i + 1))

    k = 0
    for si in s:
        r = np.linspace(0.1, 1, 300)
        errors = []
        for ri in r:
            abs_error, rel_error = general_approximation(0, 1, si, 2, ri, -4, 4, 0.01, plot=False)
            errors.append(rel_error)
        errors = np.array(errors)
        min_idx = r[errors == min(errors)][0]
        axs[k].plot(r, errors)
        mi = axs[k].axvline(min_idx, linestyle='--', color='k', label=f'$0.{int(min_idx * 1000)}$')
        axs[k].set_title(f'$f_p/f_a = 0.{int(si * 10)}$')
        axs[k].set_xlabel('$0.5 R_{min}$')
        axs[k].set_ylabel('Relative Error (%)')
        axs[k].legend(handles=[mi], loc='upper left')
        k += 1
    fig.tight_layout()
    plt.show()


def error_plot():
    s = np.linspace(0.1, 0.95, 86)
    mis = []
    k = 0
    for si in s:
        r = np.linspace(0.1, 1, 300)
        errors = []
        for ri in r:
            abs_error, rel_error = general_approximation(0, 1, si, 2, ri, -4, 4, 0.01, plot=False)
            errors.append(rel_error)
        errors = np.array(errors)
        min_idx = r[errors == min(errors)][0]
        mis.append(min_idx)
        k += 1
    params, cov = scipy.optimize.curve_fit(lambda x, a, loc, scale: scipy.stats.skewnorm.pdf(x=0.5-x, a=a, loc=loc, scale=scale), s, mis)
    plt.scatter(s, mis, color='r', s=5)
    plt.plot()
    plt.xlabel(f'$f_p/f_a$')
    plt.ylabel('Error-Minimizing $0.5R_{min}$')
    plt.show()
