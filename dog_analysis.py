from dog_interpolation import *
from analytic_dog import *
import time
import re
import cProfile


def basic_comparison():
    def get_std(fwhm):
        return fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = get_std(1)
    s = get_std(0.6)
    r = 0.375
    print(greedy_r(sigma, s))
    n = 3

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    X = np.linspace(-2, 2, 401)
    X_sttr = np.linspace(-2, 2, 21)

    A = compute_dog_coeff(sigma, s, r, n)
    ax[0].plot(X, Z(X, 0, sigma), color='b')
    ax[0].scatter(X_sttr, plot_dog_approx(X_sttr, s, r, A), color='r', s=20)
    dog_dc, = ax[0].plot(X, Z(X, 0, sigma) - plot_dog_approx(X, s, r, A), color='r', label=f'MISE: {round(err(sigma, s, r, n, A), 10)}')
    ax[0].axhline(color='k', linestyle='--')
    for i in range(-n, n+1):
        ax[0].plot(X, A[i+n]*Z(X, i*r, s), linestyle=':', color='k')
    ax[0].set_title(f"DoG Interpolation (n={2*n-1})")
    ax[0].legend(handles=[dog_dc])
    print(f"DoG Error: {err_op(sigma, s, r, n, A) / (sigma*np.sqrt(np.pi))}")

    A_rbf = np.zeros(2*n+1)
    A_rbf_input = coefficients(r, s, sigma, n+1)
    for i in range(-n, n+1):
        A_rbf[i+n] = A_rbf_input[abs(i)]
    ax[1].plot(X, Z(X, 0, sigma), color='b')
    ax[1].scatter(X_sttr, plot_dog_approx(X_sttr, s, r, A_rbf), color='r', s=20)
    rbf_dc, = ax[1].plot(X, Z(X, 0, sigma) - plot_dog_approx(X, s, r, A_rbf), color='r', label=f'MISE: {round(err(sigma, s, r, n, A_rbf), 10)}')
    ax[1].axhline(color='k', linestyle='--')
    for i in range(-n, n+1):
        ax[1].plot(X, A_rbf[i+n] * Z(X, i * r, s), linestyle=':', color='k')
    ax[1].set_title(f"RBF Interpolation (n={2*n-1})")
    ax[1].legend(handles=[rbf_dc])
    print(f"Standard RBF Error: {err(sigma, s, r, n, A_rbf) / (sigma*np.sqrt(np.pi))}")

    plt.show()


def direct_convolution(sigma, r_max, r_prec):
    x = np.linspace(-r_max//2, r_max//2, r_max*r_prec)
    return np.e**(-x**2/(2*sigma**2))


def dog_convolution(sigma, s, r, n):
    A = compute_dog_coeff(sigma, s, r, n)
    err = 0
    return A, err, r


def time_comparison():
    def get_std(fwhm):
        return fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = get_std(1)
    s = get_std(0.6)
    n = 2
    r_prec = 100
    r_max = 8
    iter = 1000

    start = time.time()
    x = np.linspace(-r_max//2, r_max//2, r_max*r_prec)
    r = greedy_r(sigma, s)
    r = 0.375
    exprs = get_expr_dict(s, r, n)
    t_err = 0
    for i in range(iter):
        A = analytic_dog_coeff(sigma, s, n, exprs)
        for j in range(-n+1, n):
            x[int(r*r_prec) + r_max*r_prec//2] += A[j]
    print(f"DoG Convolution Time: {time.time() - start}")

    start = time.time()
    x = np.linspace(-r_max // 2, r_max // 2, r_max * r_prec)
    r = greedy_r(sigma, s)
    r = 0.375
    t_err = 0
    for i in range(iter):
        A = compute_dog_coeff(sigma, s, r, n)
        t_err += err_op(sigma, s, r, n, A) / (sigma*np.sqrt(np.pi))
    print(f"DoG Error Time: {time.time() - start}, Average Error: {t_err / iter * 100}%")

    start = time.time()
    x = np.linspace(-r_max//2, r_max//2, r_max*r_prec)
    for i in range(iter):
        x += direct_convolution(sigma, r_max, r_prec)
    print(f"Direct Convolution Time: {time.time() - start}")


def minimal_comparison():
    pass


if __name__ == "__main__":
    cProfile.run('time_comparison()')
