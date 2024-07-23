import numpy as np
import matplotlib.pyplot as plt
from linear_system_rbf import *
import scipy

def Z(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))


def compute_dog_coeff(sigma, s, r, n: int):
    gamma = np.exp(-r**2 / (4*s**2))
    beta = sigma**2 / (sigma**2+s**2)
    alpha = np.exp(-(beta/(2*sigma**2))*r**2)

    C_mat = np.zeros((2*n-1, 2*n-1))
    for i in range(-n+1, n):
        for j in range(-n+1, n):
            C_mat[i+n-1][j+n-1] = gamma**((i-j)**2) * s*np.sqrt(np.pi)
    b_vec = np.zeros(2*n-1)
    for i in range(-n+1, n):
        b_vec[i+n-1] = alpha**(i**2) * s*np.sqrt(2*np.pi*beta)

    A = np.linalg.solve(C_mat, b_vec)
    return A


def plot_dog_approx(x, s, r, A):
    n = len(A)//2 + 1
    sog = 0
    for i in range(-n+1, n):
        sog += A[i+n-1] * Z(x, i*r, s)
    return sog


def err(sigma, s, r, n: int, A):
    # Input mismatch
    if len(A)//2+1 != n:
        return -1

    # Compute
    gamma = np.exp(-r**2/(4*s**2))
    beta = sigma**2/(sigma**2 + s**2)
    alpha = np.exp(-(beta/(2*sigma**2))*r**2)
    e = sigma*np.sqrt(np.pi)

    for i in range(-n+1, n):
        e -= 2 * A[i+n-1] * alpha**(i**2) * s * np.sqrt(2*np.pi*beta)
        for j in range(-n+1, n):
            e += A[i+n-1] * A[j+n-1] * gamma**((i-j)**2) * s * np.sqrt(np.pi)

    return e


if __name__ == "__main__":
    def get_std(fwhm):
        return fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma = get_std(1)
    s = get_std(0.6)
    r = 0.375
    n = 3

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    X = np.linspace(-2, 2, 401)
    X_sttr = np.linspace(-2, 2, 21)

    A = compute_dog_coeff(sigma, s, r, n)
    ax[0].plot(X, Z(X, 0, sigma), color='b')
    ax[0].scatter(X_sttr, plot_dog_approx(X_sttr, s, r, A), color='r', s=20)
    dog_dc, = ax[0].plot(X, Z(X, 0, sigma) - plot_dog_approx(X, s, r, A), color='r', label=f'MISE: {round(err(sigma, s, r, n, A), 10)}')
    ax[0].axhline(color='k', linestyle='--')
    for i in range(-n+1, n):
        ax[0].plot(X, A[i+n-1]*Z(X, i*r, s), linestyle=':', color='k')
    ax[0].set_title(f"DoG Interpolation (n={2*n-1})")
    ax[0].legend(handles=[dog_dc])
    print(f"DoG Error: {err(sigma, s, r, n, A)}")

    A_rbf = np.zeros(2*n-1)
    A_rbf_input = coefficients(r, s, sigma, n)
    for i in range(-n+1, n):
        A_rbf[i+n-1] = A_rbf_input[abs(i)]
    ax[1].plot(X, Z(X, 0, sigma), color='b')
    ax[1].scatter(X_sttr, plot_dog_approx(X_sttr, s, r, A_rbf), color='r', s=20)
    rbf_dc, = ax[1].plot(X, Z(X, 0, sigma) - plot_dog_approx(X, s, r, A_rbf), color='r', label=f'MISE: {round(err(sigma, s, r, n, A_rbf), 10)}')
    ax[1].axhline(color='k', linestyle='--')
    for i in range(-n + 1, n):
        ax[1].plot(X, A_rbf[i + n - 1] * Z(X, i * r, s), linestyle=':', color='k')
    ax[1].set_title(f"RBF Interpolation (n={2*n-1})")
    ax[1].legend(handles=[rbf_dc])
    print(f"Standard RBF Error: {err(sigma, s, r, n, A_rbf)}")

    plt.show()
