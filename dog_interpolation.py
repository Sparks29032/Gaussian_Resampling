import numpy as np
import matplotlib.pyplot as plt
from linear_system_rbf import *
import scipy

def Z(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))


def compute_dog_coeff(sigma, s, r, n: int):
    n = n+1
    gamma = np.exp(-r**2 / (4*s**2))
    beta = sigma**2 / (sigma**2+s**2)
    alpha = np.exp(-(beta/(2*sigma**2))*r**2)

    C_mat = np.zeros((n, n))
    for i in range(0, n):
        C_mat[i][0] = gamma**(i**2)
        for j in range(1, n):
            C_mat[i][j] = (gamma**((i-j)**2) + gamma**((i+j)**2))
    b_vec = np.zeros(n)
    for i in range(0, n):
        b_vec[i] = alpha**(i**2) * np.sqrt(2*beta)

    A_half = np.linalg.solve(C_mat, b_vec)
    A = np.zeros(2*n-1)
    A[n-1:] = A_half
    A[0:n-1] = A_half[:0:-1]
    return A


def plot_dog_approx(x, s, r, A):
    n = len(A)//2 + 1
    sog = 0
    for i in range(-n+1, n):
        sog += A[i+n-1] * Z(x, i*r, s)
    return sog


def err(sigma, s, r, n: int, A):
    n = n+1

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


def err_op(sigma, s, r, n: int, A):
    n = n+1
    if len(A)//2+1 != n:
        return -1

    # Compute
    gamma = np.exp(-r**2/(4*s**2))
    beta = sigma**2/(sigma**2 + s**2)
    alpha = np.exp(-(beta/(2*sigma**2))*r**2)
    e = sigma*np.sqrt(np.pi)

    for i in range(-n+1, n):
        e -= A[i+n-1] * alpha**(i**2) * s * np.sqrt(2*np.pi*beta)

    return e


def greedy_r(sigma, s):
    return s * sigma * np.sqrt(4 * np.log(sigma / s) / (sigma**2 - s**2))
