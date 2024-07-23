from linear_system_rbf import *


def lay_sticks(u, fwhm_target, fwhm_approximator, n, r, r_min, r_max, dr, plot=True):
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

    sticks = np.zeros(r_n)

    for xi in x:
        if u - dr/2 <= xi < u + dr/2:
            sticks[x == xi] = c * a[0]
    for i in range(1, n):
        for xi in x:
            if u + i*r - dr/2 <= xi < u + i*r + dr/2:
                sticks[x == xi] = c * a[i]
            if u - i*r - dr/2 <= xi < u - i*r + dr/2:
                sticks[x == xi] = c * a[i]

    g_thin = unit_gaussian(x, u, s)
    g_convolve = np.convolve(sticks, g_thin, mode='same')
    x_convolve = np.linspace(r_min, r_max, 2 * r_n - 1)

    if plot:
        stks, = plt.plot(x, sticks, color='k', label='Sticks')
        cnvl, = plt.plot(x, g_convolve, color='b', label='Convolution')

        sep = 5
        trgt = plt.scatter(x[::sep], g_target[::sep], color='r', s=5, label='Target')

        plt.legend(handles=[stks, cnvl, trgt])
        plt.show()

    abs_err = g_convolve - g_target
    rel_err = (g_convolve - g_target) / g_target

    return abs(abs_err), abs(rel_err)


def abs_err_min(FWHM_target, FWHM_approx, std, red=1, sticks=None):
    """
    Decides spacing based on where the absolute error between the two Gaussians is maximized.
    Approximation accurate up to a user-specified number of standard deviations.

    Parameters
    ----------
    FWHM_target: FWHM of target Gaussian
    FWHM_approx: FWHM of Gaussians used to approximate the target
    std: How many standard deviations of accuracy
    red: Scale factor to reduce spacing by
    sticks: Specify number of sticks to use
    """

    rmax = FWHM_target * 2
    b = FWHM_target / (2 * np.sqrt(2 * np.log(2)))
    a = FWHM_approx / (2 * np.sqrt(2 * np.log(2)))

    d = (4 * np.log(b / a) * (a ** 2 * b ** 2) / (b ** 2 - a ** 2))**(1/2) / red
    if sticks is None:
        n = int(b * std / d)
    else:
        n = int((sticks - 1) / 2)
    abs_err, rel_err = lay_sticks(0, FWHM_target, FWHM_approx, n, d, -rmax, rmax, 0.01, plot=True)

    print(f"Sticks used: {2 * n + 1}")
    print(f"Absolute error: {max(abs_err)}")
    print(f"Relative error: {rel_err[np.argmax(abs_err)] * 100}%")

if __name__ == '__main__':
    abs_err_min(np.sqrt(10), 1, 3, red=1.2)
