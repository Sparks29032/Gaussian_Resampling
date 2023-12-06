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


if __name__ == '__main__':
    lay_sticks(0, 1, 0.6, 3, 0.37, -2, 2, 0.01, plot=True)
