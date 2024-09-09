import sympy
import numpy as np
from linear_system_rbf import *


def analytic_coefficients(n):
    e = sympy.symbols('e')
    r, s, o = sympy.symbols('r s o')
    cf_str = ""
    for i in range(n):
        cf_str += f"a{i} "
    a = sympy.symbols(cf_str)

    def gaussian(f, sd):
        return e ** (-(f * r) ** 2 / (2 * sd ** 2))

    eqns = []
    for i in range(n):
        eqn = gaussian(i, s) * a[0]
        for j in range(1, n):
            eqn += (gaussian(i - j, s) + gaussian(i + j, s)) * a[j]
        eqn -= gaussian(i, o)
        eqns.append(eqn)
    solutions = sympy.solve(eqns, a)

    for ai in solutions.keys():
        solutions[ai] = sympy.simplify(solutions[ai])

    return solutions


def ni(i):
    n = i
    ra, oa, sa = 0.37, 1 / (2 * np.sqrt(2 * np.log(2))), 0.6 / (2 * np.sqrt(2 * np.log(2)))
    e, r, o, s = sympy.symbols('e r o s')
    acoef = analytic_coefficients(n)
    for ai in acoef.keys():
        print(f"{ai} = {acoef[ai]}")
    ncoef = coefficients(r=ra, s=sa, o=oa, n=n)
    for i in range(n):
        ai = sympy.symbols(f'a{i}')
        analytic_coef = acoef[ai].subs([(e, np.e), (r, ra), (o, oa), (s, sa)])
        numerical_coef = ncoef[i]
        sentence = f"[A_{i}] Analytic: {analytic_coef}, Numerical: {numerical_coef}"

        print(sentence)


ni(4)
