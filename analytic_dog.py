import sympy
import numpy as np

def square_system(n):
    alpha, gamma = sympy.symbols("α γ")
    cf_str = ""
    for i in range(0, n+1):
        cf_str += f"A{i} "
    A = sympy.symbols(cf_str)
    if type(A) is sympy.Symbol:
        A = [A]

    eqns = []
    for i in range(0, n+1):
        eqn = A[0] * gamma**(i**2)
        for j in range(1, n+1):
            eqn += A[j] * (gamma**((i-j)**2) + gamma**((i+j)**2))
        eqn -= alpha**(i**2)
        eqns.append(eqn)
    solutions = sympy.solve(eqns, A)

    for Ai in solutions.keys():
        solutions[Ai] = sympy.simplify(solutions[Ai])

    return solutions


def get_expr_dict(s, r, n):
    gamma = np.exp(-r ** 2 / (4 * s ** 2))
    exprs = {}
    if n==1:
        exprs.update({"g2": gamma**2, "g4": gamma**4, "g4-2g2+1": gamma**4-2*gamma**2+1})
    if n==2:
        exprs.update({"g2": gamma**2, "g3": gamma**3, "g4": gamma**4, "g5": gamma**5, "g6": gamma**6, "g7": gamma**7,
                     "g8": gamma**8, "g9": gamma**9, "g10": gamma**10, "g12": gamma**12,
                      "g4-2g2+1": gamma ** 4 - 2 * gamma ** 2 + 1,
                      "g16-g14-2g12+g10+2g8+g6-2g4-g2+1": gamma**16-gamma**14+2*gamma**12+gamma*810+2*gamma*8+gamma**6-2*gamma*4-gamma*2+1})
    return exprs


def analytic_dog_coeff(sigma, s, n, exprs):
    beta = sigma**2 / (sigma**2+s**2)
    exprs.update({"q2b": np.sqrt(2*beta)})
    if n==0:
        return [exprs["q2b"]], 0

    r = s * sigma * np.sqrt(4 * np.log(sigma / s) / (sigma**2 - s**2))
    alpha = np.exp(-(beta/(2*sigma**2))*r**2)
    gamma = np.exp(-r**2 / (4*s**2))
    if n==1:
        A0 = exprs["q2b"] * (-2*alpha*gamma + exprs["g4"] + 1) / exprs["g4-2g2+1"]
        A1 = exprs["q2b"] * (alpha - gamma) / exprs["g4-2g2+1"]
        return [A1, A0, A1]
    if n==2:
        A0 = exprs["q2b"] * (-2*alpha**4*exprs["g2"] + 2*alpha*exprs["g9"] + 2*alpha*exprs["g7"] + 2*alpha*exprs["g3"] + 2*alpha*gamma - exprs["g12"] - exprs["g8"] - 2*exprs["g6"] - exprs["g4"] - 1) / exprs["g16-g14-2g12+g10+2g8+g6-2g4-g2+1"]
        A1 = exprs["q2b"] * (-alpha**4*gamma + alpha*exprs["g8"] + 2*alpha*exprs["g4"] + alpha - exprs["g9"] - exprs["g5"] - gamma) / exprs["g16-g14-2g12+g10+2g8+g6-2g4-g2+1"]
        A2 = exprs["q2b"] * (alpha**4 - alpha*exprs["g5"] - 2*alpha*exprs["g3"] - alpha*gamma + exprs["g6"] + exprs["g4"] + exprs["g2"]) / exprs["g16-g14-2g12+g10+2g8+g6-2g4-g2+1"]
        return [A2, A1, A0, A1, A2]


if __name__ == "__main__":
    A = square_system(1)
    print(A)
