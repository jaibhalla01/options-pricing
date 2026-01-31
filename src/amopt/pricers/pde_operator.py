import numpy as np


def crank_nicholson_imp_coefficients(L, D, U, dt):
    N = len(L)
    L_imp = np.zeros(N)
    D_imp = np.zeros(N)
    U_imp = np.zeros(N)

    for i in range(N):
        L_imp[i] = -0.5 * dt * L[i]
        D_imp[i] = 1 - 0.5 * dt * D[i]
        U_imp[i] = -0.5 * dt * U[i]

    return L_imp, D_imp, U_imp


def crank_nicholson_exp_coefficients(L, D, U, dt):
    N = len(L)
    L_exp = np.zeros(N)
    D_exp = np.zeros(N)
    U_exp = np.zeros(N)

    for i in range(N):
        L_exp[i] = 0.5 * dt * L[i]
        D_exp[i] = 1 + 0.5 * dt * D[i]
        U_exp[i] = 0.5 * dt * U[i]

    return L_exp, D_exp, U_exp
