import numpy as np


def bs_spatial_operator(S, r, q, sigma):
    M = len(S)
    dS = S[1] - S[0]

    L = np.zeros(M)
    D = np.zeros(M)
    U = np.zeros(M)

    for i in range(1, M-1):
        alpha = 0.5 * (sigma**2 * S[i]**2) / dS ** 2
        beta = 0.5 * ((r-q) * S[i]) / dS

        L[i] = alpha - beta
        D[i] = -2 * alpha - r
        U[i] = alpha + beta

    return L, D, U

