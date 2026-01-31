import numpy as np


def stock_grid(Smax, steps):
    delta = Smax / steps
    S = delta * np.arange(0, steps + 1, 1)
    return S


def time_grid(T, Nt):
    dt = T / Nt
    t = dt * np.arange(0, Nt + 1, 1)
    return t
