from amopt.pricers.american_fd import american_fd_surface

import numpy as np


def extract_boundary(option_type, K, r, q, sigma, T, stock_intervals, time_intervals):
    V, s_grid = american_fd_surface(
        option_type, K, r, q, sigma, T, stock_intervals, time_intervals
    )

    # Payoff
    if option_type.lower() == 'call':
        payoff = np.maximum(s_grid - K, 0.0)
    else:
        payoff = np.maximum(K - s_grid, 0.0)

    # Use t = 0 slice
    V_t0 = V[:, 0]

    # Exclude spatial boundaries
    V_int = V_t0[1:-1]
    S_int = s_grid[1:-1]
    payoff_int = payoff[1:-1]

    diff = V_int - payoff_int

    # ---- Handle degenerate cases first ----
    if np.all(diff > 0):
        # Continuation everywhere → no early exercise
        return np.nan

    if np.all(diff <= 0):
        # Immediate exercise everywhere
        return S_int[-1] if option_type == 'put' else S_int[0]

    # ---- Genuine free boundary exists ----
    if option_type.lower() == 'call':
        # First S where continuation starts
        idx = np.where(diff > 0)[0][0]
    else:
        # Last S where continuation starts
        idx = np.where(diff > 0)[0][-1]

    return S_int[idx]


def _interpolate_boundary(s_grid, diff, lower_idx, upper_idx, target):
    s0, s1 = s_grid[lower_idx], s_grid[upper_idx]
    d0, d1 = diff[lower_idx], diff[upper_idx]
    if d1 == d0:
        return s0
    return s0 + (target - d0) * (s1 - s0) / (d1 - d0)


def _boundary_from_diff(option_type, s_grid, diff, tol):
    exercise_mask = diff <= tol
    exercise_idx = np.where(exercise_mask)[0]
    option = option_type.lower()

    if exercise_idx.size == 0:
        return s_grid[-1] if option == "call" else s_grid[0]

    if exercise_idx.size == s_grid.size:
        return s_grid[0] if option == "call" else s_grid[-1]

    if option == "call":
        first_exercise = exercise_idx[0]
        lower_idx = first_exercise - 1
        upper_idx = first_exercise
    else:
        last_exercise = exercise_idx[-1]
        lower_idx = last_exercise
        upper_idx = last_exercise + 1

    return _interpolate_boundary(s_grid, diff, lower_idx, upper_idx, tol)


def extract_boundary_curve(option_type, K, r, q, sigma, T, stock_intervals, time_intervals):
    V, s_grid = american_fd_surface(
        option_type, K, r, q, sigma, T, stock_intervals, time_intervals
    )

    boundary = np.zeros(V.shape[1])

    # Payoff
    option = option_type.lower()
    if option == "call":
        payoff = np.maximum(s_grid - K, 0.0)
    else:
        payoff = np.maximum(K - s_grid, 0.0)

    # Interior nodes only
    V_int = V[1:-1, :]
    S_int = s_grid[1:-1]
    payoff_int = payoff[1:-1]

    tol = max(1e-10, 1e-6 * K)

    for n in range(V.shape[1]):
        diff = V_int[:, n] - payoff_int
        boundary[n] = _boundary_from_diff(option, S_int, diff, tol)

    return boundary
































