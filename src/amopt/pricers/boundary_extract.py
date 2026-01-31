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


def extract_boundary_curve(option_type, K, r, q, sigma, T, stock_intervals, time_intervals):
    V, s_grid = american_fd_surface(
        option_type, K, r, q, sigma, T, stock_intervals, time_intervals
    )

    boundary = np.zeros(V.shape[1])

    # Payoff
    if option_type.lower() == "call":
        payoff = np.maximum(s_grid - K, 0.0)
    else:
        payoff = np.maximum(K - s_grid, 0.0)

    # Interior nodes only
    V_int = V[1:-1, :]
    S_int = s_grid[1:-1]
    payoff_int = payoff[1:-1]

    for n in range(V.shape[1]):
        diff = V_int[:, n] - payoff_int

        exercise_idx = np.where(diff <= 0)[0]
        continuation_idx = np.where(diff > 0)[0]

        # Case 1: continuation everywhere → no early exercise
        if len(exercise_idx) == 0:
            boundary[n] = S_int[-1] if option_type == "call" else S_int[0]
            continue

        # Case 2: exercise everywhere → immediate exercise
        if len(continuation_idx) == 0:
            boundary[n] = S_int[0] if option_type == "call" else S_int[-1]
            continue

        # Case 3: genuine free boundary
        if option_type == "call":
            # lowest S where continuation starts
            boundary[n] = S_int[continuation_idx[0]]
        else:
            # highest S where continuation starts
            boundary[n] = S_int[continuation_idx[-1]]

    return boundary

































