from amopt.dataclasses.grids import time_grid, stock_grid
from amopt.uniform.operators import bs_spatial_operator
from amopt.pricers.pde_operator import crank_nicholson_exp_coefficients, crank_nicholson_imp_coefficients
from amopt.lcp.penalty import penalty_pde_solver
from amopt.pricers.pde_solver import time_marching_pde_solver

import numpy as np


def american_fd_pricer(option_type, S, K, r, q, sigma, T, stock_intervals, time_intervals):
    """The pricing logic is intentionally split into two functions.
        american_fd_pricer provides a simple, user-facing API that returns a single option price at a given spot, while
        american_fd_surface exposes the full finite-difference solution and spatial grid.
        This separation keeps the public interface clean while enabling robust testing and validation, such as
        free-boundary enforcement, convergence checks, and intrinsic value comparisons, which require access to the
        entire price surface rather than a single interpolated value."""

    fd_prices, s_grid = american_fd_surface(option_type, K, r, q, sigma, T, stock_intervals, time_intervals)
    # Find which grid price is closest to the actual spot price S
    idx = np.argmin(np.abs(s_grid - S))

    # Extract option value when t=0
    price = fd_prices[idx, 0]

    return price


# The solver is split into a surface-level function and a scalar pricer to keep the public API simple while allowing
# tests to access the full grid for convergence, boundary, and payoff validation.
def american_fd_surface(option_type, K, r, q, sigma, T, stock_intervals, time_intervals):
    dt = T / time_intervals
    Smax = 4 * K

    s_grid = stock_grid(Smax, stock_intervals)
    # t_grid = time_grid(T, time_intervals)

    L, D, U = bs_spatial_operator(s_grid, r, q, sigma)

    L_imp, D_imp, U_imp = crank_nicholson_imp_coefficients(L, D, U, dt)
    L_exp, D_exp, U_exp = crank_nicholson_exp_coefficients(L, D, U, dt)

    # Financial logic: no early exercise for calls with zero dividends
    # For a call option with no dividends early-exercise is never optimal
    # Calls should only be exercised if dividends make holding suboptimal
    if q == 0 and option_type.lower() == 'call':
        fd_prices = time_marching_pde_solver(option_type, L_imp, D_imp, U_imp,
                                             L_exp, D_exp, U_exp,
                                             K, s_grid, T, r, q, time_intervals)
    else:
        fd_prices = penalty_pde_solver(option_type, L_imp, D_imp, U_imp,
                                       L_exp, D_exp, U_exp,
                                       K, s_grid, T, r, q, time_intervals)

    return fd_prices, s_grid
