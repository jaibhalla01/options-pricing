from amopt.dataclasses.grids import time_grid, stock_grid
from amopt.uniform.operators import bs_spatial_operator
from amopt.pricers.pde_operator import crank_nicholson_exp_coefficients, crank_nicholson_imp_coefficients
from amopt.pricers.pde_solver import time_marching_pde_solver

import numpy as np


# STEP 1: Create function that accepts market parameters and returns an option value
def european_fd_pricer(option_type, S, K, r, q, sigma, T, stock_intervals, time_intervals):

    # Set value for Smax to truncate the numerical model to an upper boundary instead of infinity - allowing computation
    dt = T / time_intervals
    Smax = 4 * K

    # STEP 2: Construct time and space grid
    s_grid = stock_grid(Smax, stock_intervals)
    t_grid = time_grid(T, time_intervals)

    # STEP 3: Build PDE operator
    L, D, U = bs_spatial_operator(s_grid, r, q, sigma)

    # STEP 4: Create Crank-Nicolson coefficients
    L_imp, D_imp, U_imp = crank_nicholson_imp_coefficients(L, D, U, dt)
    L_exp, D_exp, U_exp = crank_nicholson_exp_coefficients(L, D, U, dt)

    # STEP 5: Call PDE solver
    european_fd_prices = time_marching_pde_solver(option_type, L_imp, D_imp, U_imp, L_exp, D_exp, U_exp, K, s_grid, T, r, q, time_intervals)

    # STEP 6: Return option price at S0 and t=0 - this is already the case for the array we've produced
    # Find index of grid point closest to S0
    idx = np.argmin(np.abs(s_grid - S))

    # Extract option price at S0
    price = european_fd_prices[idx, 0]

    return price


