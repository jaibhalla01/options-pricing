from amopt.pricers.closed_form import euro_vanilla_price
from amopt.pricers.european_fd import european_fd_pricer

import numpy as np


# Helper function
def calculate_option_price_error(option_type, S, K, r, q, sigma, T, stock_intervals, time_intervals):
    fd_price = european_fd_pricer(option_type, S, K, r, q, sigma, T, stock_intervals, time_intervals)
    bs_price = euro_vanilla_price(option_type, S, K, r, T, sigma, q)

    error = np.abs(fd_price - bs_price)
    return error


def test_error_tolerance():
    # Grid chosen empirically to achieve <1% absolute error
    error = calculate_option_price_error('call', 100, 100, 0.05, 0, 0.2, 1.0, 800, 800)
    tolerance = 1e-2
    assert error < tolerance


def test_convergence_error():
    error_fine = calculate_option_price_error('call', 100, 100, 0.05, 0, 0.2, 1.0, 200, 200)
    error_coarse = calculate_option_price_error('call', 100, 100, 0.05, 0, 0.2, 1.0, 50, 50)
    assert error_fine < error_coarse
