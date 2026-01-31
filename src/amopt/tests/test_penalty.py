from amopt.pricers.american_fd import american_fd_pricer, american_fd_surface
from amopt.pricers.closed_form import euro_vanilla_price
from amopt.pricers.american_binomial import american_binomial_price

import pytest
import numpy as np


@pytest.fixture
def fd_surface(request):
    option_type = request.param
    if option_type.lower() == 'call':
        fd_surface_values, s_grid = american_fd_surface('call', 100, 0.05, 0.02, 0.2,
                                                        1.0, 80, 80)
    else:
        fd_surface_values, s_grid = american_fd_surface('put', 100, 0.05, 0.02, 0.2,
                                                        1.0, 80, 80)

    return fd_surface_values[:, 0], s_grid


@pytest.fixture
def default_american_option_price(request):
    option_type = request.param
    if option_type.lower() == 'call':
        american_opt_value = american_fd_pricer('call', 100, 100, 0.05, 0.02, 0.2,
                                                1.0, 100, 100)
    else:
        american_opt_value = american_fd_pricer('put', 100, 100, 0.05, 0.02, 0.2,
                                                1.0, 100, 100)

    return american_opt_value


@pytest.fixture
def default_european_option_price(request):
    option_type = request.param
    if option_type == 'call':
        euro_opt_value = euro_vanilla_price('call', 100, 100, 0.05, 1.0, 0.2, 0.02)
    else:
        euro_opt_value = euro_vanilla_price('put', 100, 100, 0.05, 1.0, 0.2, 0.02)

    return euro_opt_value


@pytest.mark.parametrize("default_american_option_price", ["call"], indirect=True)
def test_american_binomial_vs_penalty_pde_solver(default_american_option_price):
    fd_price = default_american_option_price
    binomial_price = american_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 800)

    assert fd_price == pytest.approx(binomial_price, rel=1e-2)


# To run fast tests: pytest -m "not slow"
# Tests receive fixture outputs
# indirect=True => pass the param into the fixture not the tests
# American options can never be worth less than it's European counterpart
@pytest.mark.parametrize("default_american_option_price, default_european_option_price", [("call", "call")],
                         indirect=True)
def test_american_call_greater_than_or_equal_to_euro(default_american_option_price, default_european_option_price):
    # Up to 1% numerical error, the American call is not cheaper
    tolerance = 1e-2 * default_european_option_price
    assert default_american_option_price + tolerance >= default_european_option_price


@pytest.mark.parametrize("default_american_option_price, default_european_option_price", [("put", "put")],
                         indirect=True)
def test_american_put_greater_than_or_equal_to_euro(default_american_option_price, default_european_option_price):
    assert default_american_option_price >= default_european_option_price


def test_no_dividend_american_equals_european():
    american_opt_value = american_fd_pricer('call', 100, 100, 0.05, 0.0, 0.2,
                                            1.0, 200, 200)
    euro_opt_value = euro_vanilla_price('call', 100, 100, 0.05, 1.0, 0.2, 0.0)

    assert american_opt_value == pytest.approx(euro_opt_value, rel=1e-2)


# To run slow tests pytest -m slow
# To exclude slow tests pytest -m "not slow"
@pytest.mark.parametrize("default_american_option_price", ["call"], indirect=True)
@pytest.mark.slow
def test_penalty_convergence(default_american_option_price):
    ref_price = american_fd_pricer('call', 100, 100, 0.05, 0.02, 0.2, 1.0,
                                   300, 300)
    fine_price = american_fd_pricer('call', 100, 100, 0.05, 0.02, 0.2, 1.0,
                                    200, 200)
    coarse_price = default_american_option_price

    assert np.abs(ref_price - fine_price) < np.abs(ref_price - coarse_price)


# This test verifies free-boundary enforcement for the American option.
#
# Financially/Theoretically, an American option value must never fall below its intrinsic
# (exercise) value. However, because we solve the problem numerically using a
# penalised PDE method, small violations can occur due to discretisation and
# convergence tolerance.
#
# Instead of requiring strict inequality at every grid point, we measure the
# worst downward violation across the entire spatial grid and assert that it
# remains within an acceptable numerical tolerance ε.
@pytest.mark.parametrize("fd_surface", ["call"], indirect=True)
def test_american_call_option_greater_than_or_equal_to_intrinsic_value(fd_surface):
    epsilon = 1e-3

    V, S = fd_surface
    K = 100

    # interior nodes only (exclude boundaries)
    V_int = V[1:-1]
    S_int = S[1:-1]

    # Intrinsic value at every grid point
    payoff = np.maximum(S_int - K, 0)

    # Find the most negative violation across the grid
    violation = np.min(V_int - payoff)

    # The American option value may dip slightly below intrinsic value, but the maximum violation must be smaller than ε
    assert violation >= -epsilon


# For American puts, early exercise is optimal in some regions.
# Numerical PDE solutions may slightly violate the inequality near
# the free boundary, so we allow a small tolerance ε.
@pytest.mark.parametrize("fd_surface", ["put"], indirect=True)
def test_american_put_option_greater_than_or_equal_to_intrinsic_value(fd_surface):
    epsilon = 1e-3

    V, S = fd_surface
    K = 100

    # interior nodes only (exclude boundaries)
    V_int = V[1:-1]
    S_int = S[1:-1]

    # Intrinsic value at every grid point
    payoff = np.maximum(K - S_int, 0)

    # Find the most negative violation across the grid
    violation = np.min(V_int - payoff)

    # The American option value may dip slightly below intrinsic value, but the maximum violation must be smaller than ε
    assert violation >= -epsilon


# This test checks that moving forward along the stock price grid never causes the option value to decrease, which must
# be true for a call option.
@pytest.mark.parametrize("fd_surface", ["call"], indirect=True)
def test_call_value_increases_as_spot_price_increases(fd_surface):
    V, S = fd_surface

    # interior nodes only
    V_int = V[1:-1]

    # forward differences
    diffs = V_int[1:] - V_int[:-1]

    # allow tiny numerical noise
    assert np.all(diffs >= -1e-6)


# As the stock price increases, the option value should not increase.
@pytest.mark.parametrize("fd_surface", ["put"], indirect=True)
def test_put_value_decreases_as_spot_price_increases(fd_surface):
    V, S = fd_surface

    # interior nodes only
    V_int = V[1:-1]

    # backward differences
    diffs = V_int[:-1] - V_int[1:]

    # allow tiny numerical noise
    assert np.all(diffs >= -1e-6)


@pytest.mark.parametrize("default_american_option_price", ["call"], indirect=True)
def test_call_value_decreases_as_strike_price_increases(default_american_option_price):
    new_K = 120

    new_c_value = american_fd_pricer('call', 100, new_K, 0.05, 0.02, 0.2,
                                     1.0, 50, 50)

    assert new_c_value <= default_american_option_price


@pytest.mark.parametrize("default_american_option_price", ["put"], indirect=True)
def test_put_value_increases_as_strike_price_increases(default_american_option_price):
    new_K = 120

    new_p_value = american_fd_pricer('put', 100, new_K, 0.05, 0.02, 0.2,
                                     1.0, 50, 50)

    assert new_p_value >= default_american_option_price

