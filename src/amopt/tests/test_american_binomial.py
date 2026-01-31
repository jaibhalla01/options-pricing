import pytest
import numpy as np

from amopt.pricers.american_binomial import american_binomial_price
from amopt.pricers.european_fd import european_fd_pricer


@pytest.fixture
def binomial_american_option_price(request):
    option_type = request.param

    if option_type.lower() == 'call':
        binomial_price = american_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 800)
    else:
        binomial_price = american_binomial_price('put', 100, 100, 0.05, 0.02, 0.2, 1, 800)

    return binomial_price


@pytest.fixture
def default_european_option_price(request):
    option_type = request.param
    if option_type == 'call':
        euro_opt_value = european_fd_pricer('call', 100, 100, 0.05, 0.02, 0.2, 1, 200, 200)
    else:
        euro_opt_value = european_fd_pricer('put', 100, 100, 0.05, 0.02, 0.2, 1, 200, 200)

    return euro_opt_value


# For calls with zero dividends early-exercise is never optimal
def test_american_binomial_converges_to_european_no_div():
    binomial_price = american_binomial_price('call', 100, 100, 0.05, 0, 0.2, 1, 800)
    euro_fd_price = european_fd_pricer('call', 100, 100, 0.05, 0, 0.2, 1, 200, 200)

    assert binomial_price == pytest.approx(euro_fd_price, rel=1e-2)


@pytest.mark.parametrize("binomial_american_option_price, default_european_option_price", [("call", "call")],
                         indirect=True)
def test_binomial_american_call_greater_than_or_equal_to_euro(binomial_american_option_price, default_european_option_price):
    assert binomial_american_option_price >= default_european_option_price


# American option should never be priced less than a European counterpart under the same model
@pytest.mark.parametrize("binomial_american_option_price, default_european_option_price", [("put", "put")],
                         indirect=True)
def test_binomial_american_put_greater_than_or_equal_to_euro(binomial_american_option_price, default_european_option_price):
    assert binomial_american_option_price >= default_european_option_price


@pytest.mark.parametrize("binomial_american_option_price", ["call"], indirect=True)
def test_binomial_convergence(binomial_american_option_price):
    ref_price = binomial_american_option_price
    fine_price = american_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 600)
    coarse_price = american_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 200)

    assert np.abs(ref_price - fine_price) <= np.abs(ref_price - coarse_price)


@pytest.mark.parametrize("binomial_american_option_price", ["call"], indirect=True)
def test_call_value_increases_as_spot_price_increases(binomial_american_option_price):
    S_new = 150
    spot_increase_price = american_binomial_price('call', S_new, 100, 0.05, 0.02, 0.2, 1, 800)
    assert spot_increase_price >= binomial_american_option_price


@pytest.mark.parametrize("binomial_american_option_price", ["put"], indirect=True)
def test_put_value_decreases_as_spot_price_increases(binomial_american_option_price):
    S_new = 150
    spot_increase_price = american_binomial_price('put', S_new, 100, 0.05, 0.02, 0.2, 1, 800)
    assert spot_increase_price <= binomial_american_option_price



