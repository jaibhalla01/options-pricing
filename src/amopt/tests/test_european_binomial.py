import numpy as np
import pytest

from amopt.pricers.closed_form import euro_vanilla_price
from amopt.pricers.european_fd import european_fd_pricer
from amopt.pricers.binomial import european_binomial_price


@pytest.fixture
def binomial_european_price(request):
    option_type = request.param
    if option_type.lower() == "call":
        euro_opt_value = european_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 800)
    else:
        euro_opt_value = european_binomial_price('put', 100, 100, 0.05, 0.02, 0.2, 1, 800)

    return euro_opt_value


@pytest.mark.parametrize("binomial_european_price", ['call'], indirect=True)
def test_binomial_pricer_vs_closed_form_pricer(binomial_european_price):
    closed_form_price = euro_vanilla_price('call', 100, 100, 0.05, 1, 0.2, 0.02)

    assert binomial_european_price == pytest.approx(closed_form_price, rel=1e-3)


@pytest.mark.parametrize("binomial_european_price", ['call'], indirect=True)
def test_binomial_pricer_vs_fd_pricer(binomial_european_price):
    fd_price = european_fd_pricer('call', 100, 100, 0.05, 0.02, 0.2, 1, 200, 200)

    assert binomial_european_price == pytest.approx(fd_price, rel=1e-2)


@pytest.mark.parametrize("binomial_european_price", ['call'], indirect=True)
def test_binomial_convergence(binomial_european_price):
    ref_price = binomial_european_price
    fine_price = european_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 600)
    coarse_price = european_binomial_price('call', 100, 100, 0.05, 0.02, 0.2, 1, 200)

    assert np.abs(ref_price - fine_price) <= np.abs(ref_price - coarse_price)


@pytest.mark.parametrize("binomial_european_price", ['put'], indirect=True)
def test_binomial_put_vs_closed_form_put(binomial_european_price):
    closed_form_price = euro_vanilla_price('put', 100, 100, 0.05, 1, 0.2, 0.02)

    assert binomial_european_price == pytest.approx(closed_form_price, rel=1e-3)


@pytest.mark.parametrize("binomial_european_price", ['put'], indirect=True)
def test_put_decreases_as_spot_increases(binomial_european_price):
    S_new = 150
    p_value_new = european_binomial_price('put', S_new, 100, 0.05, 0.02, 0.2, 1, 800)

    assert p_value_new <= binomial_european_price


@pytest.mark.parametrize("binomial_european_price", ['call'], indirect=True)
def test_call_increases_as_spot_increases(binomial_european_price):
    S_new = 150
    c_value_new = european_binomial_price('call', S_new, 100, 0.05, 0.02, 0.2, 1, 800)

    assert c_value_new >= binomial_european_price




















