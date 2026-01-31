from amopt.pricers.boundary_extract import extract_boundary_curve

import pytest
import numpy as np


@pytest.fixture()
def default_boundary_curve(request):
    option_type = request.param
    if option_type == 'call':
        call_boundary_curve = extract_boundary_curve('call', 100, 0.05, 0.02, 0.2, 2,
                                                     800, 800)
        return call_boundary_curve

    else:
        put_boundary_curve = extract_boundary_curve('put', 100, 0.07, 0.01, 0.25, 2,
                                                    800, 800)
        return put_boundary_curve


# To observe a visible boundary - increase q, increase grid size, and decrease sigma
@pytest.mark.parametrize("default_boundary_curve", ["call"], indirect=True)
@pytest.mark.slow
def test_call_with_div_boundary_is_non_increasing_wrt_time(default_boundary_curve):
    epsilon = 1e-6
    number_of_points = default_boundary_curve.shape[0]
    split = int(0.5 * number_of_points)

    boundary_t_first_partition = default_boundary_curve[:split]
    boundary_t_last_partition = default_boundary_curve[split:]

    median_first_partition = np.median(boundary_t_first_partition)
    median_last_partition = np.median(boundary_t_last_partition)

    # difference should be non-positive
    trend_wrt_time = median_last_partition - median_first_partition

    # The boundary curve is flat boundary curve is continuous however the grid is discrete hence
    # lower precision of grid spacings leads to 'equal' boundary points - this is a limitation of finite-diff solvers
    assert trend_wrt_time <= epsilon


@pytest.mark.parametrize("default_boundary_curve", ["put"], indirect=True)
@pytest.mark.slow
def test_put_boundary_increases_wrt_time(default_boundary_curve):
    epsilon = 1e-6
    number_of_points = default_boundary_curve.shape[0]
    split = int(0.5 * number_of_points)

    boundary_t_first_partition = default_boundary_curve[:split]
    boundary_t_last_partition = default_boundary_curve[split:]

    median_first_partition = np.median(boundary_t_first_partition)
    median_last_partition = np.median(boundary_t_last_partition)

    # diff should be non-negative
    trend_wrt_time = median_last_partition - median_first_partition

    assert trend_wrt_time >= -epsilon


# For a non-dividend-paying American call, early exercise is never optimal, therefore the free boundary is constant
# in time.
# TODO
# def test_no_div_call_no_meaningful_boundary():
#     dS =
#     epsilon = dS
#     boundary_curve = extract_boundary_curve('call', 100, 0.05, 0.0, 0.05, 1,
#                                             400, 400)
#
#     # Calculate all successive differences
#     diff = boundary_curve[1:] - boundary_curve[:-1]
#
#     # Check the largest absolute change in the boundary over time is negligible
#     assert np.max(np.abs(diff)) < epsilon


# As dividend yield increases it becomes more attractive to exercise at lower stock prices
def test_boundary_shifts_with_div_yield():
    low_div_boundary = extract_boundary_curve('call', 100, 0.05, 0.05, 0.1, 1,
                                            100, 100)
    high_div_boundary = extract_boundary_curve('call', 100, 0.05, 0.1, 0.1, 1,
                                              100, 100)

    mean_low = np.mean(low_div_boundary)
    mean_high = np.mean(high_div_boundary)

    assert mean_high <= mean_low + 1e-6


# As volatility increases, the time value of the option increases, making early exercise less attractive.
# The exercise region therefore shrinks and the free boundary shifts upward,
# meaning the option is exercised only at higher stock prices.
def test_boundary_shifts_with_volatility():
    low_vol_boundary = extract_boundary_curve('call', 100, 0.05, 0.05, 0.1, 1,
                                              100, 100)
    high_vol_boundary = extract_boundary_curve('call', 100, 0.05, 0.05, 0.4, 1,
                                               100, 100)

    mean_low = np.mean(low_vol_boundary)
    mean_high = np.mean(high_vol_boundary)

    assert mean_high >= mean_low - 1e-6
















