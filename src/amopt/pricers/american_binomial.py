import numpy as np


# For a non-dividend paying option it is never optimal to exercise the option early because the continuation
# value is usually worth more
def american_binomial_price(kind, S, K, r, q, sigma, T, steps):
    """
            Calculates the price of an American option using a two-step binomial tree.

            Parameters:
                kind (string): specifies type of option
                S (float): initial stock price
                K (float): strike price
                r (float): risk-free interest rate
                q (float): dividend yield
                sigma (float): stock price volatility
                T (float): time to maturity (in years)
                steps (int): number of time steps in the binomial tree

            Returns:
                (float) price of the put/call option
            """
    steps = int(steps)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    a = np.exp((r - q) * dt)

    # risk neutral probability discounted by dividend yield
    p = (a - d) / (u - d)

    q_prob = 1 - p
    disc = np.exp(-r * dt)

    # Calculate stock prices up until last node/maturity when t=T
    terminal_asset_prices = S * d ** np.arange(steps, -1, -1) * u ** np.arange(0, steps + 1)

    # calculate option values at each node of the tree
    c_maturity = np.maximum(terminal_asset_prices - K, 0)
    p_maturity = np.maximum(K - terminal_asset_prices, 0)

    # At maturity (t=T) the option's value should equal the payoff vector
    if kind == 'call':
        option_values_next = c_maturity
    else:
        option_values_next = p_maturity

    # Work backwards through tree from the final price of the option to its present time stopping at i=0
    # i refers to the timestep
    for i in range(steps, 0, -1):
        option_values_now = np.zeros(i)
        # j refers to the number of times the stock price moves up
        for j in range(0, i, 1):
            # Calculate intrinsic (exercise) value at each node (i, j)
            asset_price = S * u ** j * d ** (i - j)

            # Calculate payoff of option at time j - exercise value
            if kind == 'call':
                exercise = np.maximum(asset_price - K, 0)
            else:
                exercise = np.maximum(K - asset_price, 0)

            # Calculate the value of holding the option
            continuation = disc * (p * option_values_next[j + 1] + q_prob * option_values_next[j])

            # Compare the intrinsic value of the option against the discounted expected value of its two children
            option_values_now[j] = np.maximum(exercise, continuation)

        # Shrink the option_values to be of size t=i-1
        option_values_next = option_values_now

    return option_values_next[0]
