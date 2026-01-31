import numpy as np


def european_binomial_price(kind, S, K, r, q, sigma, T, steps):
    """
        Calculates the price of a European option using a two-step binomial tree.

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
    dt = T/steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    # risk neutral probability discounted by dividend yield
    p = (np.exp((r-q)*dt) - d) / (u - d)
    q_prob = 1 - p
    disc = np.exp(-r * dt)

    # Calculate the stock prices at the last node/maturity
    asset_prices = S * d ** np.arange(steps, -1, -1) * u ** (np.arange(0, steps+1, 1))

    # calculate option values at each node of the tree
    c_terminal_payoff = np.maximum(asset_prices - K, np.zeros(steps+1))
    p_terminal_payoff = np.maximum(K-asset_prices, np.zeros(steps+1))

    # Work backwards through tree from the final price of the option to its present time stopping at i=1
    if kind == 'call':
        payoff = c_terminal_payoff
    else:
        payoff = p_terminal_payoff

    for i in range(steps, 0, -1):
        right_children = payoff[1:i + 1]
        left_children = payoff[0:i]

        # Look at all up node children [1:i+1] and all down node children [0:i] for each timestep - triangular matrix
        # Replace payoff vector with the shrunken version

        payoff = disc * (p * right_children + q_prob * left_children)

    # return the root node which is the option price today
    return payoff[0]


