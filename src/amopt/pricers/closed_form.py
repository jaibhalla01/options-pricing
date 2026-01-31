import numpy as np
from scipy import stats

# Take into account dividend yield discounting option price read about this


def euro_vanilla_price(kind, S, K, r, T, sigma, q):
    # Accounting for edge cases where the option is at expiry or there's no uncertainty otherwise the BS-formula breaks
    if T <= 0:
        return max(S - K, 0) if kind == "call" else max(K - S, 0)

    d1 = 1/(sigma * np.sqrt(T)) * (np.log(S/K) + (r - q + sigma**2/2) * T)
    d2 = d1 - sigma * np.sqrt(T)

    call = S * np.exp(-q*T) * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
    put = K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * np.exp(-q*T) * stats.norm.cdf(-d1)

    if kind == 'call':
        return call
    elif kind == 'put':
        return put
    else:
        raise ValueError("kind must be 'call' or 'put'")



