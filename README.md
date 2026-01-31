# American Option Pricing (Finite Difference & Binomial)

A Python library for pricing European and American options using:

- Crank–Nicolson finite-difference methods
- Penalty and LCP formulations for early exercise
- Cox–Ross–Rubinstein binomial trees
- Free-boundary extraction and visualisation

The project focuses on **numerical stability, correctness, and validation**
against closed-form and independent numerical baselines.

---

## Features

- European option pricing:
  - Black–Scholes closed form
  - Finite difference (CN)
  - Binomial tree
- American option pricing:
  - Finite difference + penalty method
  - Binomial early-exercise model
- Free-boundary extraction for American options
- Extensive test suite validating:
  - Convergence
  - No-arbitrage conditions
  - Monotonicity
  - Cross-method consistency

---

## Project Structure

src/amopt/
├─ dataclasses/        # Model & grid containers
├─ uniform/            # Low-level PDE operators and numerical building blocks
├─ lcp/                # Penalty and PSOR solvers (early exercise)
├─ pricers/            # Core pricing algorithms (FD time-stepping, penalty/LCP logic, binomial trees)
├─ tests/              # Validation & convergence tests
├─ notebooks/          # Free-boundary & diagnostic visualisations
---

## Installation

```bash
git clone https://github.com/jaibhalla01/options-pricing.git
cd american-options
pip install -e .

## Quick Start

Price an American option using the finite-difference penalty method:

Returns an option price at spot price (S) and time (t=0)

```python
from amopt.pricers.american_fd import american_fd_pricer

price = american_fd_pricer(
    option_type="call",
    S=100,
    K=100,
    r=0.05,
    q=0.02,
    sigma=0.2,
    T=1.0,
    stock_intervals=200,
    time_intervals=200
)

print(price)