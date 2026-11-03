"""
Black-Litterman expected returns.

The weakest input to Markowitz is the expected return of each stock. Using
the historical average means betting that the best stocks of the past few
years stay the best, which is why the optimizer piles into last year's
winners.

Black-Litterman starts somewhere more neutral. In the market as a whole,
big companies get most of the money. If investors are being sensible, the
market's own weights must be a good portfolio, and that tells us what
returns investors must be expecting. These are the market-implied returns:

    implied excess return = risk aversion x covariance x market weights

Then you can add your own views ("I expect Volvo to return 12% a year"),
and the model blends them with the market-implied returns. The more
confident you are, the more your view counts. Without views, the result is
simply the market-implied returns.

Two standard choices are fixed here:
    DELTA (risk aversion) = 2.5: how much extra return investors demand per
        unit of variance. 2.5 is the usual textbook value.
    TAU = 0.05: how uncertain the market-implied returns themselves are.
"""

import numpy as np
import pandas as pd

DELTA = 2.5
TAU = 0.05


def market_weights(market_caps, names):
    """
    Each stock's share of the total market value of the stocks.
    Stocks without a known market value get the median of the others; if
    no values are known at all, every stock gets the same weight.
    """
    caps = pd.Series(market_caps, dtype=float).reindex(names)
    caps = caps.where(caps > 0)
    if caps.isna().all():
        return pd.Series(1 / len(names), index=names)
    caps = caps.fillna(caps.median())
    return caps / caps.sum()


def implied_returns(annual_cov, weights, risk_free_rate, delta=DELTA):
    """The returns that make the market weights the sensible portfolio."""
    excess = delta * annual_cov.to_numpy() @ weights.reindex(annual_cov.index).to_numpy()
    return pd.Series(risk_free_rate + excess, index=annual_cov.index)


def black_litterman(annual_cov, weights, risk_free_rate, views=None,
                    confidence=0.5, delta=DELTA, tau=TAU):
    """
    Expected returns per year after blending in your views.

    annual_cov:  annual covariance matrix (stocks as index and columns)
    weights:     market weights (from market_weights)
    views:       {stock: expected return per year}, e.g. {"Volvo": 0.12}
    confidence:  0 to 1. At 0.5 a view counts about as much as the market's
                 own estimate; near 1 the result moves almost all the way
                 to your view.
    """
    names = list(annual_cov.index)
    sigma = annual_cov.to_numpy()
    pi = implied_returns(annual_cov, weights, 0.0, delta).to_numpy()   # excess

    views = {k: v for k, v in (views or {}).items()
             if k in names and v is not None and not pd.isna(v)}
    if not views:
        return pd.Series(risk_free_rate + pi, index=names)

    # One row per view, picking out the stock the view is about
    P = np.zeros((len(views), len(names)))
    for row, stock in enumerate(views):
        P[row, names.index(stock)] = 1.0
    Q = np.array([v - risk_free_rate for v in views.values()])          # excess

    c = min(max(confidence, 0.01), 0.99)
    tau_sigma = tau * sigma
    omega = np.diag(np.diag(P @ tau_sigma @ P.T) * (1 - c) / c)       # view uncertainty

    inv_ts = np.linalg.inv(tau_sigma)
    inv_om = np.linalg.inv(omega)
    combined = np.linalg.solve(inv_ts + P.T @ inv_om @ P, inv_ts @ pi + P.T @ inv_om @ Q)
    return pd.Series(risk_free_rate + combined, index=names)


def make_return_model(market_caps, risk_free_rate):
    """
    A function the backtests can call with each training period:
    (training returns, covariance) -> market-implied returns.
    Views are left out of the backtests on purpose: they're today's
    opinions, and using them on the past would be seeing the future.
    """
    def model(train_returns, annual_cov):
        weights = market_weights(market_caps, list(annual_cov.index))
        return implied_returns(annual_cov, weights, risk_free_rate)
    return model
