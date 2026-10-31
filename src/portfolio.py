"""
Markowitz portfolio optimization.

All inputs are ANNUAL figures from calculations.py:
    mean_returns: pd.Series of expected annual returns, one per stock
    covariance:   pd.DataFrame annual covariance matrix

The two core formulas are the same as in the 2-stock example:
    portfolio return   = w · mu
    portfolio variance = wᵀ Σ w
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------

def portfolio_return(weights, mean_returns):
    """Expected return: weighted average of the stocks' returns."""
    return float(np.dot(weights, mean_returns))


def portfolio_volatility(weights, covariance):
    """Volatility: square root of wᵀ Σ w."""
    return float(np.sqrt(weights @ covariance @ weights))


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _prepare(mean_returns, covariance):
    """Make sure returns and covariance are in the same stock order."""
    tickers = mean_returns.index
    mu = mean_returns.to_numpy(dtype=float)
    cov = covariance.loc[tickers, tickers].to_numpy(dtype=float)
    return tickers, mu, cov


def _check_max_weight(n_assets, max_weight):
    """With N stocks, a cap below 1/N makes it impossible to invest 100%."""
    if max_weight * n_assets < 1 - 1e-9:
        raise ValueError(
            f"max_weight {max_weight:.0%} is too low for {n_assets} stocks. "
            f"It must be at least {1 / n_assets:.1%}."
        )


def _return_range(mu, max_weight):
    """
    Lowest and highest return reachable without short selling
    when no stock may exceed max_weight.

    Highest: fill the best stocks up to the cap, one after another.
    Lowest:  same thing with the worst stocks.
    """
    def fill(sorted_returns):
        total, remaining = 0.0, 1.0
        for r in sorted_returns:
            w = min(max_weight, remaining)
            total += w * r
            remaining -= w
            if remaining <= 1e-12:
                break
        return total

    return fill(np.sort(mu)), fill(np.sort(mu)[::-1])


def _summarize(weights, tickers, mu, cov, risk_free_rate):
    """Package the result in an easy-to-read dictionary."""
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)
    return {
        "weights": pd.Series(weights, index=tickers),
        "return": ret,
        "volatility": vol,
        "sharpe": (ret - risk_free_rate) / vol,
    }


def _optimize(objective, n_assets, extra_constraints, allow_short,
              max_weight=1.0):
    """
    Run the optimizer.

    Always enforced: weights sum to 1 (fully invested).
    Every weight is at most max_weight.
    If allow_short is False: every weight is also at least 0.
    """
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    constraints += extra_constraints

    lower = None if allow_short else 0.0
    bounds = [(lower, max_weight)] * n_assets
    start = np.full(n_assets, 1.0 / n_assets)  # start from equal weights

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    weights = result.x
    if not allow_short:
        # Remove tiny negative values like -1e-17 caused by rounding
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()

    return weights


# ---------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------

def min_variance_portfolio(mean_returns, covariance, target_return=None,
                           allow_short=False, risk_free_rate=0.0,
                           max_weight=1.0):
    """
    Lowest-risk portfolio.

    target_return=None -> the global minimum variance portfolio
    target_return=0.12 -> lowest-risk portfolio that earns 12%
    max_weight=0.25    -> no stock may be more than 25% of the portfolio
    """
    tickers, mu, cov = _prepare(mean_returns, covariance)
    _check_max_weight(len(mu), max_weight)

    extra = []
    if target_return is not None:
        if not allow_short:
            low, high = _return_range(mu, max_weight)
            if not (low - 1e-9 <= target_return <= high + 1e-9):
                raise ValueError(
                    f"Target return {target_return:.2%} is not reachable with a "
                    f"{max_weight:.0%} cap and no short selling. "
                    f"Choose between {low:.2%} and {high:.2%}."
                )
        extra.append({
            "type": "eq",
            "fun": lambda w: portfolio_return(w, mu) - target_return,
        })

    def variance(w):
        return w @ cov @ w

    weights = _optimize(variance, len(mu), extra, allow_short, max_weight)
    return _summarize(weights, tickers, mu, cov, risk_free_rate)


def max_sharpe_portfolio(mean_returns, covariance, risk_free_rate=0.02,
                         allow_short=False, max_weight=1.0):
    """
    Portfolio with the best return per unit of risk
    (the "tangency portfolio").

    Sharpe ratio = (return - risk-free rate) / volatility
    """
    tickers, mu, cov = _prepare(mean_returns, covariance)
    _check_max_weight(len(mu), max_weight)

    def negative_sharpe(w):
        # The optimizer minimizes, so we minimize the NEGATIVE Sharpe ratio
        return -(portfolio_return(w, mu) - risk_free_rate) / portfolio_volatility(w, cov)

    weights = _optimize(negative_sharpe, len(mu), [], allow_short, max_weight)
    return _summarize(weights, tickers, mu, cov, risk_free_rate)


def equal_weight_portfolio(mean_returns, covariance, risk_free_rate=0.0):
    """Simple 1/N benchmark to compare the optimized portfolios against."""
    tickers, mu, cov = _prepare(mean_returns, covariance)
    weights = np.full(len(mu), 1.0 / len(mu))
    return _summarize(weights, tickers, mu, cov, risk_free_rate)


def efficient_frontier(mean_returns, covariance, n_points=50,
                       allow_short=False, max_weight=1.0):
    """
    Trace the efficient frontier: for a range of target returns,
    find the lowest-risk portfolio for each one.

    Returns a DataFrame with one row per point:
    Return, Volatility, and one weight column per stock.
    """
    tickers, mu, cov = _prepare(mean_returns, covariance)

    # The frontier starts at the minimum variance portfolio and goes up
    # to the highest return reachable under the weight cap.
    start = min_variance_portfolio(mean_returns, covariance,
                                   allow_short=allow_short,
                                   max_weight=max_weight)["return"]
    end = mu.max() if allow_short else _return_range(mu, max_weight)[1]
    targets = np.linspace(start, end, n_points)

    rows = []
    for target in targets:
        try:
            p = min_variance_portfolio(mean_returns, covariance,
                                       target_return=target,
                                       allow_short=allow_short,
                                       max_weight=max_weight)
        except ValueError:
            continue  # skip points the optimizer can't solve

        row = {"Return": p["return"], "Volatility": p["volatility"]}
        row.update(p["weights"].to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def print_portfolio(name, portfolio, min_weight=0.001):
    """Pretty-print a portfolio, hiding weights that are basically zero."""
    print(f"\n{name}")
    print(f"  Expected return: {portfolio['return']:.2%}")
    print(f"  Volatility:      {portfolio['volatility']:.2%}")
    print(f"  Sharpe ratio:    {portfolio['sharpe']:.2f}")
    print("  Weights:")

    weights = portfolio["weights"].sort_values(ascending=False)
    for ticker, weight in weights.items():
        if abs(weight) >= min_weight:
            print(f"    {ticker:<12} {weight:>7.1%}")