"""
Out-of-sample backtest.

The idea:
  1. TRAINING period: estimate returns and covariance, and pick the
     portfolio weights, using ONLY this data.
  2. TEST period: hold those weights and see what actually happened.

The optimizer never sees the test period, just like in real life
you never see the future when you choose your portfolio.

Weights are kept fixed in the test period (rebalanced back to the
target weights every month).
"""

import numpy as np
import pandas as pd

from src import calculations
from src import portfolio


def split_returns(returns, train_end):
    """Split monthly returns into a training part and a test part."""

    cutoff = pd.Timestamp(train_end)
    train = returns[returns.index <= cutoff]
    test = returns[returns.index > cutoff]

    if len(train) < 24:
        raise ValueError(
            f"Only {len(train)} training months. Use at least 24 (2 years)."
        )
    if len(test) < 6:
        raise ValueError(
            f"Only {len(test)} test months. Use at least 6."
        )

    return train, test


def build_portfolios(train_returns, risk_free_rate, max_weight):
    """Pick the weights using only the training data."""

    statistics = calculations.calculate_statistics(train_returns)
    _, annual_cov = calculations.calculate_covariance(train_returns)
    mean_returns = statistics["Annual Return"]

    return {
        "Max Sharpe": portfolio.max_sharpe_portfolio(
            mean_returns, annual_cov,
            risk_free_rate=risk_free_rate, max_weight=max_weight
        ),
        "Min Variance": portfolio.min_variance_portfolio(
            mean_returns, annual_cov,
            risk_free_rate=risk_free_rate, max_weight=max_weight
        ),
        "Equal Weight": portfolio.equal_weight_portfolio(
            mean_returns, annual_cov, risk_free_rate=risk_free_rate
        ),
    }


def portfolio_monthly_returns(test_returns, weights):
    """Each month's portfolio return = weighted sum of the stock returns."""
    return test_returns[weights.index] @ weights


def performance(monthly_returns, risk_free_rate):
    """
    Measure what actually happened.

    Annual return uses the same method as the prediction
    (average monthly return x 12) so the two are comparable.
    """
    growth = (1 + monthly_returns).cumprod()
    annual_return = monthly_returns.mean() * 12
    annual_vol = monthly_returns.std() * np.sqrt(12)

    # Max drawdown: the worst drop from a previous peak
    peaks = np.maximum(growth.cummax(), 1.0)
    max_drawdown = (growth / peaks - 1).min()

    return {
        "Actual Return": annual_return,
        "Actual Volatility": annual_vol,
        "Actual Sharpe": (annual_return - risk_free_rate) / annual_vol,
        "Total Return": growth.iloc[-1] - 1,
        "Max Drawdown": min(max_drawdown, 0.0),
    }


def run_backtest(returns, train_end, risk_free_rate=0.02, max_weight=1.0,
                 benchmark_returns=None):
    """
    Run the full backtest.

    returns:           monthly stock returns (from calculations.calculate_returns)
    train_end:         last date of the training period, e.g. "2024-12-31"
    benchmark_returns: optional monthly returns of an index, e.g. OMXS30
    """
    train, test = split_returns(returns, train_end)
    portfolios = build_portfolios(train, risk_free_rate, max_weight)

    rows = {}
    monthly = {}

    for name, p in portfolios.items():
        monthly[name] = portfolio_monthly_returns(test, p["weights"])
        row = {
            "Predicted Return": p["return"],
            "Predicted Volatility": p["volatility"],
            "Predicted Sharpe": p["sharpe"],
        }
        row.update(performance(monthly[name], risk_free_rate))
        rows[name] = row

    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(test.index).dropna()
        if len(bench) > 0:
            name = benchmark_returns.name or "Benchmark"
            monthly[name] = bench
            rows[name] = performance(bench, risk_free_rate)

    results = pd.DataFrame(rows).T

    # Value of 100 kr invested at the start of the test period
    growth = pd.DataFrame(monthly)
    growth = (1 + growth).cumprod() * 100
    start_row = pd.DataFrame(
        [[100.0] * growth.shape[1]],
        columns=growth.columns,
        index=[train.index[-1]]
    )
    growth = pd.concat([start_row, growth])

    weights = pd.DataFrame({n: p["weights"] for n, p in portfolios.items()})

    return {
        "results": results,
        "growth": growth,
        "weights": weights,
        "train_period": (train.index[0], train.index[-1]),
        "test_period": (test.index[0], test.index[-1]),
        "test_months": len(test),
    }


def print_backtest(backtest):
    """Print the backtest in a readable way."""

    (train_start, train_stop) = backtest["train_period"]
    (test_start, test_stop) = backtest["test_period"]

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE BACKTEST")
    print("=" * 60)
    print(f"Training: {train_start:%Y-%m} to {train_stop:%Y-%m}  (weights chosen here)")
    print(f"Test:     {test_start:%Y-%m} to {test_stop:%Y-%m}  "
          f"({backtest['test_months']} months, weights held fixed)")

    print("\nWEIGHTS CHOSEN FROM TRAINING DATA")
    weights = backtest["weights"]
    weights = weights[(weights.abs() > 0.001).any(axis=1)]
    print((weights * 100).round(1).astype(str).add("%").to_string())

    results = backtest["results"]
    percent_columns = [
        "Predicted Return", "Actual Return",
        "Predicted Volatility", "Actual Volatility",
        "Total Return", "Max Drawdown",
    ]
    ratio_columns = ["Predicted Sharpe", "Actual Sharpe"]

    table = pd.DataFrame(index=results.index)
    for col in percent_columns:
        if col in results:
            table[col] = results[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.1%}"
            )
    for col in ratio_columns:
        if col in results:
            table[col] = results[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.2f}"
            )

    print("\nPREDICTED (from training) vs ACTUAL (in test period)")
    print(table[["Predicted Return", "Actual Return",
                 "Predicted Volatility", "Actual Volatility"]].to_string())

    print("\nTEST PERIOD RESULTS")
    print(table[["Total Return", "Max Drawdown",
                 "Predicted Sharpe", "Actual Sharpe"]].to_string())