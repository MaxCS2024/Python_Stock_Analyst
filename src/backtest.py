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


def build_portfolios(train_returns, risk_free_rate, max_weight, shrink=False):
    """Pick the weights using only the training data."""

    statistics = calculations.calculate_statistics(train_returns)
    _, annual_cov = calculations.calculate_covariance(train_returns, shrink)
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
        "Compounded Return": calculations.compounded_return(monthly_returns),
        "Actual Volatility": annual_vol,
        "Actual Sharpe": (annual_return - risk_free_rate) / annual_vol,
        "Total Return": growth.iloc[-1] - 1,
        "Max Drawdown": min(max_drawdown, 0.0),
    }


def run_backtest(returns, train_end, risk_free_rate=0.02, max_weight=1.0,
                 benchmark_returns=None, shrink=False):
    """
    Run the full backtest.

    returns:           monthly stock returns (from calculations.calculate_returns)
    train_end:         last date of the training period, e.g. "2024-12-31"
    benchmark_returns: optional monthly returns of an index, e.g. OMXS30
    shrink:            use Ledoit-Wolf shrinkage when picking the weights
    """
    train, test = split_returns(returns, train_end)
    portfolios = build_portfolios(train, risk_free_rate, max_weight, shrink)

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


# ---------------------------------------------------------------
# Rolling backtest: re-pick the weights every year
# ---------------------------------------------------------------

def rolling_backtest(returns, risk_free_rate=0.02, max_weight=1.0,
                     benchmark_returns=None, shrink=False, min_train_months=24):
    """
    A fairer test than a single split: every calendar year, pick new weights
    using all data up to the end of the previous year, then hold them for
    that year. This is how you would actually run the strategy, and it gives
    one result per year instead of a single number.

    Returns a dict with:
        years:   one row per year with each portfolio's return that year
        summary: performance over all the test years combined
        growth:  value of 100 kr over all the test years
        wins:    how many years Max Sharpe and Min Variance beat equal weight
    """
    years = sorted(set(returns.index.year))
    test_years = [
        y for y in years
        if (returns.index <= pd.Timestamp(f"{y - 1}-12-31")).sum() >= min_train_months
    ]
    if len(test_years) < 2:
        raise ValueError(
            f"The rolling test needs at least {min_train_months} months of training "
            "data plus two calendar years to test. Pick an earlier start date."
        )

    names = ["Max Sharpe", "Min Variance", "Equal Weight"]
    monthly = {name: [] for name in names}
    year_rows = []

    for year in test_years:
        train = returns[returns.index <= pd.Timestamp(f"{year - 1}-12-31")]
        test = returns[returns.index.year == year]
        portfolios = build_portfolios(train, risk_free_rate, max_weight, shrink)

        row = {"Year": str(year) if len(test) == 12 else f"{year} ({len(test)} mo)"}
        for name in names:
            r = portfolio_monthly_returns(test, portfolios[name]["weights"])
            monthly[name].append(r)
            row[name] = (1 + r).prod() - 1
        year_rows.append(row)

    monthly = {name: pd.concat(parts) for name, parts in monthly.items()}

    bench_name = None
    if benchmark_returns is not None:
        index = monthly["Max Sharpe"].index
        bench = benchmark_returns.reindex(index).dropna()
        if len(bench) == len(index):
            bench_name = benchmark_returns.name or "Benchmark"
            monthly[bench_name] = bench
            for row in year_rows:
                year = int(row["Year"][:4])
                b = bench[bench.index.year == year]
                row[bench_name] = (1 + b).prod() - 1

    year_table = pd.DataFrame(year_rows).set_index("Year")
    year_table["Best"] = year_table[names].idxmax(axis=1)

    summary = pd.DataFrame(
        {name: performance(r, risk_free_rate) for name, r in monthly.items()}
    ).T

    growth = (1 + pd.DataFrame(monthly)).cumprod() * 100
    start = returns.index[returns.index < growth.index[0]][-1]
    growth = pd.concat([pd.DataFrame([[100.0] * growth.shape[1]],
                                     columns=growth.columns, index=[start]), growth])

    wins = {
        name: int((year_table[name] > year_table["Equal Weight"]).sum())
        for name in ["Max Sharpe", "Min Variance"]
    }

    return {
        "years": year_table,
        "summary": summary,
        "growth": growth,
        "wins": wins,
        "n_years": len(year_table),
        "benchmark": bench_name,
    }


def print_rolling(rolling):
    """Print the rolling backtest."""

    print("\n" + "=" * 60)
    print("ROLLING BACKTEST (new weights every year)")
    print("=" * 60)

    table = rolling["years"].copy()
    for col in table.columns:
        if col != "Best":
            table[col] = table[col].map(lambda x: f"{x:.1%}")
    print(table.to_string())

    n = rolling["n_years"]
    for name, w in rolling["wins"].items():
        print(f"{name} beat equal weight in {w} of {n} years")

    summary = rolling["summary"]
    print("\nALL TEST YEARS COMBINED")
    print(pd.DataFrame({
        "Compounded per year": summary["Compounded Return"].map(lambda x: f"{x:.1%}"),
        "Volatility": summary["Actual Volatility"].map(lambda x: f"{x:.1%}"),
        "Sharpe": summary["Actual Sharpe"].map(lambda x: f"{x:.2f}"),
        "Worst drop": summary["Max Drawdown"].map(lambda x: f"{x:.1%}"),
    }).to_string())
