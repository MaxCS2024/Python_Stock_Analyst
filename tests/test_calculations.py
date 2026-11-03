"""
Checks compounded returns, Ledoit-Wolf shrinkage and the rolling backtest.

Run from the project folder:  python tests/test_calculations.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import backtest, calculations


def test_compounded_return_by_hand():
    # +1% every month for a year: 1.01^12 - 1 = 12.68%
    r = pd.Series([0.01] * 12)
    assert np.isclose(calculations.compounded_return(r), 1.01 ** 12 - 1)

    # +50% then -50%: the average is 0%, but 100 kr becomes 75 kr in 2 months
    r = pd.Series([0.5, -0.5] * 6)
    assert np.isclose(calculations.compounded_return(r), 0.75 ** 6 - 1)


def make_returns(months=60, stocks=6, seed=1):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.008, 0.04, (months, 1))
    data = market + rng.normal(0, 0.05, (months, stocks))
    index = pd.date_range("2021-01-01", periods=months, freq="MS")
    return pd.DataFrame(data, index=index, columns=[f"S{i}" for i in range(stocks)])


def test_ledoit_wolf_is_a_blend():
    returns = make_returns()
    shrunk, intensity = calculations.ledoit_wolf(returns)
    assert 0 <= intensity <= 1
    assert np.allclose(shrunk, shrunk.T)                     # still symmetric
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)            # still a valid covariance

    # It is exactly the blend described in the docstring
    X = returns - returns.mean()
    sample = X.T @ X / len(X)
    target = np.trace(sample) / len(sample) * np.eye(len(sample))
    assert np.allclose(shrunk, (1 - intensity) * sample + intensity * target)


def test_shrink_switch():
    returns = make_returns()
    _, normal = calculations.calculate_covariance(returns, shrink=False)
    _, shrunk = calculations.calculate_covariance(returns, shrink=True)
    assert np.allclose(normal, returns.cov() * 12)
    assert not np.allclose(normal, shrunk)


def test_rolling_backtest_years():
    returns = make_returns(months=60)                        # Jan 2021 to Dec 2025
    rolling = backtest.rolling_backtest(returns, risk_free_rate=0.02, max_weight=0.4)
    # 24 months of training first, so 2023, 2024 and 2025 are tested
    assert list(rolling["years"].index) == ["2023", "2024", "2025"]
    assert rolling["n_years"] == 3
    assert all(0 <= w <= 3 for w in rolling["wins"].values())
    # 100 kr at the start of the test, 36 months later at the end
    assert rolling["growth"].iloc[0].eq(100).all()
    assert len(rolling["growth"]) == 37


if __name__ == "__main__":
    test_compounded_return_by_hand()
    test_ledoit_wolf_is_a_blend()
    test_shrink_switch()
    test_rolling_backtest_years()
    print("All calculation tests passed!")
