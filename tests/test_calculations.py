"""
Checks compounded returns, Ledoit-Wolf shrinkage, the rolling backtest,
trading costs and Black-Litterman.

Run from the project folder:  python tests/test_calculations.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import backtest, black_litterman, calculations


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


def test_trading_costs_by_hand():
    dates = pd.date_range("2025-01-01", periods=2, freq="MS")
    r = pd.DataFrame({"A": [0.10, 0.0], "B": [-0.10, 0.0]}, index=dates)
    w = pd.Series({"A": 0.5, "B": 0.5})
    net, held, fees = backtest.simulate(r, w, cost=0.001)
    # Month 1: buy everything from cash, so 100% is traded and the fee is 0.1%
    assert np.isclose(fees.iloc[0], 0.001)
    assert np.isclose(net.iloc[0], -0.001)
    # After month 1, A is 55% and B 45%; back to 50/50 trades 10%, fee 0.01%
    assert np.isclose(fees.iloc[1], 0.0001)
    # Without costs nothing changes compared with the old calculation
    net0, _, _ = backtest.simulate(r, w, cost=0.0)
    assert np.allclose(net0, backtest.portfolio_monthly_returns(r, w))


def test_costs_lower_backtest_results():
    returns = make_returns(months=60)
    free = backtest.rolling_backtest(returns, max_weight=0.4, cost=0.0)
    costly = backtest.rolling_backtest(returns, max_weight=0.4, cost=0.01)
    for name in ["Max Sharpe", "Min Variance", "Equal Weight"]:
        assert costly["summary"].loc[name, "Compounded Return"] < \
            free["summary"].loc[name, "Compounded Return"]
        assert costly["summary"].loc[name, "Costs Paid"] > 0


def make_cov():
    names = ["Big", "Mid", "Small"]
    vol = np.array([0.18, 0.22, 0.30])
    corr = np.array([[1, .5, .3], [.5, 1, .4], [.3, .4, 1]])
    return pd.DataFrame(np.outer(vol, vol) * corr, index=names, columns=names)


def test_black_litterman_without_views_gives_market_portfolio():
    cov = make_cov()
    w = black_litterman.market_weights({"Big": 600, "Mid": 300, "Small": 100}, list(cov.index))
    implied = black_litterman.black_litterman(cov, w, risk_free_rate=0.02)
    # The best portfolio for the implied returns is exactly the market itself
    best = np.linalg.solve(cov.values, (implied - 0.02).values)
    assert np.allclose(best / best.sum(), [0.6, 0.3, 0.1])


def test_black_litterman_views_move_returns():
    cov = make_cov()
    w = black_litterman.market_weights({"Big": 600, "Mid": 300, "Small": 100}, list(cov.index))
    implied = black_litterman.implied_returns(cov, w, 0.02)["Small"]
    low = black_litterman.black_litterman(cov, w, 0.02, {"Small": 0.20}, confidence=0.25)["Small"]
    high = black_litterman.black_litterman(cov, w, 0.02, {"Small": 0.20}, confidence=0.99)["Small"]
    # More confidence moves the expected return closer to the view
    assert implied < low < high < 0.20 + 1e-9
    assert abs(high - 0.20) < 0.005


def test_market_weights_fallbacks():
    names = ["A", "B", "C"]
    assert np.allclose(black_litterman.market_weights({}, names), 1 / 3)
    w = black_litterman.market_weights({"A": 100, "B": 300}, names)    # C unknown
    assert np.isclose(w.sum(), 1) and w["C"] > 0


if __name__ == "__main__":
    test_compounded_return_by_hand()
    test_ledoit_wolf_is_a_blend()
    test_shrink_switch()
    test_rolling_backtest_years()
    test_trading_costs_by_hand()
    test_costs_lower_backtest_results()
    test_black_litterman_without_views_gives_market_portfolio()
    test_black_litterman_views_move_returns()
    test_market_weights_fallbacks()
    print("All calculation tests passed!")
