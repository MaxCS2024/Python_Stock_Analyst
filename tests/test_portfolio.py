"""
Checks portfolio.py against the 2-stock example worked out by hand:
  returns 10% and 15%, volatilities 20% and 30%, correlation 0.3
  -> 12% target gives 60% A / 40% B and about 19.35% volatility.

Run from the project folder:  python tests/test_portfolio.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import portfolio


mean_returns = pd.Series([0.10, 0.15], index=["A", "B"])
covariance = pd.DataFrame(
    [[0.04, 0.018],
     [0.018, 0.09]],
    index=["A", "B"], columns=["A", "B"]
)


def test_hand_calculated_example():
    p = portfolio.min_variance_portfolio(mean_returns, covariance, target_return=0.12)
    assert np.isclose(p["weights"]["A"], 0.6, atol=1e-4)
    assert np.isclose(p["weights"]["B"], 0.4, atol=1e-4)
    assert np.isclose(p["volatility"] ** 2, 0.03744, atol=1e-5)
    assert np.isclose(p["volatility"], 0.1935, atol=1e-4)


def test_weights_sum_to_one():
    for p in [
        portfolio.min_variance_portfolio(mean_returns, covariance),
        portfolio.max_sharpe_portfolio(mean_returns, covariance),
    ]:
        assert np.isclose(p["weights"].sum(), 1.0)
        assert (p["weights"] >= 0).all()


def test_unreachable_target_raises():
    try:
        portfolio.min_variance_portfolio(mean_returns, covariance, target_return=0.20)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for a 20% target")

def test_max_weight_is_respected():
    # 4 stocks where one clearly dominates: without a cap it would get most of the money
    mu = pd.Series([0.30, 0.08, 0.07, 0.06], index=["W", "X", "Y", "Z"])
    cov = pd.DataFrame(np.diag([0.04, 0.04, 0.04, 0.04]), index=mu.index, columns=mu.index)

    uncapped = portfolio.max_sharpe_portfolio(mu, cov, risk_free_rate=0.02)
    capped = portfolio.max_sharpe_portfolio(mu, cov, risk_free_rate=0.02, max_weight=0.30)

    assert uncapped["weights"]["W"] > 0.30
    assert capped["weights"].max() <= 0.30 + 1e-6
    assert np.isclose(capped["weights"].sum(), 1.0)


def test_max_weight_too_low_raises():
    try:
        portfolio.min_variance_portfolio(mean_returns, covariance, max_weight=0.4)
    except ValueError:
        return
    raise AssertionError("Expected ValueError: 2 stocks x 40% can't reach 100%")


if __name__ == "__main__":
    test_hand_calculated_example()
    test_weights_sum_to_one()
    test_unreachable_target_raises()
    test_max_weight_is_respected()
    test_max_weight_too_low_raises()
    p = portfolio.min_variance_portfolio(mean_returns, covariance, target_return=0.12)
    portfolio.print_portfolio("Hand-calculated example (12% target)", p)
    print("\nAll tests passed!")