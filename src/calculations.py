import pandas as pd
import numpy as np


def calculate_returns(prices):
    """Calculate monthly percentage returns."""

    if prices.empty:
        raise ValueError("No stock price data available.")

    returns = prices.pct_change(fill_method=None)
    returns = returns.dropna(how="any")

    if returns.empty:
        raise ValueError("Not enough price history to calculate returns.")

    return returns


def calculate_statistics(returns):
    """Calculate monthly and annualized return and volatility."""

    results = pd.DataFrame({
        "Monthly Return": returns.mean(),
        "Annual Return": returns.mean() * 12,
        "Monthly Volatility": returns.std(),
        "Annual Volatility": returns.std() * np.sqrt(12)
    })

    return results


def calculate_covariance(returns):
    """Calculate monthly and annualized covariance."""

    covariance = returns.cov()
    annual_covariance = covariance * 12

    return covariance, annual_covariance


def calculate_correlation(returns):
    """Calculate correlation between stocks."""

    return returns.corr()


def calculate_all(prices):
    """Run all financial calculations."""

    returns = calculate_returns(prices)

    statistics = calculate_statistics(returns)

    covariance, annual_covariance = calculate_covariance(returns)

    correlation = calculate_correlation(returns)

    return {
        "returns": returns,
        "statistics": statistics,
        "covariance": covariance,
        "annual_covariance": annual_covariance,
        "correlation": correlation
    }