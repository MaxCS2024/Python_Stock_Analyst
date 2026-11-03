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


def compounded_return(returns):
    """
    What the money actually grew by per year (also called CAGR).

    For volatile stocks the average return x 12 overstates this, because losses
    hurt more than gains help: +50% then -50% averages 0%, but 100 kr becomes 75 kr.
    """
    growth = (1 + returns).prod()
    years = len(returns) / 12
    return growth ** (1 / years) - 1


def calculate_statistics(returns):
    """Calculate monthly and annualized return and volatility."""

    results = pd.DataFrame({
        "Monthly Return": returns.mean(),
        "Annual Return": returns.mean() * 12,
        "Compounded Return": compounded_return(returns),
        "Monthly Volatility": returns.std(),
        "Annual Volatility": returns.std() * np.sqrt(12)
    })

    return results


def ledoit_wolf(returns):
    """
    Ledoit-Wolf shrinkage of the covariance matrix.

    With few months of data and many stocks, the normal covariance estimate
    contains a lot of noise, and the optimizer reacts to that noise by putting
    too much money in a few stocks. Ledoit-Wolf blends the normal estimate
    with a much simpler one (every stock has the average variance and no
    correlation), and calculates from the data how much to blend:

        shrunk = (1 - intensity) x normal estimate + intensity x simple estimate

    intensity 0 means "trust the data completely", 1 means "ignore it".

    Returns (monthly covariance as a DataFrame, intensity between 0 and 1).
    This is the same formula as scikit-learn's LedoitWolf.
    """
    X = returns.to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    n, p = X.shape

    sample = X.T @ X / n
    mu = np.trace(sample) / p                      # average variance

    X2 = X ** 2
    beta_ = np.sum(X2.T @ X2)
    delta_ = np.sum(sample ** 2)
    beta = (beta_ / n - delta_) / (p * n)          # how noisy the sample is
    delta = (delta_ - 2 * mu * np.trace(sample) + p * mu ** 2) / p
    beta = min(beta, delta)
    intensity = 0.0 if delta == 0 else beta / delta

    shrunk = (1 - intensity) * sample + intensity * mu * np.eye(p)
    cov = pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
    return cov, float(intensity)


def calculate_covariance(returns, shrink=False):
    """
    Calculate monthly and annualized covariance.

    shrink=True uses Ledoit-Wolf shrinkage for steadier estimates.
    """
    if shrink:
        covariance, _ = ledoit_wolf(returns)
    else:
        covariance = returns.cov()

    annual_covariance = covariance * 12
    return covariance, annual_covariance


def calculate_correlation(returns):
    """Calculate correlation between stocks."""

    return returns.corr()


def calculate_all(prices, shrink=False):
    """Run all financial calculations."""

    returns = calculate_returns(prices)

    statistics = calculate_statistics(returns)

    covariance, annual_covariance = calculate_covariance(returns, shrink)

    correlation = calculate_correlation(returns)

    return {
        "returns": returns,
        "statistics": statistics,
        "covariance": covariance,
        "annual_covariance": annual_covariance,
        "correlation": correlation,
        "shrinkage": ledoit_wolf(returns)[1] if shrink else 0.0,
    }
