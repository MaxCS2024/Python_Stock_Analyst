"""
Every calculation in one go.

The PDF report and the Excel export both need all the results, so they get
them from run() and always agree with each other and with main.py.
"""

import math

from src import backtest, black_litterman, calculations, portfolio, settings


def run(prices, config, benchmark_returns=None, market_caps=None):
    """
    prices:            monthly prices in SEK, one column per stock (tickers or names)
    config:            settings dict (see src/settings.py)
    benchmark_returns: optional monthly benchmark returns
    market_caps:       {ticker: market value in SEK}, needed for Black-Litterman

    Returns a dict with everything: statistics, expected returns, portfolios,
    frontier, backtest and rolling backtest. Results that can't be calculated
    (for example too little data for a backtest) are None, with a note saying why.
    """
    prices = prices.rename(columns=settings.display_name)
    shrink = bool(config.get("shrinkage", False))
    cost = float(config.get("trading_cost", 0.0))
    model = config.get("return_model", "historical")

    analysis = calculations.calculate_all(prices, shrink=shrink)
    returns = analysis["returns"]
    stats = analysis["statistics"]
    annual_cov = analysis["annual_covariance"]
    names = list(stats.index)

    rf = config["risk_free_rate"]
    target_return = config["target_return"]
    max_weight = config["max_weight"]
    if max_weight * len(names) < 1:
        max_weight = math.ceil(100 / len(names)) / 100

    # --- Expected returns: historical average or Black-Litterman ---
    market_weights = implied = None
    return_model = None
    if model == "black_litterman":
        caps = {settings.display_name(t): v for t, v in (market_caps or {}).items()}
        views = {settings.display_name(t): v for t, v in config.get("views", {}).items()}
        market_weights = black_litterman.market_weights(caps, names)
        implied = black_litterman.implied_returns(annual_cov, market_weights, rf)
        mean_returns = black_litterman.black_litterman(
            annual_cov, market_weights, rf, views, config.get("view_confidence", 0.5))
        return_model = black_litterman.make_return_model(caps, rf)
    else:
        mean_returns = stats["Annual Return"]

    # --- Portfolios ---
    max_sharpe = portfolio.max_sharpe_portfolio(
        mean_returns, annual_cov, risk_free_rate=rf, max_weight=max_weight)
    min_var = portfolio.min_variance_portfolio(
        mean_returns, annual_cov, risk_free_rate=rf, max_weight=max_weight)
    equal = portfolio.equal_weight_portfolio(mean_returns, annual_cov, risk_free_rate=rf)
    portfolios = {"Max Sharpe": max_sharpe, "Min Variance": min_var, "Equal Weight": equal}
    frontier = portfolio.efficient_frontier(mean_returns, annual_cov, max_weight=max_weight)

    try:
        target = portfolio.min_variance_portfolio(
            mean_returns, annual_cov, target_return=target_return,
            risk_free_rate=rf, max_weight=max_weight)
        target_note = None
    except ValueError as error:
        target, target_note = None, str(error)

    # --- Backtests ---
    common = dict(risk_free_rate=rf, max_weight=max_weight,
                  benchmark_returns=benchmark_returns, shrink=shrink,
                  cost=cost, return_model=return_model)
    try:
        test = backtest.run_backtest(returns, train_end=config["train_end"], **common)
        test_note = None
    except ValueError as error:
        test, test_note = None, str(error)

    try:
        rolling = backtest.rolling_backtest(returns, **common)
        rolling_note = None
    except ValueError as error:
        rolling, rolling_note = None, str(error)

    return {
        "prices": prices,
        "returns": returns,
        "statistics": stats,
        "correlation": analysis["correlation"],
        "annual_covariance": annual_cov,
        "shrinkage": analysis["shrinkage"],
        "names": names,
        "risk_free_rate": rf,
        "max_weight": max_weight,
        "target_return": target_return,
        "trading_cost": cost,
        "return_model": model,
        "expected_returns": mean_returns,
        "market_weights": market_weights,
        "implied_returns": implied,
        "portfolios": portfolios,
        "frontier": frontier,
        "target": target,
        "target_note": target_note,
        "test": test,
        "test_note": test_note,
        "rolling": rolling,
        "rolling_note": rolling_note,
    }
