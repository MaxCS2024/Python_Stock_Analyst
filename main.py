import yfinance as yf
import pandas as pd

from src import calculations
from src import portfolio
from src import backtest
from src import settings
from src import visualizations


# Settings come from settings.json (edit them with the dashboard's
# "Save as default" button, or by editing the file directly)

config = settings.load()

tickers = config["tickers"]
START_DATE = config["start_date"]
END_DATE = settings.end_date()

RISK_FREE_RATE = config["risk_free_rate"]
TARGET_RETURN = config["target_return"]
MAX_WEIGHT = config["max_weight"]

TRAIN_END = config["train_end"]
BENCHMARK = config["benchmark"]

print(f"Using {len(tickers)} stocks: "
      + ", ".join(settings.display_name(t) for t in tickers))


# Step 1: Download stock data
data = yf.download(
    tickers,
    start=START_DATE,
    end=END_DATE,
    interval="1mo",
    auto_adjust=True,
    progress=False,
    threads=False
)

if data.empty:
    raise ValueError("Yahoo Finance returned no data.")

prices = data["Close"].dropna(axis=1, how="all").sort_index()

if prices.empty:
    raise ValueError("No valid stock prices were downloaded.")


# Step 2: Run financial calculations

analysis = calculations.calculate_all(prices)


# Step 3: Print results

print("\nSTOCK ANALYSIS")
print(analysis["statistics"].round(4))

print("\nANNUAL COVARIANCE")
print(analysis["annual_covariance"].round(4))

print("\nCORRELATION")
print(analysis["correlation"].round(4))


# Step 4: Portfolio optimization (Markowitz) on the full period

print(f"\nMax weight per stock: {MAX_WEIGHT:.0%}")

mean_returns = analysis["statistics"]["Annual Return"]
annual_cov = analysis["annual_covariance"]

min_var = portfolio.min_variance_portfolio(
    mean_returns, annual_cov,
    risk_free_rate=RISK_FREE_RATE,
    max_weight=MAX_WEIGHT
)

max_sharpe = portfolio.max_sharpe_portfolio(
    mean_returns, annual_cov,
    risk_free_rate=RISK_FREE_RATE,
    max_weight=MAX_WEIGHT
)

equal_weight = portfolio.equal_weight_portfolio(
    mean_returns, annual_cov, risk_free_rate=RISK_FREE_RATE
)

portfolio.print_portfolio("MINIMUM VARIANCE PORTFOLIO", min_var)
portfolio.print_portfolio("MAXIMUM SHARPE PORTFOLIO", max_sharpe)
portfolio.print_portfolio("EQUAL WEIGHT PORTFOLIO (benchmark)", equal_weight)

highlighted = {
    "Max Sharpe": max_sharpe,
    "Min Variance": min_var,
    "Equal Weight": equal_weight,
}

try:
    target = portfolio.min_variance_portfolio(
        mean_returns, annual_cov,
        target_return=TARGET_RETURN,
        risk_free_rate=RISK_FREE_RATE,
        max_weight=MAX_WEIGHT
    )
    portfolio.print_portfolio(
        f"LOWEST-RISK PORTFOLIO FOR {TARGET_RETURN:.0%} RETURN", target
    )
    highlighted[f"Target {TARGET_RETURN:.0%}"] = target
except ValueError as error:
    print(f"\nSkipping target-return portfolio: {error}")

frontier = portfolio.efficient_frontier(
    mean_returns, annual_cov, max_weight=MAX_WEIGHT
)


# Step 5: Out-of-sample backtest

benchmark_returns = None

try:
    bench_data = yf.download(
        BENCHMARK,
        start=START_DATE,
        end=END_DATE,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        threads=False
    )
    bench_close = bench_data["Close"]
    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]
    bench_close = bench_close.dropna().sort_index()

    if not bench_close.empty:
        benchmark_returns = bench_close.pct_change().dropna()
        benchmark_returns.name = "OMXS30"
except Exception as error:
    print(f"\nCould not download benchmark {BENCHMARK}: {error}")

test = backtest.run_backtest(
    analysis["returns"],
    train_end=TRAIN_END,
    risk_free_rate=RISK_FREE_RATE,
    max_weight=MAX_WEIGHT,
    benchmark_returns=benchmark_returns
)

backtest.print_backtest(test)


# Step 6: Generate visualizations

visualizations.correlation_heatmap(
    analysis["correlation"]
)

visualizations.risk_return_chart(
    analysis["statistics"]
)

visualizations.stock_performance_chart(
    prices
)

visualizations.efficient_frontier_chart(
    frontier,
    analysis["statistics"],
    highlighted
)

visualizations.backtest_growth_chart(
    test["growth"]
)

visualizations.predicted_vs_actual_chart(
    test["results"]
)

print("\nAnalysis completed successfully!")
print("Opening charts... close the windows to exit.")

visualizations.show_all()