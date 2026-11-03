import warnings

# Hide harmless pandas deprecation warnings coming from inside yfinance
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

from src import calculations
from src import data
from src import portfolio
from src import backtest
from src import report
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
SHRINKAGE = config["shrinkage"]

settings.lookup_names(tickers)   # readable names from Yahoo (cached after the first run)

print(f"Using {len(tickers)} stocks: "
      + ", ".join(settings.display_name(t) for t in tickers))


# Step 1: Download stock data and convert it to Swedish kronor

loaded = data.load_prices(tickers, START_DATE, END_DATE)
prices = loaded["prices"]
currency_info = loaded["currency_info"]

if loaded["missing"]:
    print(f"\nNo data for {', '.join(loaded['missing'])}. They were left out.")

print("\nCURRENCIES (all prices converted to SEK)")
for ticker, row in currency_info.iterrows():
    status = "" if row["Converted"] else "  <- no exchange rate, left out"
    print(f"  {settings.display_name(ticker):<14} {row['Currency']}{status}")

if prices.shape[1] < 2:
    raise ValueError("Fewer than two stocks have usable prices. Check the tickers.")


# Step 2: Run financial calculations

analysis = calculations.calculate_all(prices, shrink=SHRINKAGE)

if SHRINKAGE:
    print(f"\nLedoit-Wolf shrinkage is on (intensity {analysis['shrinkage']:.0%}): "
          "covariance blended toward a simpler estimate for steadier weights.")


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

benchmark_returns = data.load_benchmark(BENCHMARK, START_DATE, END_DATE)
if benchmark_returns is None:
    print(f"\nCould not download the benchmark {BENCHMARK}; the backtest runs without it.")

test = backtest.run_backtest(
    analysis["returns"],
    train_end=TRAIN_END,
    risk_free_rate=RISK_FREE_RATE,
    max_weight=MAX_WEIGHT,
    benchmark_returns=benchmark_returns,
    shrink=SHRINKAGE
)

backtest.print_backtest(test)

try:
    rolling = backtest.rolling_backtest(
        analysis["returns"],
        risk_free_rate=RISK_FREE_RATE,
        max_weight=MAX_WEIGHT,
        benchmark_returns=benchmark_returns,
        shrink=SHRINKAGE
    )
    backtest.print_rolling(rolling)
except ValueError as error:
    print(f"\nSkipping rolling backtest: {error}")


# Step 6: PDF report (saved in the reports/ folder)

pdf = report.build_report(prices, config, benchmark_returns, currency_info)
report_path = report.save_report(pdf)
print(f"\nPDF report saved: {report_path}")


# Step 7: Generate visualizations

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
