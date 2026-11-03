"""
Full analysis in the terminal.

    python main.py            prints the results and saves a PDF report and an
                              Excel file in the reports folder
    python main.py --charts   also saves the charts as images and opens them
                              in windows (close the windows to exit)

The dashboard (streamlit run dashboard.py) shows the same results interactively.
"""

import argparse
import warnings

# Hide harmless pandas deprecation warnings coming from inside yfinance
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

from src import analysis
from src import backtest
from src import data
from src import export
from src import portfolio
from src import report
from src import settings


parser = argparse.ArgumentParser(description="Markowitz portfolio analysis.")
parser.add_argument("--charts", action="store_true",
                    help="also save the charts as images and open them in windows")
args = parser.parse_args()


# Settings come from settings.json (edit them with the dashboard's
# "Save as default" button, or by editing the file directly)

config = settings.load()

tickers = config["tickers"]
START_DATE = config["start_date"]
END_DATE = settings.end_date()
BENCHMARK = config["benchmark"]
BLACK_LITTERMAN = config["return_model"] == "black_litterman"

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

benchmark_returns = data.load_benchmark(BENCHMARK, START_DATE, END_DATE)
if benchmark_returns is None:
    print(f"\nCould not download the benchmark {BENCHMARK}; the backtests run without it.")

market_caps = data.load_market_caps(prices.columns) if BLACK_LITTERMAN else None


# Step 2: Run every calculation (the same ones the report and Excel file use)

results = analysis.run(prices, config, benchmark_returns, market_caps)


# Step 3: Print the stock statistics

print("\nSTOCK ANALYSIS")
print(results["statistics"].round(4))

print("\nCORRELATION")
print(results["correlation"].round(2))

if config["shrinkage"]:
    print(f"\nLedoit-Wolf shrinkage is on (intensity {results['shrinkage']:.0%}): "
          "covariance blended toward a simpler estimate for steadier weights.")


# Step 4: Portfolio optimization (Markowitz) on the full period

print(f"\nMax weight per stock: {results['max_weight']:.0%}")

if BLACK_LITTERMAN:
    print("\nEXPECTED RETURNS (Black-Litterman)")
    table = results["statistics"][["Annual Return"]].rename(
        columns={"Annual Return": "Historical"})
    table["Market weight"] = results["market_weights"]
    table["Market-implied"] = results["implied_returns"]
    table["Used"] = results["expected_returns"]
    print((table * 100).round(1).astype(str).add("%").to_string())

portfolios = results["portfolios"]
portfolio.print_portfolio("MINIMUM VARIANCE PORTFOLIO", portfolios["Min Variance"])
portfolio.print_portfolio("MAXIMUM SHARPE PORTFOLIO", portfolios["Max Sharpe"])
portfolio.print_portfolio("EQUAL WEIGHT PORTFOLIO (benchmark)", portfolios["Equal Weight"])

highlighted = dict(portfolios)
if results["target"] is not None:
    portfolio.print_portfolio(
        f"LOWEST-RISK PORTFOLIO FOR {results['target_return']:.0%} RETURN", results["target"])
    highlighted[f"Target {results['target_return']:.0%}"] = results["target"]
else:
    print(f"\nSkipping target-return portfolio: {results['target_note']}")


# Step 5: Backtests (after trading costs)

print(f"\nTrading costs in the backtests: {results['trading_cost']:.2%} per trade")

if results["test"] is not None:
    backtest.print_backtest(results["test"])
else:
    print(f"\nSkipping backtest: {results['test_note']}")

if results["rolling"] is not None:
    backtest.print_rolling(results["rolling"])
else:
    print(f"\nSkipping rolling backtest: {results['rolling_note']}")


# Step 6: PDF report and Excel file (saved in the reports/ folder)

pdf = report.build_report(prices, config, benchmark_returns, currency_info,
                          market_caps, results=results)
print(f"\nPDF report saved:  {report.save_report(pdf)}")

xlsx = export.build_excel(results, config, currency_info)
print(f"Excel file saved:  {export.save_excel(xlsx)}")


# Step 7 (optional): chart windows

if not args.charts:
    print("\nAnalysis completed successfully!")
    print("Tip: run 'python main.py --charts' to also open the charts in windows.")
else:
    from src import visualizations   # only loaded when the charts are wanted

    stats = results["statistics"]
    # Plot the stocks at the expected returns the optimizer used
    stats_for_frontier = stats.assign(**{"Annual Return": results["expected_returns"]})

    visualizations.correlation_heatmap(results["correlation"])
    visualizations.risk_return_chart(stats)
    visualizations.stock_performance_chart(results["prices"])
    visualizations.efficient_frontier_chart(results["frontier"], stats_for_frontier, highlighted)

    if results["test"] is not None:
        visualizations.backtest_growth_chart(results["test"]["growth"])
        visualizations.predicted_vs_actual_chart(results["test"]["results"])

    print("\nAnalysis completed successfully!")
    print(f"Charts saved in {visualizations.VISUALIZATION_DIR}")
    print("Opening charts... close the windows to exit.")

    visualizations.show_all()
