# Python Stock Analyst

![Tests](https://github.com/MaxCS2024/Python_Stock_Analyst/actions/workflows/tests.yml/badge.svg)

A Markowitz portfolio optimizer for Swedish stocks. It downloads price history from Yahoo Finance, measures risk and return, finds the best mix of stocks, and then tests honestly whether that mix would have worked.

There are two ways to use it:

- **`main.py`** runs the full analysis in the terminal and saves a PDF report and an Excel file.
- **`dashboard.py`** is an interactive dashboard in the browser, with sliders and plain-language explanations.

Both read the same settings, so they always show the same results.

## What it does

1. **Downloads monthly prices** for your chosen stocks, including dividends, and converts them all to Swedish kronor so foreign stocks include the currency effect.
2. **Calculates the ingredients:** expected return (average and compounded), volatility, covariance and correlation for each stock. Optionally uses Ledoit-Wolf shrinkage for steadier covariance estimates, and Black-Litterman for expected returns: the returns the market as a whole implies, adjusted by your own views.
3. **Optimizes the portfolio** using Markowitz mean-variance optimization:
   - **Max Sharpe:** the best return per unit of risk
   - **Min Variance:** the lowest-risk mix possible
   - **Target return:** the lowest-risk way to reach a return you choose
   - **Equal Weight:** the same amount in every stock, as a simple benchmark
4. **Draws the efficient frontier:** the best possible return at each level of risk.
5. **Backtests out of sample:** picks the weights using only older data, then measures what happened afterwards and compares it with a benchmark index of your choice.
6. **Tests year by year:** re-picks the weights every January using only earlier data, and counts how many years each optimized portfolio beat equal weight.
7. **Includes trading costs** in both backtests: every trade costs a share of its value, so strategies that trade more have to earn it.
8. **Exports everything:** a PDF report to read and share, and an Excel file with all the numbers as live formulas.

Short selling isn't allowed, and you can cap how much any single stock can make up of the portfolio (25% by default).

## Project structure

```text
Python_Stock_Analyst/
├── main.py                 # Full analysis in the terminal, saves PDF + Excel
├── dashboard.py            # Interactive dashboard
├── requirements.txt        # Packages to install
├── settings.json           # Your saved stocks and settings (created on first save)
├── .github/workflows/
│   └── tests.yml           # Runs the tests on GitHub on every push
├── .streamlit/
│   └── config.toml         # Dashboard colors, fonts and network setting
├── src/
│   ├── settings.py         # Loads and saves settings.json, benchmarks, stock names
│   ├── data.py             # Downloads prices, benchmarks and company values, in SEK
│   ├── analysis.py         # Runs every calculation in one go
│   ├── black_litterman.py  # Market-implied expected returns and your views
│   ├── export.py           # Builds the Excel file
│   ├── currency.py         # Converts all prices to SEK
│   ├── calculations.py     # Returns, volatility, covariance, correlation
│   ├── portfolio.py        # Markowitz optimization
│   ├── backtest.py         # Out-of-sample backtest
│   └── visualizations.py   # Optional chart windows (main.py --charts)
├── tests/
│   ├── test_portfolio.py   # Checks the math against a hand-calculated example
│   ├── test_currency.py    # Checks the currency conversion
│   ├── test_calculations.py # Checks compounded returns, Ledoit-Wolf and the yearly test
│   ├── test_app.py         # Runs the whole dashboard and builds a report with fake data
│   └── fake_yfinance.py    # Made-up prices so the tests run without internet
└── visualizations/         # Chart images (only with main.py --charts)
```

## Installation

You need Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the virtual environment.

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

On Mac or Linux:

```bash
source .venv/bin/activate
```

Then install the packages:

```bash
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## Usage

### Dashboard

```bash
streamlit run dashboard.py
```

It opens in your browser at `http://localhost:8501`. Use the sidebar to pick stocks, change the weight cap, the risk-free rate and the target return. Changes are temporary until you click **Save as default**, which also updates what `main.py` uses.

The dashboard has four tabs that follow the method step by step:

| Tab | What it shows |
|---|---|
| 1. The stocks | How 100 kr grew in each stock, risk vs return |
| 2. How they move together | Correlation heatmap and the diversification effect |
| 3. The best mix | The optimized portfolios, efficient frontier and weights |
| 4. Did it hold up? | The backtest, with a slider to move the training/test split |
| 5. Year by year | New weights every year: how many years each portfolio won |

### Terminal

```bash
python main.py
```

Prints all the statistics, portfolios and backtest results, and saves a PDF report and an Excel file to `reports/` (with the date and time in the name).

```bash
python main.py --charts
```

Does the same, and also saves the charts to `visualizations/` and opens them in windows. Close the windows to exit.

### Tests

```bash
python -m pytest tests -v
```

This runs all the tests, including one that runs the whole dashboard with made-up data, so no internet is needed. The same tests run automatically on GitHub every time you push; the badge at the top shows whether they passed.

The main test checks the optimizer against a two-stock example worked out by hand: returns of 10% and 15%, volatilities of 20% and 30%, and a correlation of 0.3. A 12% target should give 60% / 40% weights and 19.35% volatility.

## Settings

Settings are stored in `settings.json`. The easiest way to change them is the dashboard's **Save as default** button, but you can also edit the file by hand:

```json
{
  "tickers": ["SWED-A.ST", "SAND.ST", "VOLV-B.ST", "INVE-B.ST",
              "ATCO-A.ST", "ERIC-B.ST", "SEB-A.ST", "HM-B.ST"],
  "start_date": "2021-10-01",
  "max_weight": 0.25,
  "risk_free_rate": 0.02,
  "target_return": 0.15,
  "train_end": "2024-12-31",
  "benchmark": "^OMX",
  "shrinkage": false,
  "trading_cost": 0.001,
  "return_model": "historical",
  "views": {},
  "view_confidence": 0.5
}
```

| Setting | Meaning |
|---|---|
| `tickers` | Yahoo Finance tickers. Stockholm stocks end in `.ST` |
| `start_date` | First month of price history |
| `max_weight` | Largest share any single stock may have (0.25 = 25%) |
| `risk_free_rate` | Return with no risk, used for the Sharpe ratio |
| `target_return` | Return for the "lowest risk for this return" portfolio |
| `train_end` | Backtest: weights are chosen using data up to this date |
| `benchmark` | Index to compare against: `^OMX` (OMXS30), `URTH` (MSCI World), `ACWI` (MSCI All Country World) or `SPY` (S&P 500) |
| `shrinkage` | `true` uses Ledoit-Wolf shrinkage for steadier covariance estimates |
| `trading_cost` | Cost of every trade as a share of its value, used in the backtests (0.001 = 0.1%) |
| `return_model` | `"historical"` (past averages) or `"black_litterman"` (market-implied returns plus your views) |
| `views` | Black-Litterman views: expected return per year by ticker, e.g. `{"VOLV-B.ST": 0.12}` |
| `view_confidence` | How much to trust your views, from 0 to 1 |

If `settings.json` doesn't exist, the defaults in `src/settings.py` are used.

## How the math works

- **Returns** are monthly percentage changes. The annual return is the average monthly return × 12.
- **Volatility** is the standard deviation of monthly returns × √12.
- **Portfolio return** is the weighted average of the stock returns: w · μ.
- **Portfolio variance** is wᵀΣw, where Σ is the covariance matrix. This is where diversification shows up: stocks with low correlation lower the portfolio's risk.
- **The optimizer** (SciPy's SLSQP) searches for the weights that minimize variance or maximize the Sharpe ratio. The weights must sum to 100%, none may be negative, and none may exceed the cap.

## Reading the results

- **Past returns are a poor forecast.** The optimizer trusts historical average returns completely, so the backtest's predicted vs actual comparison is the most important result.
- **Min Variance is often the most reliable portfolio**, because it only uses volatility and correlation, which are estimated more accurately than returns.
- **Pick a benchmark that matches your stocks:** OMXS30 for Swedish stocks, a world index for a global mix. OMXS30 excludes dividends, so add roughly 3–4% a year to it for a fair comparison; the world indexes include them.
- **A short backtest is one data point, not proof.** The year-by-year test is more convincing: a strategy that wins most years is more likely to be real than lucky.
- **Compounded returns are what you actually earn.** The optimizer uses average returns, which overstate growth for volatile stocks.
- **Trading costs add up.** A strategy that re-picks its weights often has to beat equal weight by more than it pays in costs.
- **Black-Litterman is a more neutral starting point** than historical averages, because it doesn't assume last year's winners keep winning. Your views are not used in the backtests, since that would mean knowing the future.
- **Choosing stocks you already know did well** gives a hindsight advantage the index doesn't have.

## Disclaimer

This is a learning project, not financial advice. Results are based on historical data, which doesn't predict future returns.