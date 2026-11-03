"""
Excel export.

build_excel() writes every number behind the analysis to a workbook, so
anyone can check the calculations or reuse the data.

Wherever Excel can calculate a result itself, the cell holds a formula
instead of a number: returns are calculated from prices, statistics from
returns, and portfolio return and risk from the weights. Change a price or a
weight and everything that depends on it updates.

Colors follow the usual spreadsheet convention:
    blue text   = inputs (prices from Yahoo, the optimizer's weights, settings)
    black text  = formulas
    green text  = links to another sheet
"""

import io
import os
from datetime import date

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src import report, settings

FONT = "Arial"
BLUE, BLACK, GREEN, GREY = "0000FF", "000000", "008000", "666666"
HEADER_FILL = PatternFill("solid", start_color="E7EDEB")
PCT = '0.0%;-0.0%;"-"'
PCT2 = '0.00%;-0.00%;"-"'
NUM = '#,##0.00'
RATIO = '0.00'


def _font(color=BLACK, bold=False, size=10, italic=False):
    return Font(name=FONT, color=color, bold=bold, size=size, italic=italic)


def _header(ws, row, values, start_col=1):
    for i, value in enumerate(values):
        cell = ws.cell(row=row, column=start_col + i, value=value)
        cell.font = _font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center" if i else "left", wrap_text=True)
        cell.border = Border(bottom=Side(style="thin"))


def _put(ws, row, col, value, color=BLACK, fmt=None, bold=False):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = _font(color, bold)
    if fmt:
        cell.number_format = fmt
    return cell


def _widths(ws, widths):
    for col, width in widths.items():
        ws.column_dimensions[col].width = width


def _note(ws, row, text, col=1):
    cell = ws.cell(row=row, column=col, value=text)
    cell.font = _font(GREY, italic=True, size=9)
    return cell


def _col(i):
    return get_column_letter(i)


def build_excel(results, config, currency_info=None):
    """
    results: the dict from analysis.run()
    config:  the settings used
    Returns the workbook as bytes.
    """
    names = results["names"]
    n = len(names)
    returns = results["returns"]
    prices = results["prices"][names]
    stats = results["statistics"]
    portfolios = dict(results["portfolios"])
    if results["target"] is not None:
        portfolios[f"Target {results['target_return']:.0%}"] = results["target"]
    shrink = bool(config.get("shrinkage", False))
    bl_on = results["return_model"] == "black_litterman"

    wb = Workbook()

    # ===========================================================
    # About
    # ===========================================================
    ws = wb.active
    ws.title = "About"
    _put(ws, 1, 1, "Portfolio analysis", bold=True).font = _font(bold=True, size=14)
    _note(ws, 2, f"Created {date.today().isoformat()} by Python Stock Analyst. "
                 "Based on historical data. Not financial advice.")

    _header(ws, 4, ["Setting", "Value", "Note"])
    rf_row = 7
    settings_rows = [
        ("Stocks", ", ".join(names), "Monthly prices from Yahoo Finance, dividends included"),
        ("Period", f"{returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y} "
                   f"({len(returns)} months)", ""),
        ("Risk-free rate", results["risk_free_rate"], "Input. Used for every Sharpe ratio"),
        ("Max weight per stock", results["max_weight"], "Input. No short selling"),
        ("Target return", results["target_return"], "Input"),
        ("Currency", "SEK", "All prices converted to Swedish kronor"
         + (f" (from {', '.join(sorted(set(currency_info['Currency'])))})"
            if currency_info is not None else "")),
        ("Covariance", "Ledoit-Wolf shrinkage" if shrink else "Sample estimate",
         f"Shrinkage intensity {results['shrinkage']:.0%}" if shrink else
         "Calculated with formulas in the Covariance sheet"),
        ("Expected returns", "Black-Litterman" if bl_on else "Historical average",
         "Market-implied returns" + (", adjusted for your views" if config.get("views") else "")
         if bl_on else "Average monthly return x 12, from the Stocks sheet"),
        ("Trading costs", results["trading_cost"], "Input. Share of the value of every trade, "
                                                    "used in the backtests"),
    ]
    for i, (label, value, note) in enumerate(settings_rows):
        row = 5 + i
        _put(ws, row, 1, label, bold=True)
        is_input = isinstance(value, float)
        fmt = PCT2 if label in ("Risk-free rate", "Trading costs") else (PCT if is_input else None)
        _put(ws, row, 2, value, BLUE if is_input else BLACK, fmt)
        _note(ws, row, note, col=3)
    assert ws.cell(row=rf_row, column=1).value == "Risk-free rate"
    rf_ref = f"About!$B${rf_row}"

    row = 5 + len(settings_rows) + 1
    _put(ws, row, 1, "Sheets", bold=True)
    sheet_notes = [
        ("Prices (SEK)", "Monthly prices (inputs from Yahoo Finance)."),
        ("Returns", "Formulas: this month's price / last month's price - 1."),
        ("Stocks", "Formulas: average, compounded return, volatility and Sharpe ratio."),
        ("Correlation", "Formulas: CORREL between every pair of stocks."),
        ("Covariance", "Covariance per year, used for portfolio risk."),
        ("Portfolios", "The optimizer's weights (inputs). Return, volatility and Sharpe "
                       "are formulas: change a weight and they update."),
        ("Backtest", "Results of the app's backtest (calculated by the app)."),
        ("Year by year", "Results of the app's yearly backtest (calculated by the app)."),
    ]
    for i, (sheet, text) in enumerate(sheet_notes):
        _put(ws, row + 1 + i, 1, sheet)
        _put(ws, row + 1 + i, 2, text)
    row = row + len(sheet_notes) + 2
    _note(ws, row, "Colors: blue = inputs, black = formulas, green = links to another sheet.")
    _widths(ws, {"A": 24, "B": 62, "C": 60})

    # ===========================================================
    # Prices (SEK)
    # ===========================================================
    wp = wb.create_sheet("Prices (SEK)")
    first = prices.index.get_loc(returns.index[0]) - 1
    price_rows = prices.iloc[first:prices.index.get_loc(returns.index[-1]) + 1]
    prices_complete = not price_rows.isna().any().any() and len(price_rows) == len(returns) + 1

    _header(wp, 1, ["Month"] + names)
    for r, (month, values) in enumerate(price_rows.iterrows(), start=2):
        _put(wp, r, 1, month.to_pydatetime(), fmt="yyyy-mm")
        for c, value in enumerate(values, start=2):
            _put(wp, r, c, None if pd.isna(value) else float(value), BLUE, NUM)
    wp.freeze_panes = "B2"
    _widths(wp, {"A": 10, **{_col(c): 14 for c in range(2, n + 2)}})

    # ===========================================================
    # Returns
    # ===========================================================
    wr = wb.create_sheet("Returns")
    _header(wr, 1, ["Month"] + names)
    last_row = len(returns) + 1
    for r, (month, values) in enumerate(returns.iterrows(), start=2):
        _put(wr, r, 1, month.to_pydatetime(), fmt="yyyy-mm")
        for c, value in enumerate(values, start=2):
            if prices_complete:
                L = _col(c)
                _put(wr, r, c, f"='Prices (SEK)'!{L}{r + 1}/'Prices (SEK)'!{L}{r}-1", fmt=PCT2)
            else:
                _put(wr, r, c, float(value), BLUE, PCT2)
    if not prices_complete:
        _note(wr, last_row + 2, "Returns are values from the app because some months have "
                                "missing prices.")
    wr.freeze_panes = "B2"
    _widths(wr, {"A": 10, **{_col(c): 14 for c in range(2, n + 2)}})

    def rng(i):
        """The Returns column of stock number i (0-based)."""
        L = _col(i + 2)
        return f"Returns!${L}$2:${L}${last_row}"

    # ===========================================================
    # Stocks
    # ===========================================================
    wsk = wb.create_sheet("Stocks")
    _header(wsk, 1, ["Stock", "Currency", "Average return per year",
                     "Compounded per year", "Volatility per year", "Sharpe ratio",
                     "Expected return used"] + (["Market weight"] if bl_on else []))
    cur = {} if currency_info is None else {
        settings.display_name(t): c for t, c in currency_info["Currency"].items()}

    for i, name in enumerate(names):
        r = i + 2
        _put(wsk, r, 1, name, bold=True)
        _put(wsk, r, 2, cur.get(name, ""))
        _put(wsk, r, 3, f"=AVERAGE({rng(i)})*12", fmt=PCT)
        _put(wsk, r, 4, f"=EXP(SUMPRODUCT(LN(1+{rng(i)}))*12/COUNT({rng(i)}))-1", fmt=PCT)
        _put(wsk, r, 5, f"=STDEV({rng(i)})*SQRT(12)", fmt=PCT)
        _put(wsk, r, 6, f"=IF(E{r}=0,0,(C{r}-{rf_ref})/E{r})", fmt=RATIO)
        if bl_on:
            _put(wsk, r, 7, float(results["expected_returns"][name]), BLUE, PCT)
            _put(wsk, r, 8, float(results["market_weights"][name]), BLUE, PCT)
        else:
            _put(wsk, r, 7, f"=C{r}", fmt=PCT)
    note_row = n + 3
    if bl_on:
        _note(wsk, note_row, "Expected return used: Black-Litterman returns calculated by the "
                             "app from the market weights (company values in SEK), the "
                             "covariance and your views.")
    else:
        _note(wsk, note_row, "Expected return used = the historical average.")
    _note(wsk, note_row + 1, "Compounded per year is what the money actually grew by; the "
                             "average is what the optimizer uses.")
    wsk.freeze_panes = "B2"
    _widths(wsk, {"A": 22, "B": 10, "C": 14, "D": 14, "E": 14, "F": 12, "G": 14, "H": 12})

    # ===========================================================
    # Correlation
    # ===========================================================
    wc = wb.create_sheet("Correlation")
    _header(wc, 1, [""] + names)
    for i, a in enumerate(names):
        _put(wc, i + 2, 1, a, bold=True)
        for j in range(n):
            _put(wc, i + 2, j + 2, f"=CORREL({rng(i)},{rng(j)})", fmt=RATIO)
    _note(wc, n + 3, "1 = always move together, 0 = unrelated, -1 = always opposite.")
    _widths(wc, {"A": 22, **{_col(c): 12 for c in range(2, n + 2)}})

    # ===========================================================
    # Covariance (per year)
    # ===========================================================
    wv = wb.create_sheet("Covariance")
    _header(wv, 1, [""] + names)
    cov = results["annual_covariance"]
    for i, a in enumerate(names):
        _put(wv, i + 2, 1, a, bold=True)
        for j, b in enumerate(names):
            if shrink:
                _put(wv, i + 2, j + 2, float(cov.loc[a, b]), BLUE, "0.0000")
            else:
                _put(wv, i + 2, j + 2,
                     f"=COVAR({rng(i)},{rng(j)})*COUNT({rng(i)})/(COUNT({rng(i)})-1)*12",
                     fmt="0.0000")
    _note(wv, n + 3, "Ledoit-Wolf shrunk covariance per year, calculated by the app."
          if shrink else "Sample covariance per year: monthly covariance x 12.")
    _widths(wv, {"A": 22, **{_col(c): 12 for c in range(2, n + 2)}})

    # ===========================================================
    # Portfolios
    # ===========================================================
    wpf = wb.create_sheet("Portfolios")
    pnames = list(portfolios)
    k = len(pnames)
    helper_start = 3 + k + 1          # leave one empty column between weights and helpers
    _header(wpf, 1, ["Stock", "Expected return"] + pnames)
    _header(wpf, 1, [f"Covariance x weights: {p}" for p in pnames], start_col=helper_start)
    end = n + 1

    for i, name in enumerate(names):
        r = i + 2
        _put(wpf, r, 1, name, bold=True)
        _put(wpf, r, 2, f"=Stocks!G{r}", GREEN, PCT)
        cov_col = _col(i + 2)
        for p, pname in enumerate(pnames):
            w = float(portfolios[pname]["weights"].get(name, 0.0))
            _put(wpf, r, 3 + p, round(w, 6), BLUE, PCT)
            wcol = _col(3 + p)
            _put(wpf, r, helper_start + p,
                 f"=SUMPRODUCT(Covariance!${cov_col}$2:${cov_col}${end},${wcol}$2:${wcol}${end})",
                 fmt="0.0000")

    s = n + 3
    labels = ["Sum of weights", "Expected return", "Volatility", "Sharpe ratio"]
    for j, label in enumerate(labels):
        _put(wpf, s + j, 1, label, bold=True)
    for p in range(k):
        W = _col(3 + p)
        H = _col(helper_start + p)
        _put(wpf, s, 3 + p, f"=SUM({W}2:{W}{end})", fmt=PCT, bold=True)
        _put(wpf, s + 1, 3 + p, f"=SUMPRODUCT({W}2:{W}{end},$B$2:$B${end})", fmt=PCT, bold=True)
        _put(wpf, s + 2, 3 + p, f"=SQRT(SUMPRODUCT({W}2:{W}{end},{H}2:{H}{end}))",
             fmt=PCT, bold=True)
        _put(wpf, s + 3, 3 + p, f"=IF({W}{s + 2}=0,0,({W}{s + 1}-{rf_ref})/{W}{s + 2})",
             fmt=RATIO, bold=True)
    _note(wpf, s + 5, "Weights (blue) come from the optimizer in the app. Change them to see "
                      "how return and risk change; keep the sum at 100%.")
    _note(wpf, s + 6, "Volatility = square root of (weights x covariance x weights), using the "
                      "helper columns on the right.")
    wpf.freeze_panes = "B2"
    _widths(wpf, {"A": 22, "B": 14, **{_col(c): 14 for c in range(3, helper_start + k)}})

    # ===========================================================
    # Backtest
    # ===========================================================
    wb_ = wb.create_sheet("Backtest")
    test = results["test"]
    if test is None:
        _note(wb_, 1, f"The backtest could not run: {results['test_note']}")
    else:
        t0, t1 = test["train_period"]
        e0, e1 = test["test_period"]
        _put(wb_, 1, 1, f"Weights chosen with data from {t0:%b %Y} to {t1:%b %Y}, "
                        f"tested on {e0:%b %Y} to {e1:%b %Y}.", bold=True)
        cols = ["Predicted Return", "Actual Return", "Compounded Return", "Predicted Volatility",
                "Actual Volatility", "Total Return", "Max Drawdown", "Costs Paid",
                "Predicted Sharpe", "Actual Sharpe"]
        res = test["results"]
        _header(wb_, 3, ["Portfolio"] + cols)
        for r, (name, row) in enumerate(res.iterrows(), start=4):
            _put(wb_, r, 1, name, bold=True)
            for c, col in enumerate(cols, start=2):
                value = row.get(col)
                if value is None or pd.isna(value):
                    continue
                _put(wb_, r, c, float(value), BLUE,
                     RATIO if "Sharpe" in col else (PCT2 if col == "Costs Paid" else PCT))
        r = 4 + len(res) + 1
        _note(wb_, r, "Calculated by the app. Returns per year; total return, worst drop "
                      f"(max drawdown) and costs cover the whole test, after trading costs of "
                      f"{results['trading_cost']:.2%} per trade.")

        g = test["growth"]
        r += 2
        _put(wb_, r, 1, "Value of 100 kr", bold=True)
        _header(wb_, r + 1, ["Month"] + list(g.columns))
        for i, (month, values) in enumerate(g.iterrows(), start=r + 2):
            _put(wb_, i, 1, month.to_pydatetime(), fmt="yyyy-mm")
            for c, value in enumerate(values, start=2):
                _put(wb_, i, c, float(value), BLUE, NUM)
    _widths(wb_, {"A": 18, **{_col(c): 13 for c in range(2, 13)}})

    # ===========================================================
    # Year by year
    # ===========================================================
    wy = wb.create_sheet("Year by year")
    rolling = results["rolling"]
    if rolling is None:
        _note(wy, 1, f"The yearly test could not run: {results['rolling_note']}")
    else:
        _put(wy, 1, 1, "New weights every January, chosen with all earlier data.", bold=True)
        years = rolling["years"]
        cols = list(years.columns)
        _header(wy, 3, ["Year"] + cols)
        for r, (year, row) in enumerate(years.iterrows(), start=4):
            _put(wy, r, 1, str(year), bold=True)
            for c, col in enumerate(cols, start=2):
                value = row[col]
                if col == "Best":
                    _put(wy, r, c, value)
                elif not pd.isna(value):
                    _put(wy, r, c, float(value), BLUE, PCT)

        r = 4 + len(years) + 1
        _put(wy, r, 1, "All test years combined", bold=True)
        summary = rolling["summary"]
        scols = ["Compounded Return", "Actual Volatility", "Actual Sharpe", "Max Drawdown",
                 "Costs Paid"]
        _header(wy, r + 1, ["Portfolio", "Compounded per year", "Volatility", "Sharpe ratio",
                            "Worst drop", "Costs paid"])
        for i, (name, row) in enumerate(summary.iterrows(), start=r + 2):
            _put(wy, i, 1, name, bold=True)
            for c, col in enumerate(scols, start=2):
                value = row.get(col)
                if value is None or pd.isna(value):
                    continue
                _put(wy, i, c, float(value), BLUE,
                     RATIO if "Sharpe" in col else (PCT2 if col == "Costs Paid" else PCT))
        wins = rolling["wins"]
        end_row = r + 2 + len(summary) + 1
        _note(wy, end_row, f"Max Sharpe beat equal weight in {wins['Max Sharpe']} of "
                           f"{rolling['n_years']} years, Min Variance in "
                           f"{wins['Min Variance']}. Calculated by the app, after trading costs.")
    _widths(wy, {"A": 18, **{_col(c): 15 for c in range(2, 9)}})

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def save_excel(xlsx_bytes, folder=None):
    """Save the workbook in the reports folder and return the path."""
    folder = folder or os.path.join(settings.PROJECT_ROOT, "reports")
    path = report.unique_path(folder, "portfolio_data", "xlsx")
    with open(path, "wb") as f:
        f.write(xlsx_bytes)
    return path
