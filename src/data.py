"""
Downloading data from Yahoo Finance.

main.py and dashboard.py both get their data through these functions, so a
fix or improvement here applies to both.

    load_prices()     monthly stock prices, converted to Swedish kronor
    load_benchmark()  monthly benchmark returns, converted to Swedish kronor
"""

import pandas as pd
import yfinance as yf

from src import currency, settings


def download_prices(tickers, start, end):
    """
    Monthly closing prices in each stock's own currency.
    Dividends are included (auto_adjust), so returns are total returns.
    Tickers Yahoo has no data for are left out.
    """
    tickers = list(tickers)
    data = yf.download(
        tickers, start=start, end=end, interval="1mo",
        auto_adjust=True, progress=False, threads=False,
    )
    if data.empty:
        return pd.DataFrame()

    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])

    return close.dropna(axis=1, how="all").sort_index()


def load_prices(tickers, start, end):
    """
    Download prices and convert them to Swedish kronor.

    Returns a dict with:
        prices:        prices in SEK, one column per ticker
        local_prices:  the same prices in each stock's own currency
        currency_info: each ticker's currency and whether it was converted
        missing:       tickers Yahoo had no data for
        not_converted: tickers whose exchange rate couldn't be downloaded
    """
    local = download_prices(tickers, start, end)
    missing = [t for t in tickers if t not in local.columns]

    if local.empty:
        return {
            "prices": local, "local_prices": local,
            "currency_info": pd.DataFrame(columns=["Currency", "Converted"]),
            "missing": missing, "not_converted": [],
        }

    prices, currency_info = currency.convert_to_sek(local, start, end)
    not_converted = list(currency_info.index[~currency_info["Converted"]])

    return {
        "prices": prices,
        "local_prices": local.drop(columns=not_converted),
        "currency_info": currency_info,
        "missing": missing,
        "not_converted": not_converted,
    }


def load_benchmark(benchmark, start, end):
    """
    Monthly returns of the benchmark index in SEK, named after the index
    (e.g. "OMXS30"). Returns None if the index can't be downloaded.
    """
    try:
        prices = download_prices([benchmark], start, end)
        if prices.empty:
            return None

        prices_sek, _ = currency.convert_to_sek(prices, start, end)
        if prices_sek.empty:
            return None

        returns = prices_sek.iloc[:, 0].dropna().pct_change().dropna()
        returns.name = settings.benchmark_name(benchmark)
        return returns
    except Exception:
        return None
