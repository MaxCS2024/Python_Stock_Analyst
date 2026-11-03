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


def load_market_caps(tickers):
    """
    Each company's market value (share price x number of shares) in SEK.
    Used by Black-Litterman. Values Yahoo doesn't have are left out.
    """
    caps, cap_currency = {}, {}
    for t in tickers:
        try:
            value = yf.Ticker(t).fast_info["marketCap"]
        except Exception:
            value = None
        if value and value > 0:
            caps[t] = float(value)
            cur = currency.lookup_currency(t)
            cap_currency[t] = currency.MINOR_UNITS.get(cur, (cur, 1))[0].upper()

    # Convert to SEK with the latest monthly exchange rates
    today = pd.Timestamp.today()
    fx = currency.download_fx(
        set(cap_currency.values()),
        (today - pd.DateOffset(months=3)).strftime("%Y-%m-%d"),
        (today + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    in_sek = {}
    for t, value in caps.items():
        cur = cap_currency[t]
        if cur == currency.BASE_CURRENCY:
            in_sek[t] = value
        elif cur in fx.columns and fx[cur].dropna().size:
            in_sek[t] = value * float(fx[cur].dropna().iloc[-1])
    return in_sek
