"""
Currency conversion.

Yahoo Finance gives each stock's price in its own currency: Microsoft in
US dollars, Siemens Energy in euros, Equinor in Norwegian kroner, and so on.
For a Swedish investor, the return that matters is the return in kronor,
which includes what the currency did against the krona.

This module converts every price to Swedish kronor (SEK):

    price in SEK = price in local currency x exchange rate (SEK per 1 unit)

Example: a stock at 100 USD when 1 USD = 10 SEK is worth 1,000 SEK.
If the stock stays at 100 USD but the dollar falls to 9 SEK, a Swedish
investor has lost 10%, even though the stock didn't move.
"""

import pandas as pd
import yfinance as yf

BASE_CURRENCY = "SEK"

# Used when Yahoo doesn't report a currency: guess from the ticker's ending
SUFFIX_CURRENCY = {
    ".ST": "SEK",   # Stockholm
    ".OL": "NOK",   # Oslo
    ".CO": "DKK",   # Copenhagen
    ".HE": "EUR",   # Helsinki
    ".DE": "EUR",   # Germany (Xetra)
    ".F": "EUR",    # Frankfurt
    ".PA": "EUR",   # Paris
    ".AS": "EUR",   # Amsterdam
    ".BR": "EUR",   # Brussels
    ".MI": "EUR",   # Milan
    ".MC": "EUR",   # Madrid
    ".L": "GBp",    # London (quoted in pence)
    ".SW": "CHF",   # Switzerland
    ".TO": "CAD",   # Toronto
    ".T": "JPY",    # Tokyo
    ".HK": "HKD",   # Hong Kong
    ".AX": "AUD",   # Australia
}

# Some markets quote prices in the small unit: pence instead of pounds, etc.
MINOR_UNITS = {
    "GBp": ("GBP", 100),
    "GBX": ("GBP", 100),
    "ZAc": ("ZAR", 100),
    "ILA": ("ILS", 100),
}


def guess_currency(ticker):
    """Guess a ticker's currency from its ending. No ending usually means USD."""
    upper = ticker.upper()
    for suffix, cur in SUFFIX_CURRENCY.items():
        if upper.endswith(suffix.upper()):
            return cur
    return "USD"


def lookup_currency(ticker):
    """Ask Yahoo which currency the ticker trades in, or guess if that fails."""
    try:
        cur = yf.Ticker(ticker).fast_info["currency"]
        if cur:
            return cur
    except Exception:
        pass
    return guess_currency(ticker)


def _clean_index(frame):
    """Plain dates without time zones, so stock and currency data line up."""
    frame = frame.copy()
    index = pd.to_datetime(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    return frame


def download_fx(currencies, start, end):
    """
    Monthly exchange rates to SEK, one column per currency.
    For example the "USD" column holds how many SEK one US dollar cost.
    """
    pairs = {c: f"{c}{BASE_CURRENCY}=X" for c in sorted(currencies) if c != BASE_CURRENCY}
    if not pairs:
        return pd.DataFrame()

    try:
        data = yf.download(
            list(pairs.values()), start=start, end=end, interval="1mo",
            auto_adjust=True, progress=False, threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(list(pairs.values())[0])

    close = close.rename(columns={pair: cur for cur, pair in pairs.items()})
    return _clean_index(close).sort_index()


def convert_to_sek(prices, start, end, currencies=None, fx=None):
    """
    Convert prices to SEK.

    prices:     monthly prices, one column per TICKER (before renaming)
    currencies: optional {ticker: currency}; looked up on Yahoo if not given
    fx:         optional exchange rates; downloaded if not given

    Returns (prices_sek, info):
        prices_sek: the converted prices
        info:       a table with each ticker's currency and whether it was converted.
                    Stocks whose exchange rate couldn't be downloaded are left out
                    of prices_sek and marked as not converted.
    """
    prices = _clean_index(prices)

    if currencies is None:
        currencies = {t: lookup_currency(t) for t in prices.columns}

    # Turn pence into pounds etc.
    major, divisor = {}, {}
    for t in prices.columns:
        cur, div = MINOR_UNITS.get(currencies[t], (currencies[t], 1))
        major[t], divisor[t] = cur.upper(), div

    if fx is None:
        fx = download_fx(set(major.values()), start, end)
    else:
        fx = _clean_index(fx)

    converted, failed = {}, []
    for t in prices.columns:
        local = prices[t] / divisor[t]
        cur = major[t]

        if cur == BASE_CURRENCY:
            converted[t] = local
            continue

        if cur not in fx.columns or fx[cur].dropna().empty:
            failed.append(t)
            continue

        # Use each month's exchange rate; fill small gaps from the nearest month
        rate = fx[cur].reindex(prices.index.union(fx.index)).sort_index().ffill().bfill()
        converted[t] = local * rate.reindex(prices.index)

    prices_sek = pd.DataFrame(converted, index=prices.index)
    info = pd.DataFrame(
        {
            "Currency": [currencies[t] for t in prices.columns],
            "Converted": [t not in failed for t in prices.columns],
        },
        index=prices.columns,
    )
    return prices_sek, info


def describe(info):
    """Short text like 'USD (6 stocks), EUR (2), NOK (1), SEK (1)'."""
    counts = info["Currency"].value_counts()
    parts = []
    for i, (cur, n) in enumerate(counts.items()):
        label = f"{n} stock{'s' if n != 1 else ''}" if i == 0 else str(n)
        parts.append(f"{cur} ({label})")
    return ", ".join(parts)