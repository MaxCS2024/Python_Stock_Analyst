"""
A stand-in for the yfinance package, used by tests/test_app.py.

It returns made-up but realistic monthly prices instead of downloading from
Yahoo Finance, so the tests run without internet (for example on GitHub) and
give the same numbers every time. Only the parts of yfinance this project
uses are imitated.
"""

import zlib

import numpy as np
import pandas as pd


def _seed(ticker):
    # zlib.crc32 gives the same number for the same ticker on every computer
    return zlib.crc32(ticker.encode())


def _series(ticker, dates):
    rng = np.random.default_rng(_seed(ticker))
    if ticker.endswith("=X"):                      # exchange rate around 10 SEK
        moves = rng.normal(0, 0.02, len(dates))
        return 10 * np.cumprod(1 + moves)
    market = np.random.default_rng(1).normal(0.006, 0.04, len(dates))
    own = rng.normal(0.003, 0.05, len(dates))
    return 100 * np.cumprod(1 + market * rng.uniform(0.5, 1.2) + own)


def download(tickers, start=None, end=None, **kwargs):
    if isinstance(tickers, str):
        tickers = [tickers]
    dates = pd.date_range("2015-01-01", pd.Timestamp.today(), freq="MS")
    frame = pd.DataFrame({t: _series(t, dates) for t in tickers}, index=dates)
    if start:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end:
        frame = frame[frame.index < pd.Timestamp(end)]
    frame.columns = pd.MultiIndex.from_product([["Close"], tickers])
    return frame


class Ticker:
    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def fast_info(self):
        t = self.ticker.upper()
        if t.endswith(".ST") or t == "^OMX":
            cur = "SEK"
        elif t.endswith(".OL"):
            cur = "NOK"
        elif t.endswith(".DE"):
            cur = "EUR"
        else:
            cur = "USD"
        return {"currency": cur}

    @property
    def info(self):
        return {"shortName": f"{self.ticker.split('.')[0].title()} Corporation"}
