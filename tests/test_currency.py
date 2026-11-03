"""
Checks the currency conversion with numbers you can verify by hand.

Run from the project folder:  python tests/test_currency.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import currency

dates = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])

# A US stock that never moves, while the dollar falls from 10 to 9 SEK
prices = pd.DataFrame({
    "US_STOCK": [100.0, 100.0, 100.0],      # USD
    "SE_STOCK": [50.0, 55.0, 60.0],         # SEK
    "UK_STOCK": [1000.0, 1000.0, 1000.0],   # GBp (pence)
}, index=dates)

fx = pd.DataFrame({
    "USD": [10.0, 9.5, 9.0],
    "GBP": [13.0, 13.0, 13.0],
}, index=dates)

currencies = {"US_STOCK": "USD", "SE_STOCK": "SEK", "UK_STOCK": "GBp"}


def test_usd_stock_is_converted():
    sek, info = currency.convert_to_sek(prices, None, None, currencies=currencies, fx=fx)
    # 100 USD x 10 = 1,000 SEK, then 950, then 900
    assert np.allclose(sek["US_STOCK"], [1000, 950, 900])
    # The stock didn't move, but a Swedish investor lost 10%
    assert np.isclose(sek["US_STOCK"].iloc[-1] / sek["US_STOCK"].iloc[0] - 1, -0.10)


def test_sek_stock_is_unchanged():
    sek, _ = currency.convert_to_sek(prices, None, None, currencies=currencies, fx=fx)
    assert np.allclose(sek["SE_STOCK"], [50, 55, 60])


def test_pence_become_pounds():
    sek, _ = currency.convert_to_sek(prices, None, None, currencies=currencies, fx=fx)
    # 1,000 pence = 10 GBP, x 13 = 130 SEK
    assert np.allclose(sek["UK_STOCK"], [130, 130, 130])


def test_missing_exchange_rate_is_reported():
    sek, info = currency.convert_to_sek(
        prices, None, None, currencies={**currencies, "US_STOCK": "XYZ"}, fx=fx)
    assert "US_STOCK" not in sek.columns
    assert not info.loc["US_STOCK", "Converted"]


def test_guess_currency_from_ticker():
    assert currency.guess_currency("SEB-A.ST") == "SEK"
    assert currency.guess_currency("EQNR.OL") == "NOK"
    assert currency.guess_currency("ENR.DE") == "EUR"
    assert currency.guess_currency("MSFT") == "USD"


if __name__ == "__main__":
    test_usd_stock_is_converted()
    test_sek_stock_is_unchanged()
    test_pence_become_pounds()
    test_missing_exchange_rate_is_reported()
    test_guess_currency_from_ticker()
    print("All currency tests passed!")