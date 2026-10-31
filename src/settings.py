"""
Shared settings for main.py and dashboard.py.

Your choices are stored in settings.json in the project folder.
The dashboard's "Save as default" button writes that file, and
main.py reads it, so both always use the same stocks and settings.

If settings.json doesn't exist yet, the DEFAULTS below are used.
"""

import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")

DEFAULTS = {
    "tickers": [
        "SWED-A.ST", "SAND.ST", "VOLV-B.ST", "INVE-B.ST",
        "ATCO-A.ST", "ERIC-B.ST", "SEB-A.ST", "HM-B.ST",
    ],
    "start_date": "2021-10-01",
    "max_weight": 0.25,       # no single stock above 25%
    "risk_free_rate": 0.02,   # roughly a Swedish savings rate
    "target_return": 0.15,    # lowest-risk portfolio that reaches 15%
    "train_end": "2024-12-31",  # backtest: weights chosen with data up to here
    "benchmark": "^OMX",      # OMXS30 index on Yahoo Finance
}

# Readable names for tickers. Unknown tickers are shown as-is.
STOCK_NAMES = {
    "SWED-A.ST": "Swedbank",
    "SAND.ST": "Sandvik",
    "VOLV-B.ST": "Volvo",
    "INVE-B.ST": "Investor",
    "ATCO-A.ST": "Atlas Copco",
    "ERIC-B.ST": "Ericsson",
    "SEB-A.ST": "SEB",
    "HM-B.ST": "H&M",
    "ABB.ST": "ABB",
    "AZN.ST": "AstraZeneca",
    "SHB-A.ST": "Handelsbanken",
    "NDA-SE.ST": "Nordea",
    "TELIA.ST": "Telia",
    "ASSA-B.ST": "Assa Abloy",
    "ALFA.ST": "Alfa Laval",
    "SAAB-B.ST": "Saab",
    "EVO.ST": "Evolution",
    "HEXA-B.ST": "Hexagon",
    "ESSITY-B.ST": "Essity",
    "BOL.ST": "Boliden",
    "SKF-B.ST": "SKF",
}


def display_name(ticker):
    return STOCK_NAMES.get(ticker, ticker)


def load():
    """Read settings.json, falling back to DEFAULTS for anything missing."""
    settings = dict(DEFAULTS)

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                saved = json.load(f)
            settings.update({k: v for k, v in saved.items() if k in DEFAULTS})
        except (json.JSONDecodeError, OSError) as error:
            print(f"Could not read settings.json ({error}). Using defaults.")

    return settings


def save(settings):
    """Write the settings to settings.json."""
    clean = {k: settings[k] for k in DEFAULTS if k in settings}
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def end_date():
    """First day of next month, so the latest month is always included."""
    return (pd.Timestamp.today() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")