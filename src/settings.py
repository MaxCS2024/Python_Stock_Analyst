"""
Shared settings for main.py and dashboard.py.

Your choices are stored in settings.json in the project folder.
The dashboard's "Save as default" button writes that file, and
main.py reads it, so both always use the same stocks and settings.

If settings.json doesn't exist yet, the DEFAULTS below are used.
"""

import json
import os
import re

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "settings.json")
NAMES_CACHE_FILE = os.path.join(PROJECT_ROOT, ".cache", "names.json")

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
    "benchmark": "^OMX",      # index to compare against (see BENCHMARKS)
    "shrinkage": False,       # Ledoit-Wolf: steadier covariance estimates
    "trading_cost": 0.001,    # 0.1% of the value of every trade
    "return_model": "historical",  # "historical" or "black_litterman"
    "views": {},              # Black-Litterman: {ticker: expected return per year}
    "view_confidence": 0.5,   # Black-Litterman: how much to trust your views (0-1)
}


# ---------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------

# Indexes you can compare against. The ETFs (exchange-traded funds) follow
# their index and include reinvested dividends, just like your stocks do.
BENCHMARKS = {
    "^OMX": {
        "name": "OMXS30",
        "description": "Sweden's 30 largest companies",
        "dividends": False,
    },
    "URTH": {
        "name": "MSCI World",
        "description": "about 1,400 large companies in 23 developed countries",
        "dividends": True,
    },
    "ACWI": {
        "name": "MSCI All Country World",
        "description": "developed and emerging markets worldwide",
        "dividends": True,
    },
    "SPY": {
        "name": "S&P 500",
        "description": "the 500 largest US companies",
        "dividends": True,
    },
}


def benchmark_name(ticker):
    return BENCHMARKS.get(ticker, {}).get("name", ticker)


def benchmark_note(ticker):
    """One sentence on how fair the comparison is."""
    name = benchmark_name(ticker)
    info = BENCHMARKS.get(ticker)
    if info is None:
        return f"{name} is a custom benchmark; check whether it includes dividends."
    if info["dividends"]:
        return (f"{name} includes dividends, like your stocks, so the comparison is fair. "
                "All values are in SEK.")
    return (f"{name} excludes dividends, while your stocks include them. "
            "Add roughly 3-4% a year to the index to compare fairly.")


# ---------------------------------------------------------------
# Stock names
# ---------------------------------------------------------------

# Short, hand-picked names. Any other ticker gets its name from Yahoo.
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
    "MSFT": "Microsoft",
    "AAPL": "Apple",
    "NVDA": "Nvidia",
    "TSM": "TSMC",
    "PG": "Procter & Gamble",
    "NEE": "NextEra Energy",
    "RIO": "Rio Tinto",
    "LVMUY": "LVMH",
    "ENR.DE": "Siemens Energy",
    "NOV.DE": "Novo Nordisk",
    "EQNR.OL": "Equinor",
}

# Company-type endings removed from Yahoo names: "Equinor ASA" -> "Equinor"
_LEGAL_ENDINGS = re.compile(
    r"[\s,]+(inc\.?|incorporated|corporation|corp\.?|company|co\.?|limited|ltd\.?|"
    r"plc|ag|se|sa|s\.a\.|nv|n\.v\.|asa|ab|a/s|oyj|spa|s\.p\.a\.|holdings?|"
    r"group|adr|\(publ\)|\(the\)|class [a-z])$",
    re.IGNORECASE,
)

_looked_up = {}       # names fetched from Yahoo in this session
_cache_loaded = False


def _clean_name(name):
    name = name.strip()
    previous = None
    while previous != name:          # remove several endings: "X Holdings plc"
        previous = name
        name = _LEGAL_ENDINGS.sub("", name).strip(" ,.")

    # Keep long names readable in charts: cut at a whole word, max ~22 characters
    if len(name) > 22:
        short = ""
        for word in name.split():
            if len(short) + len(word) + 1 > 22:
                break
            short = f"{short} {word}".strip()
        name = short or name[:22]
    return name


def _load_cache():
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    try:
        with open(NAMES_CACHE_FILE, encoding="utf-8") as f:
            _looked_up.update(json.load(f))
    except (OSError, json.JSONDecodeError):
        pass


def _save_cache():
    try:
        os.makedirs(os.path.dirname(NAMES_CACHE_FILE), exist_ok=True)
        with open(NAMES_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_looked_up, f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def lookup_names(tickers):
    """
    Fetch readable names from Yahoo for tickers we don't know yet.
    Results are saved in .cache/names.json so each ticker is only looked up once.
    """
    import yfinance as yf

    _load_cache()
    new = [t for t in tickers if t not in STOCK_NAMES and t not in _looked_up]

    for t in new:
        try:
            info = yf.Ticker(t).info
            raw = info.get("shortName") or info.get("longName")
        except Exception:
            raw = None
        name = _clean_name(raw) if raw else t

        # Two different tickers must never share a name
        taken = set(STOCK_NAMES.values()) | set(_looked_up.values())
        if name in taken:
            name = f"{name} ({t})"
        _looked_up[t] = name

    if new:
        _save_cache()

    return {t: display_name(t) for t in tickers}


def display_name(ticker):
    _load_cache()
    return STOCK_NAMES.get(ticker) or _looked_up.get(ticker) or ticker


# ---------------------------------------------------------------
# Loading and saving
# ---------------------------------------------------------------

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
