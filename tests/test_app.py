"""
End-to-end checks: runs the whole dashboard and builds a PDF report.

Uses tests/fake_yfinance.py instead of Yahoo Finance, so it works without
internet and gives the same numbers every time. It uses a temporary folder
for settings, so your own settings.json and names cache aren't touched.

Run from the project folder:  python tests/test_app.py
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Swap in the fake Yahoo Finance before the project code is imported
import fake_yfinance  # noqa: E402
sys.modules["yfinance"] = fake_yfinance

from src import currency, data, report, settings  # noqa: E402

# In case another test imported these first with the real yfinance
currency.yf = fake_yfinance
data.yf = fake_yfinance

# Keep the test away from your own settings and names cache
_tmp = tempfile.mkdtemp()
settings.SETTINGS_FILE = os.path.join(_tmp, "settings.json")
settings.NAMES_CACHE_FILE = os.path.join(_tmp, "names.json")
settings._looked_up.clear()
settings._cache_loaded = True

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = os.path.join(ROOT, "dashboard.py")


def run_dashboard():
    app = AppTest.from_file(DASHBOARD, default_timeout=180)
    app.run()
    return app


def errors(app):
    return [e.value for e in app.exception]


def test_dashboard_runs():
    app = run_dashboard()
    assert not errors(app), errors(app)
    assert [t.label for t in app.tabs] == [
        "1. The stocks", "2. How they move together", "3. The best mix",
        "4. Did it hold up?", "5. Year by year",
    ]


def test_dashboard_settings_can_change():
    app = run_dashboard()
    app.toggle[0].set_value(True).run()            # Ledoit-Wolf on
    assert not errors(app), errors(app)
    app.selectbox[0].set_value("URTH").run()       # MSCI World benchmark
    assert not errors(app), errors(app)
    app.slider[0].set_value(40).run()              # 40% weight cap
    assert not errors(app), errors(app)


def test_dashboard_with_foreign_stocks():
    app = run_dashboard()
    app.text_input[0].set_value("MSFT, EQNR.OL").run()
    assert not errors(app), errors(app)
    assert any("converted from" in c.value for c in app.caption)


def test_report_builds():
    config = settings.load()
    loaded = data.load_prices(config["tickers"], config["start_date"], settings.end_date())
    bench = data.load_benchmark(config["benchmark"], config["start_date"], settings.end_date())
    pdf = report.build_report(loaded["prices"], config, bench, loaded["currency_info"])
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 50_000


if __name__ == "__main__":
    test_dashboard_runs()
    test_dashboard_settings_can_change()
    test_dashboard_with_foreign_stocks()
    test_report_builds()
    print("All app tests passed!")
