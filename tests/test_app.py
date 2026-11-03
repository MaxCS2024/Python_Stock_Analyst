"""
End-to-end checks: runs the whole dashboard and main.py, and builds a PDF
report and an Excel file.

Uses tests/fake_yfinance.py instead of Yahoo Finance, so it works without
internet and gives the same numbers every time. It uses a temporary folder
for settings, so your own settings.json and names cache aren't touched.

Run from the project folder:  python tests/test_app.py
"""

import os
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Swap in the fake Yahoo Finance before the project code is imported
import fake_yfinance  # noqa: E402
sys.modules["yfinance"] = fake_yfinance

from src import analysis, currency, data, export, report, settings  # noqa: E402

# In case another test imported these first with the real yfinance
currency.yf = fake_yfinance
data.yf = fake_yfinance

# Keep the test away from your own settings and names cache
_tmp = tempfile.mkdtemp()
settings.SETTINGS_FILE = os.path.join(_tmp, "settings.json")
settings.NAMES_CACHE_FILE = os.path.join(_tmp, "names.json")
settings.PROJECT_ROOT = _tmp          # reports are saved here, not in your project
settings._looked_up.clear()
settings._cache_loaded = True

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = os.path.join(ROOT, "dashboard.py")
MAIN = os.path.join(ROOT, "main.py")


def run_main(*arguments):
    """Run main.py as if from the terminal, and return what it printed."""
    import contextlib
    import io
    import runpy

    import matplotlib
    matplotlib.use("Agg")                  # draw charts without opening windows
    from src import visualizations
    visualizations.VISUALIZATION_DIR = os.path.join(_tmp, "visualizations")
    os.makedirs(visualizations.VISUALIZATION_DIR, exist_ok=True)

    old_argv, sys.argv = sys.argv, ["main.py", *arguments]
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output), warnings.catch_warnings():
            # The test draws charts without a screen, so there's nothing to show
            warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
            runpy.run_path(MAIN, run_name="__main__")
    finally:
        sys.argv = old_argv
    return output.getvalue()


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


def test_black_litterman_and_costs():
    app = run_dashboard()
    app.radio[0].set_value("black_litterman").run()
    assert not errors(app), errors(app)
    cost = [s for s in app.slider if s.label.startswith("Trading cost")][0]
    cost.set_value(0.5).run()
    assert not errors(app), errors(app)
    assert any("0.50% per trade" in c.value for c in app.caption)


def test_saved_views_are_used():
    settings.save(dict(settings.DEFAULTS, return_model="black_litterman",
                       views={"VOLV-B.ST": 0.30}, view_confidence=0.9))
    try:
        app = run_dashboard()
        assert not errors(app), errors(app)
        table = [d.value for d in app.dataframe if "Your view" in d.value.columns][0]
        assert abs(table.loc["Volvo", "Your view"] - 30) < 1e-9
        # A confident view of 30% pulls Volvo's expected return well above market-implied
        assert table.loc["Volvo", "Used by the optimizer"] > table.loc["Volvo", "Market-implied"]
    finally:
        os.remove(settings.SETTINGS_FILE)


def test_generate_report_button():
    app = run_dashboard()
    [b for b in app.button if "Generate" in b.label][0].click().run()
    assert not errors(app), errors(app)
    labels = [b.label for b in app.get("download_button")]
    assert "⬇ Download PDF report" in labels and "⬇ Download Excel file" in labels
    files = os.listdir(os.path.join(_tmp, "reports"))
    assert any(f.endswith(".pdf") for f in files) and any(f.endswith(".xlsx") for f in files)


def test_report_and_excel_build():
    config = dict(settings.load(), return_model="black_litterman", shrinkage=True)
    start, end = config["start_date"], settings.end_date()
    loaded = data.load_prices(config["tickers"], start, end)
    bench = data.load_benchmark(config["benchmark"], start, end)
    caps = data.load_market_caps(config["tickers"])
    results = analysis.run(loaded["prices"], config, bench, caps)

    pdf = report.build_report(loaded["prices"], config, bench, loaded["currency_info"],
                              caps, results=results)
    assert pdf.startswith(b"%PDF") and len(pdf) > 50_000

    from io import BytesIO
    from openpyxl import load_workbook
    xlsx = export.build_excel(results, config, loaded["currency_info"])
    wb = load_workbook(BytesIO(xlsx))
    assert wb.sheetnames == ["About", "Prices (SEK)", "Returns", "Stocks", "Correlation",
                             "Covariance", "Portfolios", "Backtest", "Year by year"]
    # Returns and statistics are formulas, so the sheet recalculates
    assert wb["Returns"]["B2"].value.startswith("='Prices (SEK)'!")
    assert wb["Stocks"]["C2"].value.startswith("=AVERAGE(")
    # The optimizer's weights are stored as they are
    first = results["names"][0]
    assert abs(wb["Portfolios"]["C2"].value
               - results["portfolios"]["Max Sharpe"]["weights"][first]) < 1e-6


def test_main_without_charts():
    before = set(os.listdir(os.path.join(_tmp, "reports"))) \
        if os.path.isdir(os.path.join(_tmp, "reports")) else set()
    printed = run_main()
    assert "Analysis completed successfully!" in printed
    assert "PDF report saved" in printed and "Excel file saved" in printed
    new = set(os.listdir(os.path.join(_tmp, "reports"))) - before
    assert any(f.endswith(".pdf") for f in new) and any(f.endswith(".xlsx") for f in new)


def test_main_with_charts():
    printed = run_main("--charts")
    assert "Charts saved in" in printed
    charts = os.listdir(os.path.join(_tmp, "visualizations"))
    assert "correlation_heatmap.png" in charts and "efficient_frontier.png" in charts


if __name__ == "__main__":
    test_dashboard_runs()
    test_dashboard_settings_can_change()
    test_dashboard_with_foreign_stocks()
    test_black_litterman_and_costs()
    test_saved_views_are_used()
    test_generate_report_button()
    test_report_and_excel_build()
    test_main_without_charts()
    test_main_with_charts()
    print("All app tests passed!")