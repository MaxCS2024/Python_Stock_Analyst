"""
Standardized PDF report.

build_report() recalculates everything from the prices and settings,
so the report always has the same layout and matches main.py and the
dashboard. It returns the PDF as bytes, which the dashboard offers as
a download and main.py saves in the reports/ folder.

Page 1:  Summary (settings, key findings, results)
Page 2:  1. The stocks
Page 3:  2. How they move together
Page 4:  3. The best mix
Page 5:  4. Did it hold up? (backtest)
"""

import io
import math
import os
from datetime import date
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from src import backtest, calculations, currency, portfolio, settings


# ---------------------------------------------------------------
# Look and feel (same colors as the dashboard)
# ---------------------------------------------------------------

INK = "#1E3442"
MUTED = "#5E6B73"
GRID = "#DCE3E1"
PANEL = "#EEF2F1"

PORTFOLIO_COLORS = {
    "Max Sharpe": "#1F6FEB",
    "Min Variance": "#0F9D58",
    "Equal Weight": "#EA580C",
}
BENCHMARK_COLOR = "#64748B"
TARGET_COLOR = "#E8204F"
STOCK_COLORS = [
    "#1F6FEB", "#0F9D58", "#EA580C", "#E8204F", "#8B3DF0",
    "#0894B8", "#E0369A", "#B5651D", "#4D8C0A", "#4C51E8",
]

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

_base = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("title", parent=_base["Title"], fontName="Helvetica-Bold",
                            fontSize=24, leading=28, textColor=INK, alignment=0,
                            spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", parent=_base["Normal"], fontSize=10,
                               leading=14, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", parent=_base["Heading1"], fontName="Helvetica-Bold",
                         fontSize=17, leading=21, textColor=INK, spaceBefore=0,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=_base["Heading2"], fontName="Helvetica-Bold",
                         fontSize=12, leading=15, textColor=INK, spaceBefore=12,
                         spaceAfter=6),
    "lead": ParagraphStyle("lead", parent=_base["Normal"], fontSize=12, leading=17,
                           textColor=INK, spaceAfter=10),
    "body": ParagraphStyle("body", parent=_base["Normal"], fontSize=9.5, leading=13.5,
                           textColor=INK, spaceAfter=4),
    "bullet": ParagraphStyle("bullet", parent=_base["Normal"], fontSize=9.5,
                             leading=13.5, textColor=INK, leftIndent=12,
                             bulletIndent=2, spaceAfter=3),
    "small": ParagraphStyle("small", parent=_base["Normal"], fontSize=8, leading=11,
                            textColor=MUTED),
    "cell": ParagraphStyle("cell", parent=_base["Normal"], fontSize=9, leading=12,
                           textColor=INK),
}


def _pct(x, decimals=1):
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:.{decimals}f}%"


def _p(text, style="body"):
    """Paragraph helper. Text may contain <b> tags; names must be escaped."""
    return Paragraph(text, STYLES[style])


def _join(items):
    items = [escape(str(i)) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _sharpe_verdict(s):
    if s >= 1.0:
        return "Excellent"
    if s >= 0.7:
        return "Good"
    if s >= 0.3:
        return "OK"
    return "Poor"


def _correlation_verdict(c):
    if c < 0:
        return "opposite"
    if c < 0.3:
        return "mostly independent"
    if c < 0.6:
        return "somewhat linked"
    return "move together"


# ---------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------

def _table(rows, col_widths, number_cols=(), highlight_first_col=True):
    """A clean table: header row, light row stripes, numbers right-aligned."""
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), PANEL))
    for c in number_cols:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    if highlight_first_col:
        style.append(("FONT", (0, 1), (0, -1), "Helvetica-Bold", 9))
    table.setStyle(TableStyle(style))
    return table


def _note_box(title, lines):
    """A shaded "How to read this" box."""
    content = [_p(f"<b>{title}</b>", "body")]
    content += [Paragraph(line, STYLES["bullet"], bulletText="•") for line in lines]
    box = Table([[content]], colWidths=[CONTENT_WIDTH])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([Spacer(1, 6), box])


def _new_figure(width_mm, height_mm):
    fig = Figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=200)
    ax = fig.add_subplot(111)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _figure_to_image(fig, width_mm, height_mm):
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, facecolor="white")
    buffer.seek(0)
    return Image(buffer, width=width_mm * mm, height=height_mm * mm)


# ---------------------------------------------------------------
# Charts
# ---------------------------------------------------------------

def _growth_chart(prices, color_of, scale=1.0):
    fig, ax = _new_figure(174, 60 * scale)
    growth = prices.dropna()
    growth = growth / growth.iloc[0] * 100
    for name in growth.columns:
        ax.plot(growth.index, growth[name], color=color_of[name], linewidth=1.3, label=name)
    ax.axhline(100, color=MUTED, linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Value of 100 kr", fontsize=7.5, color=MUTED)
    ax.legend(fontsize=6.5, ncol=min(5, len(growth.columns)), frameon=False,
              loc="upper left")
    return _figure_to_image(fig, 174, 60 * scale)


def _risk_return_chart(stats, color_of, scale=1.0):
    fig, ax = _new_figure(174, 56 * scale)
    for name in stats.index:
        x = stats.loc[name, "Annual Volatility"] * 100
        y = stats.loc[name, "Annual Return"] * 100
        ax.scatter(x, y, s=40, color=color_of[name], zorder=3)
        ax.annotate(name, (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_xlabel("Volatility per year (%)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("Return per year (%)", fontsize=7.5, color=MUTED)
    return _figure_to_image(fig, 174, 56 * scale)


def _heatmap_chart(corr, scale=1.0):
    size = (120 if len(corr) <= 10 else 150) * scale
    fig = Figure(figsize=(size / 25.4, size * 0.85 / 25.4), dpi=200)
    ax = fig.add_subplot(111)
    cmap = LinearSegmentedColormap.from_list("gr", ["#16A34A", "#FFFFFF", "#DC2626"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7, color=INK)
    ax.set_yticklabels(corr.index, fontsize=7, color=INK)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=6.5, colors=MUTED)
    cbar.outline.set_visible(False)
    return _figure_to_image(fig, size, size * 0.85)


def _frontier_chart(frontier, stats, portfolios, target, target_return, scale=1.0):
    fig, ax = _new_figure(174, 88 * scale)
    ax.plot(frontier["Volatility"] * 100, frontier["Return"] * 100, color=INK,
            linewidth=2, label="Efficient frontier", zorder=2)
    ax.scatter(stats["Annual Volatility"] * 100, stats["Annual Return"] * 100,
               s=22, color="#94A3B8", zorder=3, label="Single stocks")
    for name in stats.index:
        ax.annotate(name, (stats.loc[name, "Annual Volatility"] * 100,
                           stats.loc[name, "Annual Return"] * 100),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=MUTED)
    markers = {"Max Sharpe": "*", "Min Variance": "D", "Equal Weight": "s"}
    for name, p in portfolios.items():
        ax.scatter(p["volatility"] * 100, p["return"] * 100, marker=markers[name],
                   s=130 if name == "Max Sharpe" else 55, color=PORTFOLIO_COLORS[name],
                   edgecolors="white", linewidths=0.8, zorder=4, label=name)
    if target is not None:
        ax.scatter(target["volatility"] * 100, target["return"] * 100, marker="^",
                   s=60, color=TARGET_COLOR, edgecolors="white", linewidths=0.8,
                   zorder=4, label=f"Target {target_return:.0%}")
    ax.set_xlabel("Volatility per year (%)", fontsize=7.5, color=MUTED)
    ax.set_ylabel("Return per year (%)", fontsize=7.5, color=MUTED)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    return _figure_to_image(fig, 174, 88 * scale)


def _backtest_chart(growth, scale=1.0):
    fig, ax = _new_figure(174, 62 * scale)
    for name in growth.columns:
        is_portfolio = name in PORTFOLIO_COLORS
        ax.plot(growth.index, growth[name],
                color=PORTFOLIO_COLORS.get(name, BENCHMARK_COLOR),
                linewidth=1.8, linestyle="-" if is_portfolio else "--",
                label=f"{name} ({growth[name].iloc[-1]:.0f} kr)")
    ax.axhline(100, color=MUTED, linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Value of 100 kr", fontsize=7.5, color=MUTED)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return _figure_to_image(fig, 174, 62 * scale)


class _SectionStart(Flowable):
    """Invisible marker that records the page a section starts on."""

    def __init__(self, record, name):
        super().__init__()
        self.record, self.name = record, name

    def wrap(self, available_width, available_height):
        return 0, 0

    def draw(self):
        self.record[self.name] = self.canv.getPageNumber()


# ---------------------------------------------------------------
# Page header/footer
# ---------------------------------------------------------------

def _footer(created, pages_seen):
    def draw(canvas, doc):
        pages_seen.append(doc.page)
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor(GRID))
        canvas.line(MARGIN, 14 * mm, PAGE_WIDTH - MARGIN, 14 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(MARGIN, 10 * mm,
                          f"Portfolio report, {created}. Based on historical data. "
                          "Not financial advice.")
        canvas.drawRightString(PAGE_WIDTH - MARGIN, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return draw


# ---------------------------------------------------------------
# The report
# ---------------------------------------------------------------

def build_report(prices, config, benchmark_returns=None, currency_info=None):
    """
    Build the PDF report and return it as bytes.

    prices:            monthly prices in SEK, one column per stock (tickers or names)
    config:            settings dict (see src/settings.py)
    benchmark_returns: optional monthly index returns, named after the index
    currency_info:     optional table from currency.convert_to_sek()
    """
    created = date.today().isoformat()

    # --- Calculations (same as main.py and the dashboard) ---
    prices = prices.rename(columns=settings.display_name)
    analysis = calculations.calculate_all(prices)
    returns = analysis["returns"]
    stats = analysis["statistics"]
    corr = analysis["correlation"]
    annual_cov = analysis["annual_covariance"]
    mean_returns = stats["Annual Return"]
    names = list(stats.index)
    color_of = {n: STOCK_COLORS[i % len(STOCK_COLORS)] for i, n in enumerate(names)}

    rf = config["risk_free_rate"]
    target_return = config["target_return"]
    max_weight = config["max_weight"]
    if max_weight * len(names) < 1:
        max_weight = math.ceil(100 / len(names)) / 100

    max_sharpe = portfolio.max_sharpe_portfolio(
        mean_returns, annual_cov, risk_free_rate=rf, max_weight=max_weight)
    min_var = portfolio.min_variance_portfolio(
        mean_returns, annual_cov, risk_free_rate=rf, max_weight=max_weight)
    equal = portfolio.equal_weight_portfolio(mean_returns, annual_cov, risk_free_rate=rf)
    portfolios = {"Max Sharpe": max_sharpe, "Min Variance": min_var, "Equal Weight": equal}
    frontier = portfolio.efficient_frontier(mean_returns, annual_cov, max_weight=max_weight)

    try:
        target = portfolio.min_variance_portfolio(
            mean_returns, annual_cov, target_return=target_return,
            risk_free_rate=rf, max_weight=max_weight)
        target_note = None
    except ValueError as error:
        target, target_note = None, str(error)

    try:
        test = backtest.run_backtest(
            returns, train_end=config["train_end"], risk_free_rate=rf,
            max_weight=max_weight, benchmark_returns=benchmark_returns)
        test_note = None
    except ValueError as error:
        test, test_note = None, str(error)

    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    period = f"{returns.index[0]:%b %Y} to {returns.index[-1]:%b %Y}"

    drawn = {}

    def chart(name, scale, draw):
        """Charts are slow to draw, so each chart is drawn once per size."""
        if (name, scale) not in drawn:
            drawn[(name, scale)] = draw(scale)
        return drawn[(name, scale)]

    def make_story(scales, marks):
        story = []

        # ===========================================================
        # Page 1: Summary
        # ===========================================================
        story.append(_p("Portfolio report", "title"))
        story.append(_p(
            f"Created {created}. {len(names)} stocks, monthly data from {period} "
            f"({len(returns)} months).", "subtitle"))

        settings_rows = [
            ["Setting", "Value"],
            ["Stocks", Paragraph(_join(names), STYLES["cell"])],
            ["Period", period],
            ["Currency", "All values in SEK" + (
                f", converted from {currency.describe(currency_info)}"
                if currency_info is not None
                and (currency_info["Currency"] != currency.BASE_CURRENCY).any() else "")],
            ["Max weight per stock", f"{max_weight:.0%} (no short selling)"],
            ["Benchmark", settings.benchmark_name(config["benchmark"]) + (
                f" ({settings.BENCHMARKS[config['benchmark']]['description']})"
                if config["benchmark"] in settings.BENCHMARKS else "")],
            ["Risk-free rate", _pct(rf, 2)],
            ["Target return", _pct(target_return)],
        ]
        if test is not None:
            tr0, tr1 = test["train_period"]
            te0, te1 = test["test_period"]
            settings_rows.append(["Backtest",
                                  f"Weights chosen on {tr0:%b %Y} to {tr1:%b %Y}, "
                                  f"tested on {te0:%b %Y} to {te1:%b %Y}"])
        story.append(_table(settings_rows, [45 * mm, CONTENT_WIDTH - 45 * mm]))

        # Key findings
        best = stats["Annual Return"].idxmax()
        calmest = stats["Annual Volatility"].idxmin()
        avg_vol = stats["Annual Volatility"].mean()
        used = max_sharpe["weights"][max_sharpe["weights"] > 0.005]
        top = used.nlargest(min(3, len(used)))

        findings = [
            f"<b>{escape(best)}</b> had the highest return, about "
            f"{_pct(stats.loc[best, 'Annual Return'])} a year. <b>{escape(calmest)}</b> was the "
            f"calmest, with {_pct(stats.loc[calmest, 'Annual Volatility'])} volatility.",
            f"Holding all {len(names)} stocks equally cut volatility from "
            f"{_pct(avg_vol)} (average single stock) to {_pct(equal['volatility'])}. "
            f"That is the diversification effect.",
            f"The best risk-adjusted mix (Max Sharpe) expects {_pct(max_sharpe['return'])} a "
            f"year at {_pct(max_sharpe['volatility'])} volatility. {_join(top.index)} make up "
            f"{_pct(top.sum(), 0)} of it.",
        ]
        if test is not None:
            res, final = test["results"], test["growth"].iloc[-1]
            winner = final.idxmax()
            findings.append(
                f"In the backtest, 100 kr in Max Sharpe became {final['Max Sharpe']:.0f} kr, "
                f"against {final['Equal Weight']:.0f} kr for equal weight. The best result "
                f"came from {escape(winner)}. The optimizer predicted "
                f"{_pct(res.loc['Max Sharpe', 'Predicted Return'])} a year and got "
                f"{_pct(res.loc['Max Sharpe', 'Actual Return'])}.")

        story.append(_p("Key findings", "h2"))
        for f in findings:
            story.append(Paragraph(f, STYLES["bullet"], bulletText="•"))

        story.append(_p("Portfolios (measured on the full period)", "h2"))
        rows = [["Portfolio", "Expected return", "Volatility", "Sharpe ratio"]]
        for name, p in portfolios.items():
            rows.append([name, _pct(p["return"]), _pct(p["volatility"]), f"{p['sharpe']:.2f}"])
        if target is not None:
            rows.append([f"Target {target_return:.0%}", _pct(target["return"]),
                         _pct(target["volatility"]), f"{target['sharpe']:.2f}"])
        story.append(_table(rows, [55 * mm, 40 * mm, 40 * mm, CONTENT_WIDTH - 135 * mm],
                            number_cols=(1, 2, 3)))

        if test is not None:
            story.append(_p("Backtest (weights chosen without seeing the test period)", "h2"))
            res = test["results"]
            rows = [["Portfolio", "Predicted", "Actual", "Total return", "Worst drop",
                     "Actual Sharpe"]]
            for name in res.index:
                rows.append([
                    name,
                    _pct(res.loc[name].get("Predicted Return")),
                    _pct(res.loc[name, "Actual Return"]),
                    _pct(res.loc[name, "Total Return"]),
                    _pct(res.loc[name, "Max Drawdown"]),
                    f"{res.loc[name, 'Actual Sharpe']:.2f}",
                ])
            story.append(_table(rows, [38 * mm, 24 * mm, 24 * mm, 28 * mm, 26 * mm,
                                       CONTENT_WIDTH - 140 * mm], number_cols=(1, 2, 3, 4, 5)))
            story.append(Spacer(1, 4))
            story.append(_p("Predicted and actual are returns per year. Total return and "
                            "worst drop cover the whole test period.", "small"))

        story.append(PageBreak())

        # ===========================================================
        # Page 2: The stocks
        # ===========================================================
        story.append(_SectionStart(marks, "stocks"))
        story.append(_p("1. The stocks", "h1"))
        story.append(_p(
            f"{escape(best)} grew the most, about {_pct(stats.loc[best, 'Annual Return'])} a "
            f"year. Return per unit of risk (the Sharpe ratio) is the fairest way to "
            f"compare them.", "lead"))

        ranked = stats.assign(Sharpe=(stats["Annual Return"] - rf) / stats["Annual Volatility"])
        ranked = ranked.sort_values("Sharpe", ascending=False)
        rows = [["Stock", "Return per year", "Volatility per year", "Return per unit of risk",
                 "Verdict"]]
        for name, r in ranked.iterrows():
            rows.append([name, _pct(r["Annual Return"]), _pct(r["Annual Volatility"]),
                         f"{r['Sharpe']:.2f}", _sharpe_verdict(r["Sharpe"])])
        story.append(_table(rows, [42 * mm, 30 * mm, 34 * mm, 40 * mm,
                                   CONTENT_WIDTH - 146 * mm], number_cols=(1, 2, 3)))

        story.append(_p("What 100 kr became", "h2"))
        story.append(chart("growth", scales["stocks"],
                           lambda s: _growth_chart(prices.loc[:, names], color_of, s)))
        story.append(_p("Return vs risk (up and to the left is better)", "h2"))
        story.append(chart("risk", scales["stocks"],
                           lambda s: _risk_return_chart(stats, color_of, s)))
        story.append(_note_box("How to read this", [
            "<b>Return per year</b>: average monthly return x 12, dividends included. "
            "5-10% is normal for stocks over the long run.",
            "<b>Volatility</b>: how much the price swings. 18-25% is normal for a large "
            "company; above 30% is wild.",
            "<b>Return per unit of risk</b>: below 0.3 is poor, 0.3-0.7 OK, 0.7-1.0 good, "
            "above 1.0 excellent.",
        ]))
        story.append(PageBreak())

        # ===========================================================
        # Page 3: Correlation
        # ===========================================================
        alike, different = pairs.idxmax(), pairs.idxmin()
        story.append(_SectionStart(marks, "corr"))
        story.append(_p("2. How they move together", "h1"))
        story.append(_p(
            f"Holding all {len(names)} stocks equally gives {_pct(equal['volatility'])} "
            f"volatility, against {_pct(avg_vol)} for the average single stock.", "lead"))
        story.append(chart("heatmap", scales["corr"], lambda s: _heatmap_chart(corr, s)))

        story.append(_p("Risk alone vs together", "h2"))
        rows = [["", "Volatility per year"],
                ["Average single stock", _pct(avg_vol)],
                ["Equal weight portfolio", _pct(equal["volatility"])],
                ["Min variance portfolio", _pct(min_var["volatility"])]]
        story.append(_table(rows, [70 * mm, 45 * mm], number_cols=(1,)))

        story.append(_p("Pairs worth noticing", "h2"))
        story.append(Paragraph(
            f"Most alike: <b>{escape(alike[0])}</b> and <b>{escape(alike[1])}</b> "
            f"({pairs.max():.2f}, {_correlation_verdict(pairs.max())}).",
            STYLES["bullet"], bulletText="•"))
        story.append(Paragraph(
            f"Most different: <b>{escape(different[0])}</b> and <b>{escape(different[1])}</b> "
            f"({pairs.min():.2f}, {_correlation_verdict(pairs.min())}).",
            STYLES["bullet"], bulletText="•"))
        story.append(Paragraph(
            f"Average correlation between all pairs: {pairs.mean():.2f}.",
            STYLES["bullet"], bulletText="•"))

        story.append(_note_box("How to read this", [
            "<b>Correlation</b> runs from -1 to +1. Green squares move apart (good for "
            "diversification), red squares move together.",
            "Below 0.3 is mostly independent, 0.3-0.6 somewhat linked, above 0.6 the stocks "
            "move together. Negative values are rare and valuable.",
            "Stocks in the same sector, like two banks, usually have high correlation.",
        ]))
        story.append(PageBreak())

        # ===========================================================
        # Page 4: The best mix
        # ===========================================================
        story.append(_SectionStart(marks, "mix"))
        story.append(_p("3. The best mix", "h1"))
        story.append(_p(
            f"The best return for the risk is about {_pct(max_sharpe['return'])} a year at "
            f"{_pct(max_sharpe['volatility'])} volatility, using {len(used)} of "
            f"{len(names)} stocks.", "lead"))
        story.append(chart("frontier", scales["mix"], lambda s: _frontier_chart(
            frontier, stats, portfolios, target, target_return, s)))

        story.append(_p(f"Where the money goes (cap {max_weight:.0%} per stock)", "h2"))
        weight_cols = dict(portfolios)
        if target is not None:
            weight_cols[f"Target {target_return:.0%}"] = target
        header = ["Stock"] + list(weight_cols.keys())
        rows = [header]
        weights = pd.DataFrame({n: p["weights"] for n, p in weight_cols.items()})
        weights = weights.loc[weights["Max Sharpe"].sort_values(ascending=False).index]
        for name, r in weights.iterrows():
            rows.append([name] + [_pct(v) if v >= 0.0005 else "-" for v in r])
        other = (CONTENT_WIDTH - 45 * mm) / (len(header) - 1)
        story.append(_table(rows, [45 * mm] + [other] * (len(header) - 1),
                            number_cols=tuple(range(1, len(header)))))
        if target_note:
            story.append(Spacer(1, 4))
            story.append(_p(escape(target_note), "small"))

        story.append(_note_box("How to read this", [
            "The <b>efficient frontier</b> is the best possible return for each level of "
            "risk. Single stocks sit below it.",
            "<b>Max Sharpe</b> has the best return per unit of risk. <b>Min Variance</b> has "
            "the lowest risk and ignores return forecasts, which makes it more reliable.",
            "These numbers use the same data the optimizer learned from, so they are a best "
            "case. The backtest on the next page is the honest test.",
        ]))
        story.append(PageBreak())

        # ===========================================================
        # Page 5: Backtest
        # ===========================================================
        story.append(_SectionStart(marks, "backtest"))
        story.append(_p("4. Did it hold up?", "h1"))
        if test is None:
            story.append(_p(f"The backtest could not run: {escape(test_note)}", "lead"))
        else:
            res, growth = test["results"], test["growth"]
            final = growth.iloc[-1]
            beat = final["Max Sharpe"] > final["Equal Weight"]
            story.append(_p(
                f"Max Sharpe turned 100 kr into {final['Max Sharpe']:.0f} kr. Equal weight "
                f"made {final['Equal Weight']:.0f} kr, so the optimizer "
                f"{'won' if beat else 'lost'} over the {test['test_months']}-month test.",
                "lead"))
            story.append(chart("backtest", scales["backtest"],
                               lambda s: _backtest_chart(growth, s)))

            story.append(_p("Predicted vs actual", "h2"))
            rows = [["Portfolio", "Predicted return", "Actual return", "Gap",
                     "Predicted volatility", "Actual volatility"]]
            for name in portfolios:
                pr = res.loc[name, "Predicted Return"]
                ar = res.loc[name, "Actual Return"]
                rows.append([name, _pct(pr), _pct(ar), f"{(ar - pr) * 100:+.1f} pts",
                             _pct(res.loc[name, "Predicted Volatility"]),
                             _pct(res.loc[name, "Actual Volatility"])])
            story.append(_table(rows, [34 * mm, 27 * mm, 25 * mm, 22 * mm, 32 * mm,
                                       CONTENT_WIDTH - 140 * mm], number_cols=(1, 2, 3, 4, 5)))

            story.append(_p("Weights chosen from the training data", "h2"))
            w = test["weights"]
            w = w[(w > 0.0005).any(axis=1)]
            rows = [["Stock"] + list(w.columns)]
            for name, r in w.iterrows():
                rows.append([name] + [_pct(v) if v >= 0.0005 else "-" for v in r])
            other = (CONTENT_WIDTH - 45 * mm) / len(w.columns)
            story.append(_table(rows, [45 * mm] + [other] * len(w.columns),
                                number_cols=tuple(range(1, len(w.columns) + 1))))

            story.append(_note_box("How to read this", [
                "The weights were chosen using only data before the test started, then kept "
                "fixed. This is the honest way to judge a strategy.",
                "A large <b>gap</b> between predicted and actual return means past returns "
                "were a poor guide to the future.",
                "<b>Worst drop</b>: 0 to -10% is mild, -10 to -20% a correction, below -20% a "
                "bear market.",
                escape(settings.benchmark_note(config["benchmark"])),
                "A short test period is one data point, not proof.",
            ]))

        return story

    # --- Build the PDF ---
    def render(scales):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=22 * mm,
            title="Portfolio report", author="Python Stock Analyst",
        )
        pages_seen, marks = [], {}
        footer = _footer(created, pages_seen)
        doc.build(make_story(scales, marks), onFirstPage=footer, onLaterPages=footer)
        return buffer.getvalue(), marks, max(pages_seen)

    def too_long(marks, last_page):
        """Sections that spill onto more than one page."""
        order = ["stocks", "corr", "mix", "backtest"]
        starts = [marks.get(name, 0) for name in order] + [last_page + 1]
        return [name for i, name in enumerate(order) if starts[i + 1] - starts[i] > 1]

    # Each section should fit on its own page. With many stocks the tables get
    # longer, so shrink the charts of a section that spills over, step by step.
    scales = {"stocks": 1.0, "corr": 1.0, "mix": 1.0, "backtest": 1.0}
    pdf, marks, last_page = render(scales)
    for _ in range(5):
        shrink = [s for s in too_long(marks, last_page) if scales[s] > 0.55]
        if not shrink:
            break
        for section in shrink:
            scales[section] = round(scales[section] - 0.1, 2)
        pdf, marks, last_page = render(scales)
    return pdf


def save_report(pdf_bytes, folder=None):
    """Save the report as reports/portfolio_report_YYYY-MM-DD.pdf and return the path."""
    folder = folder or os.path.join(settings.PROJECT_ROOT, "reports")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"portfolio_report_{date.today().isoformat()}.pdf")
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path
