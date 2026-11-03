"""
Interactive portfolio dashboard.

Run from the project folder:
    streamlit run dashboard.py

It uses the same calculations as main.py, but lets you change the
settings with sliders and explains each result in plain language.
"""

import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import backtest
from src import calculations
from src import currency
from src import data
from src import portfolio
from src import report
from src import settings


# If started with plain "python dashboard.py" (or VS Code's Run button),
# restart it properly with "streamlit run dashboard.py".
if __name__ == "__main__":
    from streamlit import runtime

    if not runtime.exists():
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

# Hide harmless pandas deprecation warnings coming from inside yfinance
warnings.filterwarnings("ignore", message=".*inplace keyword.*")


# ---------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------

st.set_page_config(page_title="Portfolio dashboard", page_icon="📈", layout="wide")

# Bright palette. All colors live here, so it's easy to change.
BACKGROUND = "#F6F8F7"           # near-white (matches config.toml)
SURFACE = "#E7EDEB"              # slightly darker panels
INK = "#1E3442"                  # main text
MUTED = "#5E6B73"                # secondary text
GRANITE = "#64748B"
GRID = "#DCE3E1"

PORTFOLIO_COLORS = {
    "Max Sharpe": "#1F6FEB",     # bright blue
    "Min Variance": "#0F9D58",   # emerald
    "Equal Weight": "#EA580C",   # orange
}
TARGET_COLOR = "#E8204F"         # rose
PREDICTED_COLOR = "#A9B6BD"      # light grey bars for "predicted"

STOCK_COLORS = [
    "#1F6FEB",  # blue
    "#0F9D58",  # emerald
    "#EA580C",  # orange
    "#E8204F",  # rose
    "#8B3DF0",  # violet
    "#0894B8",  # cyan
    "#E0369A",  # pink
    "#B5651D",  # copper
    "#4D8C0A",  # leaf green
    "#4C51E8",  # indigo
]

st.markdown(
    f"""
    <style>
    .headline {{
        font-family: "Source Serif 4", Georgia, serif;
        font-size: 1.65rem;
        line-height: 1.3;
        font-weight: 500;
        color: {INK};
        max-width: 48rem;
        margin: 0.4rem 0 0.3rem 0;
    }}
    .subline {{
        color: {MUTED};
        font-size: 1.02rem;
        max-width: 48rem;
        margin-bottom: 1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def headline(text, subline=None):
    """The plain-language answer at the top of each tab."""
    st.markdown(f'<p class="headline">{text}</p>', unsafe_allow_html=True)
    if subline:
        st.markdown(f'<p class="subline">{subline}</p>', unsafe_allow_html=True)


def style_figure(fig, height=440):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_family="Public Sans, sans-serif"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def join_names(items):
    """['A', 'B', 'C'] -> 'A, B and C'"""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def pct(x, decimals=1):
    return f"{x * 100:.{decimals}f}%"


# ---------------------------------------------------------------
# Data (cached for an hour so the sliders feel instant)
# ---------------------------------------------------------------

display_name = settings.display_name


@st.cache_data(ttl=3600, show_spinner="Downloading prices and converting to SEK...")
def load_prices(tickers, start, end):
    return data.load_prices(tickers, start, end)


@st.cache_data(ttl=3600, show_spinner=False)
def load_benchmark(benchmark, start, end):
    return data.load_benchmark(benchmark, start, end)


@st.cache_data(show_spinner="Looking up company names...")
def load_names(tickers):
    return settings.lookup_names(tickers)


# ---------------------------------------------------------------
# Sidebar: settings (start from settings.json, same as main.py)
# ---------------------------------------------------------------

config = settings.load()
saved_tickers = config["tickers"]

with st.sidebar:
    st.header("Settings")
    st.caption("Starts from your saved defaults, the same ones main.py uses.")

    chosen = st.multiselect(
        "Stocks",
        options=list(dict.fromkeys(list(settings.STOCK_NAMES) + saved_tickers)),
        default=saved_tickers,
        format_func=lambda t: f"{display_name(t)} ({t})",
    )
    extra = st.text_input(
        "Add other Yahoo tickers",
        placeholder="e.g. NIBE-B.ST, KINV-B.ST",
        help="Any ticker that works on finance.yahoo.com, separated by commas. "
             "Click Save as default to keep them.",
    )
    extra_tickers = [t.strip().upper() for t in extra.split(",") if t.strip()]
    tickers = tuple(dict.fromkeys(chosen + extra_tickers))
    load_names(tickers)   # readable names from Yahoo for unknown tickers

    start_date = st.date_input("Start date", value=pd.Timestamp(config["start_date"]))
    end_date = pd.Timestamp(settings.end_date())

    st.subheader("Rules")
    n = max(len(tickers), 2)
    min_cap = math.ceil(100 / n)
    max_weight = st.slider(
        "Max weight per stock (%)",
        min_value=min_cap, max_value=100,
        value=min(100, max(round(config["max_weight"] * 100), min_cap)), step=1,
        help=f"With {n} stocks the lowest possible cap is {min_cap}%, "
             "otherwise you can't invest 100%.",
    ) / 100
    risk_free = st.slider(
        "Risk-free rate (%)", 0.0, 6.0,
        min(6.0, max(0.0, round(config["risk_free_rate"] * 100, 2))), 0.25,
        help="What you'd earn with no risk, e.g. a savings account.",
    ) / 100
    target_return = st.slider(
        "Target return (%)", 0.0, 40.0,
        min(40.0, max(0.0, round(config["target_return"] * 100, 1))), 0.5,
        help="Used for the lowest-risk portfolio that reaches this return.",
    ) / 100

    st.subheader("Compare with")
    benchmark_options = list(dict.fromkeys(list(settings.BENCHMARKS) + [config["benchmark"]]))
    benchmark = st.selectbox(
        "Benchmark index",
        options=benchmark_options,
        index=benchmark_options.index(config["benchmark"]),
        format_func=settings.benchmark_name,
        help="The index the backtest compares your portfolios with. "
             "Pick one that matches your stocks: OMXS30 for Swedish stocks, "
             "a world index for a global mix.",
    )
    if benchmark in settings.BENCHMARKS:
        st.caption(settings.BENCHMARKS[benchmark]["description"].capitalize() + ".")

    st.subheader("Estimates")
    shrink = st.toggle(
        "Steadier estimates (Ledoit-Wolf)",
        value=bool(config["shrinkage"]),
        help="Blends the covariance estimate toward a simpler one, so the optimizer "
             "reacts less to noise in the data. Usually gives more spread-out, more "
             "stable weights. Compare the backtests with it on and off.",
    )


# ---------------------------------------------------------------
# Load and calculate
# ---------------------------------------------------------------

st.title("Portfolio dashboard")

if len(tickers) < 2:
    st.info("Pick at least two stocks in the sidebar to build a portfolio.")
    st.stop()

loaded = load_prices(tickers, str(start_date), str(end_date.date()))
prices = loaded["prices"]
prices_local = loaded["local_prices"]
currency_info = loaded["currency_info"]
missing = loaded["missing"]

if loaded["local_prices"].empty:
    st.error(
        "Yahoo Finance returned no prices. Check your internet connection "
        "and that the tickers exist on finance.yahoo.com."
    )
    st.stop()

if missing:
    st.warning(f"No data for {', '.join(missing)}. They were left out.")

if loaded["not_converted"]:
    st.warning(
        f"Couldn't download exchange rates for {', '.join(loaded['not_converted'])}, "
        "so they were left out."
    )

prices = prices.rename(columns=display_name)
prices_local = prices_local.rename(columns=display_name)
currency_of = {display_name(t): cur for t, cur in currency_info["Currency"].items()}

if prices.shape[1] < 2:
    st.error("At least two stocks need price data. Pick other stocks in the sidebar.")
    st.stop()

if max_weight * prices.shape[1] < 1:
    max_weight = math.ceil(100 / prices.shape[1]) / 100

try:
    analysis = calculations.calculate_all(prices, shrink=shrink)
except ValueError as error:
    st.error(str(error))
    st.stop()

returns = analysis["returns"]
stats = analysis["statistics"]
annual_cov = analysis["annual_covariance"]
corr = analysis["correlation"]
mean_returns = stats["Annual Return"]
names = list(stats.index)
color_of = {name: STOCK_COLORS[i % len(STOCK_COLORS)] for i, name in enumerate(names)}

max_sharpe = portfolio.max_sharpe_portfolio(
    mean_returns, annual_cov, risk_free_rate=risk_free, max_weight=max_weight)
min_var = portfolio.min_variance_portfolio(
    mean_returns, annual_cov, risk_free_rate=risk_free, max_weight=max_weight)
equal = portfolio.equal_weight_portfolio(
    mean_returns, annual_cov, risk_free_rate=risk_free)
frontier = portfolio.efficient_frontier(
    mean_returns, annual_cov, max_weight=max_weight)

target, target_error = None, None
try:
    target = portfolio.min_variance_portfolio(
        mean_returns, annual_cov, target_return=target_return,
        risk_free_rate=risk_free, max_weight=max_weight)
except ValueError as error:
    target_error = str(error)

portfolios = {"Max Sharpe": max_sharpe, "Min Variance": min_var, "Equal Weight": equal}

foreign = sorted({c for c in currency_of.values() if c != currency.BASE_CURRENCY})
st.caption(
    f"{len(names)} stocks, monthly data from {returns.index[0]:%B %Y} "
    f"to {returns.index[-1]:%B %Y} ({len(returns)} months). "
    f"No stock above {max_weight:.0%}, no short selling. "
    + (f"All values in SEK, converted from {', '.join(foreign)}."
       if foreign else "All values in SEK.")
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. The stocks",
    "2. How they move together",
    "3. The best mix",
    "4. Did it hold up?",
    "5. Year by year",
])


# ---------------------------------------------------------------
# Tab 1: the stocks
# ---------------------------------------------------------------

with tab1:
    best = stats["Annual Return"].idxmax()
    worst = stats["Annual Return"].idxmin()
    riskiest = stats["Annual Volatility"].idxmax()
    calmest = stats["Annual Volatility"].idxmin()

    headline(
        f"{best} grew the most, about {pct(stats.loc[best, 'Annual Return'])} a year. "
        f"{riskiest} swung the most, with {pct(stats.loc[riskiest, 'Annual Volatility'])} volatility.",
        f"{worst} had the lowest return ({pct(stats.loc[worst, 'Annual Return'])} a year) "
        f"and {calmest} was the calmest ({pct(stats.loc[calmest, 'Annual Volatility'])} volatility).",
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("What 100 kr became")
        growth = prices.loc[:, names].dropna()
        growth = growth / growth.iloc[0] * 100
        fig = go.Figure()
        for name in names:
            fig.add_trace(go.Scatter(
                x=growth.index, y=growth[name], name=name, mode="lines",
                line=dict(color=color_of[name], width=2),
                hovertemplate="%{y:.0f} kr<extra>" + name + "</extra>",
            ))
        fig.add_hline(y=100, line_color=GRANITE, line_width=1, opacity=0.5)
        fig.update_yaxes(title="Value (kr)")
        st.plotly_chart(style_figure(fig), key="growth_all")

    with right:
        st.subheader("Return vs risk")
        fig = go.Figure(go.Scatter(
            x=stats["Annual Volatility"] * 100,
            y=stats["Annual Return"] * 100,
            mode="markers+text",
            text=names,
            textposition="top center",
            marker=dict(size=13, color=[color_of[n] for n in names]),
            hovertemplate="%{text}<br>Return %{y:.1f}%<br>Volatility %{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(title="Volatility per year (%)")
        fig.update_yaxes(title="Return per year (%)")
        st.plotly_chart(style_figure(fig), key="risk_return")

    # Return in the stock's own currency, to show what the currency added or removed
    local_returns = calculations.calculate_returns(prices_local[names]).mean() * 12

    table = pd.DataFrame({
        "Currency": pd.Series(currency_of).reindex(names),
        "Return per year": stats["Annual Return"] * 100,
        "Compounded per year": stats["Compounded Return"] * 100,
        "Currency effect": (stats["Annual Return"] - local_returns) * 100,
        "Volatility per year": stats["Annual Volatility"] * 100,
        "Return per unit of risk": (stats["Annual Return"] - risk_free) / stats["Annual Volatility"],
    }).sort_values("Return per unit of risk", ascending=False)

    st.dataframe(
        table,
        column_config={
            "Return per year": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Average monthly return × 12, in SEK. This is what the optimizer uses."),
            "Compounded per year": st.column_config.NumberColumn(
                format="%.1f%%",
                help="What the money actually grew by each year, in SEK."),
            "Currency effect": st.column_config.NumberColumn(
                format="%.1f%%",
                help="Return in SEK minus return in the stock's own currency. "
                     "Negative means the currency weakened against the krona."),
            "Volatility per year": st.column_config.NumberColumn(format="%.1f%%"),
            "Return per unit of risk": st.column_config.NumberColumn(
                format="%.2f", help="Sharpe ratio: (return − risk-free rate) ÷ volatility"),
        },
    )

    with st.expander("How to read this"):
        st.markdown(
            "- **Return per year** is the average monthly return × 12, in Swedish kronor. "
            "The optimizer uses this.\n"
            "- **Compounded per year** is what your money actually grew by. For volatile "
            "stocks it's clearly lower than the average, because losses hurt more than "
            "gains help: +50% then −50% leaves 75 kr of 100.\n"
            "- **Currency effect** shows how much exchange rates changed the return. "
            "A foreign stock can rise in its own currency and still lose money in SEK "
            "if that currency falls against the krona.\n"
            "- **Volatility** is how much the price swings, the standard deviation of "
            "monthly returns × √12. Around 20% is normal for a large Swedish company.\n"
            "- In the scatter chart, **up and to the left is better**: more return for less risk.\n"
            "- These are past numbers. They describe what happened, not what will happen."
        )


# ---------------------------------------------------------------
# Tab 2: correlation and diversification
# ---------------------------------------------------------------

with tab2:
    pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    most_alike = pairs.idxmax()
    most_different = pairs.idxmin()
    avg_single_vol = stats["Annual Volatility"].mean()

    headline(
        f"Holding all {len(names)} stocks equally gives {pct(equal['volatility'])} volatility, "
        f"against {pct(avg_single_vol)} for the average single stock.",
        f"That drop is the diversification effect. {most_alike[0]} and {most_alike[1]} move "
        f"the most alike ({pairs.max():.2f}), while {most_different[0]} and "
        f"{most_different[1]} move the most differently ({pairs.min():.2f}).",
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Correlation between the stocks")
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1, zmax=1,
            colorscale=[[0, "#16A34A"], [0.5, BACKGROUND], [1, "#DC2626"]],
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            hovertemplate="%{y} & %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="", thickness=12),
        ))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_figure(fig, height=480), key="heatmap")

    with right:
        st.subheader("Risk: alone vs together")
        bars = pd.Series({
            "Average single stock": avg_single_vol,
            "Equal weight portfolio": equal["volatility"],
            "Min variance portfolio": min_var["volatility"],
        }) * 100
        fig = go.Figure(go.Bar(
            x=bars.values, y=bars.index, orientation="h",
            marker_color=[GRANITE, PORTFOLIO_COLORS["Equal Weight"], PORTFOLIO_COLORS["Min Variance"]],
            text=[f"{v:.1f}%" for v in bars.values], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(title="Volatility per year (%)", range=[0, bars.max() * 1.25])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_figure(fig, height=300), key="div_bars")

        st.metric(
            "Average correlation between pairs",
            f"{pairs.mean():.2f}",
            help="1 = always move together, 0 = unrelated, −1 = always opposite.",
        )

    with st.expander("How to read this"):
        st.markdown(
            "- **Correlation** runs from −1 to +1. Red squares are stocks that move together "
            "(less diversification), green squares move apart (more diversification).\n"
            "- Combining stocks with **low correlation** lowers risk, because when one falls "
            "the other often doesn't. That's why the portfolio bars are shorter than the "
            "single-stock bar.\n"
            "- Stocks in the same sector, like two banks, usually have high correlation and "
            "diversify each other less."
        )


# ---------------------------------------------------------------
# Tab 3: the optimized portfolios
# ---------------------------------------------------------------

with tab3:
    used = max_sharpe["weights"][max_sharpe["weights"] > 0.005]
    top = used.nlargest(min(3, len(used)))

    headline(
        f"The best return for the risk: about {pct(max_sharpe['return'])} a year "
        f"at {pct(max_sharpe['volatility'])} volatility, using {len(used)} of {len(names)} stocks.",
        f"{join_names(top.index)} make up {pct(top.sum(), 0)} of it. These numbers come from the same data the optimizer "
        f"learned from, so they are a best case. Tab 4 tests them honestly.",
    )
    if shrink:
        st.caption(
            f"Using Ledoit-Wolf shrinkage: the covariance is blended "
            f"{analysis['shrinkage']:.0%} toward a simpler estimate for steadier weights."
        )

    cols = st.columns(3)
    for col, (name, p) in zip(cols, portfolios.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                a, b, c = st.columns(3)
                a.metric("Return", pct(p["return"]))
                b.metric("Volatility", pct(p["volatility"]))
                c.metric("Sharpe", f"{p['sharpe']:.2f}")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("The efficient frontier")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frontier["Volatility"] * 100, y=frontier["Return"] * 100,
            mode="lines", name="Efficient frontier",
            line=dict(color=INK, width=3),
            hovertemplate="Return %{y:.1f}%<br>Volatility %{x:.1f}%<extra>Frontier</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=stats["Annual Volatility"] * 100, y=stats["Annual Return"] * 100,
            mode="markers+text", name="Single stocks", text=names,
            textposition="top center", textfont=dict(color=GRANITE, size=11),
            marker=dict(size=9, color=GRANITE, opacity=0.6),
            hovertemplate="%{text}<br>Return %{y:.1f}%<br>Volatility %{x:.1f}%<extra></extra>",
        ))
        markers = {"Max Sharpe": "star", "Min Variance": "diamond", "Equal Weight": "square"}
        for name, p in portfolios.items():
            fig.add_trace(go.Scatter(
                x=[p["volatility"] * 100], y=[p["return"] * 100], mode="markers", name=name,
                marker=dict(size=18 if name == "Max Sharpe" else 13, symbol=markers[name],
                            color=PORTFOLIO_COLORS[name], line=dict(color=BACKGROUND, width=1.5)),
                hovertemplate=name + "<br>Return %{y:.1f}%<br>Volatility %{x:.1f}%<extra></extra>",
            ))
        if target is not None:
            fig.add_trace(go.Scatter(
                x=[target["volatility"] * 100], y=[target["return"] * 100], mode="markers",
                name=f"Target {target_return:.0%}",
                marker=dict(size=13, symbol="triangle-up", color=TARGET_COLOR,
                            line=dict(color=BACKGROUND, width=1.5)),
            ))
        fig.update_xaxes(title="Volatility per year (%)")
        fig.update_yaxes(title="Return per year (%)")
        st.plotly_chart(style_figure(fig, height=500), key="frontier")

    with right:
        st.subheader("Where the money goes")
        weights = pd.DataFrame({n: p["weights"] for n, p in portfolios.items()})
        if target is not None:
            weights[f"Target {target_return:.0%}"] = target["weights"]
        weights = weights.loc[weights.max(axis=1).sort_values().index]

        fig = go.Figure()
        for name in weights.columns:
            fig.add_trace(go.Bar(
                y=weights.index, x=weights[name] * 100, name=name, orientation="h",
                marker_color=PORTFOLIO_COLORS.get(name, TARGET_COLOR),
                hovertemplate="%{y}: %{x:.1f}%<extra>" + name + "</extra>",
            ))
        fig.add_vline(x=max_weight * 100, line_dash="dot", line_color=TARGET_COLOR,
                      annotation_text=f"Cap {max_weight:.0%}", annotation_position="top")
        fig.update_layout(barmode="group", bargap=0.25)
        fig.update_xaxes(title="Weight (%)")
        style_figure(fig, height=500)
        fig.update_layout(margin=dict(l=110))   # room for long stock names
        st.plotly_chart(fig, key="weights")

    if target is not None:
        st.success(
            f"To reach {target_return:.0%} a year with the least risk, the volatility is "
            f"{pct(target['volatility'])}. See the red bars for the weights."
        )
    else:
        st.info(target_error)

    with st.expander("How to read this"):
        st.markdown(
            "- The **efficient frontier** is the best possible return for each level of risk. "
            "Every portfolio below it gets less return for the same risk.\n"
            "- **Max Sharpe** is the point with the best return per unit of risk.\n"
            "- **Min Variance** is the calmest possible mix. It ignores expected returns "
            "and only uses volatility and correlation, which makes it more reliable.\n"
            "- **Equal Weight** is the simple benchmark: the same amount in every stock.\n"
            "- The dotted line is your weight cap. Bars that touch it show where the optimizer "
            "wanted to go further."
        )


# ---------------------------------------------------------------
# Tab 4: out-of-sample backtest
# ---------------------------------------------------------------

train_end_to_save = config["train_end"]
bench = load_benchmark(benchmark, str(start_date), str(end_date.date()))

with tab4:
    months = list(returns.index)
    options = months[23:-6]

    if not options:
        st.info(
            "The backtest needs at least 30 months of data: 24 to choose the weights "
            "and 6 to test them. Pick an earlier start date in the sidebar."
        )
    else:
        saved_month = pd.Timestamp(config["train_end"]).strftime("%Y-%m")
        default = next((m for m in options if m.strftime("%Y-%m") == saved_month),
                       options[len(options) // 2])
        train_end = st.select_slider(
            "Choose weights using data up to",
            options=options, value=default,
            format_func=lambda d: d.strftime("%b %Y"),
            help="The optimizer only sees data up to this month. "
                 "Everything after is the test.",
        )

        train_end_to_save = (train_end + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        test = backtest.run_backtest(
            returns, train_end=train_end, risk_free_rate=risk_free,
            max_weight=max_weight, benchmark_returns=bench, shrink=shrink,
        )
        res = test["results"]
        growth = test["growth"]
        final = growth.iloc[-1]

        ms_pred = res.loc["Max Sharpe", "Predicted Return"]
        ms_act = res.loc["Max Sharpe", "Actual Return"]
        winner = final.idxmax()
        beat = final["Max Sharpe"] > final["Equal Weight"]

        headline(
            f"Max Sharpe turned 100 kr into {final['Max Sharpe']:.0f} kr. "
            f"Equal weight made {final['Equal Weight']:.0f} kr, so the optimizer "
            f"{'won' if beat else 'lost'}.",
            f"The optimizer predicted {pct(ms_pred)} a year and got {pct(ms_act)}. "
            f"Over the {test['test_months']}-month test, the best result came from {winner}.",
        )

        cols = st.columns(len(final))
        for col, name in zip(cols, final.index):
            with col:
                with st.container(border=True):
                    st.metric(
                        name, f"{final[name]:.0f} kr",
                        delta=f"{final[name] - 100:+.1f}%",
                        help="Value of 100 kr at the end of the test period.",
                    )
                    st.caption(f"Worst drop: {pct(res.loc[name, 'Max Drawdown'])}")

        left, right = st.columns([3, 2])

        with left:
            st.subheader("100 kr invested at the start of the test")
            fig = go.Figure()
            for name in growth.columns:
                fig.add_trace(go.Scatter(
                    x=growth.index, y=growth[name], name=name, mode="lines",
                    line=dict(color=PORTFOLIO_COLORS.get(name, GRANITE), width=2.5,
                              dash="solid" if name in portfolios else "dash"),
                    hovertemplate="%{y:.0f} kr<extra>" + name + "</extra>",
                ))
            fig.add_hline(y=100, line_color=GRANITE, line_width=1, opacity=0.5)
            fig.update_yaxes(title="Value (kr)")
            st.plotly_chart(style_figure(fig), key="bt_growth")

        with right:
            st.subheader("Predicted vs actual return")
            names_bt = list(portfolios.keys())
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=names_bt, y=res.loc[names_bt, "Predicted Return"].astype(float) * 100,
                name="Predicted (training)", marker_color=PREDICTED_COLOR,
                hovertemplate="%{x}: %{y:.1f}%<extra>Predicted</extra>",
            ))
            fig.add_trace(go.Bar(
                x=names_bt, y=res.loc[names_bt, "Actual Return"].astype(float) * 100,
                name="Actual (test)",
                marker_color=[PORTFOLIO_COLORS[n] for n in names_bt],
                hovertemplate="%{x}: %{y:.1f}%<extra>Actual</extra>",
            ))
            fig.update_layout(barmode="group")
            fig.update_yaxes(title="Return per year (%)")
            st.plotly_chart(style_figure(fig), key="bt_pred")

        with st.expander("Full results table"):
            show = res.copy()
            for c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce")
            pct_cols = [c for c in show.columns if "Sharpe" not in c]
            show[pct_cols] = show[pct_cols] * 100
            st.dataframe(
                show,
                column_config={
                    c: st.column_config.NumberColumn(format="%.2f" if "Sharpe" in c else "%.1f%%")
                    for c in show.columns
                },
            )

        with st.expander("Weights chosen from the training data"):
            w = test["weights"]
            w = w[(w > 0.001).any(axis=1)] * 100
            st.dataframe(
                w,
                column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in w.columns},
            )

        with st.expander("How to read this"):
            st.markdown(
                "- The weights were picked using **only** the data before the test started, "
                "then kept fixed. This is the honest way to judge a strategy, because in real "
                "life you never know the future.\n"
                "- **Predicted vs actual** shows how good the optimizer's forecast was. "
                "A big gap means the past was a poor guide.\n"
                "- **Worst drop** (max drawdown) is the biggest fall from a peak. It's often "
                "what investors feel most.\n"
                f"- **{settings.benchmark_name(benchmark)}**: "
                f"{settings.benchmark_note(benchmark)}\n"
                "- A short test period is one data point, not proof. Move the slider above "
                "to see whether the result holds for other split dates, or look at tab 5."
            )


# ---------------------------------------------------------------
# Tab 5: rolling backtest (new weights every year)
# ---------------------------------------------------------------

with tab5:
    try:
        rolling = backtest.rolling_backtest(
            returns, risk_free_rate=risk_free, max_weight=max_weight,
            benchmark_returns=bench, shrink=shrink,
        )
    except ValueError as error:
        rolling = None
        st.info(str(error))

    if rolling is not None:
        n_years = rolling["n_years"]
        wins = rolling["wins"]
        summary = rolling["summary"]
        r_growth = rolling["growth"]

        headline(
            f"Re-picking the weights every year, Max Sharpe beat equal weight in "
            f"{wins['Max Sharpe']} of {n_years} years, and Min Variance in "
            f"{wins['Min Variance']} of {n_years}.",
            f"Over all the test years combined, Max Sharpe grew "
            f"{pct(summary.loc['Max Sharpe', 'Compounded Return'])} a year, Min Variance "
            f"{pct(summary.loc['Min Variance', 'Compounded Return'])} and equal weight "
            f"{pct(summary.loc['Equal Weight', 'Compounded Return'])}.",
        )

        final_r = r_growth.iloc[-1]
        cols = st.columns(len(final_r))
        for col, name in zip(cols, final_r.index):
            with col:
                with st.container(border=True):
                    st.metric(name, f"{final_r[name]:.0f} kr",
                              delta=f"{final_r[name] - 100:+.1f}%",
                              help="Value of 100 kr after all the test years.")
                    st.caption(f"Worst drop: {pct(summary.loc[name, 'Max Drawdown'])}")

        left, right = st.columns([3, 2])
        with left:
            st.subheader(f"100 kr from {r_growth.index[0]:%b %Y}")
            fig = go.Figure()
            for name in r_growth.columns:
                fig.add_trace(go.Scatter(
                    x=r_growth.index, y=r_growth[name], name=name, mode="lines",
                    line=dict(color=PORTFOLIO_COLORS.get(name, GRANITE), width=2.5,
                              dash="solid" if name in portfolios else "dash"),
                    hovertemplate="%{y:.0f} kr<extra>" + name + "</extra>",
                ))
            fig.add_hline(y=100, line_color=GRANITE, line_width=1, opacity=0.5)
            fig.update_yaxes(title="Value (kr)")
            st.plotly_chart(style_figure(fig), key="rolling_growth")

        with right:
            st.subheader("Return each year")
            years = rolling["years"].copy()
            pct_cols = [c for c in years.columns if c != "Best"]
            years[pct_cols] = years[pct_cols] * 100
            st.dataframe(
                years,
                column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                               for c in pct_cols},
            )

        st.subheader("All test years combined")
        combined = pd.DataFrame({
            "Compounded per year": summary["Compounded Return"] * 100,
            "Volatility": summary["Actual Volatility"] * 100,
            "Sharpe ratio": summary["Actual Sharpe"],
            "Worst drop": summary["Max Drawdown"] * 100,
        })
        st.dataframe(
            combined,
            column_config={
                "Compounded per year": st.column_config.NumberColumn(format="%.1f%%"),
                "Volatility": st.column_config.NumberColumn(format="%.1f%%"),
                "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
                "Worst drop": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

        with st.expander("How to read this"):
            st.markdown(
                "- Every January, the weights are chosen again using all data up to the "
                "end of the previous year, then held for that year. This is how you would "
                "actually run the strategy.\n"
                "- One result per year is more convincing than tab 4's single test: a "
                "strategy that wins most years is more likely to be real than lucky.\n"
                "- **Compounded per year** is what the money actually grew by over all "
                "the test years together.\n"
                "- Turn **Steadier estimates (Ledoit-Wolf)** on and off in the sidebar to "
                "see whether it improves the results."
            )


# ---------------------------------------------------------------
# Save button (at the end, so it can include the backtest date)
# ---------------------------------------------------------------

current = {
    "tickers": [t for t in tickers if t not in missing],
    "start_date": str(start_date),
    "max_weight": round(max_weight, 4),
    "risk_free_rate": round(risk_free, 4),
    "target_return": round(target_return, 4),
    "train_end": train_end_to_save,
    "benchmark": benchmark,
    "shrinkage": bool(shrink),
}

saved_now = {k: config[k] for k in current}
saved_now["max_weight"] = round(saved_now["max_weight"], 4)
saved_now["risk_free_rate"] = round(saved_now["risk_free_rate"], 4)
saved_now["target_return"] = round(saved_now["target_return"], 4)

if st.session_state.pop("just_saved", False):
    st.toast("Saved. main.py and the dashboard now start with these settings.")

with st.sidebar:
    st.divider()
    if current != saved_now:
        st.caption("You have unsaved changes. main.py still uses the saved settings.")
    if st.button("Save as default", type="primary"):
        settings.save(current)
        st.session_state["just_saved"] = True
        st.rerun()   # reload so everything uses the newly saved settings


# ---------------------------------------------------------------
# PDF report (in the sidebar, under Save as default)
# ---------------------------------------------------------------

signature = json.dumps(current, sort_keys=True)

with st.sidebar:
    st.divider()
    st.subheader("Report")
    if st.button(
        "📄 Generate report", type="primary",
        help="Creates a PDF report with the settings shown on screen and saves "
             "it in the reports folder.",
    ):
        with st.spinner("Building the report..."):
            pdf = report.build_report(prices, current, bench, currency_info)
            path = report.save_report(pdf)
        st.session_state["report"] = {"signature": signature, "pdf": pdf, "path": path}

    stored = st.session_state.get("report")
    if stored and stored["signature"] == signature:
        relative = os.path.relpath(stored["path"], settings.PROJECT_ROOT)
        st.success(f"Saved to {relative}")
        st.download_button(
            "⬇ Download PDF",
            data=stored["pdf"],
            file_name=os.path.basename(stored["path"]),
            mime="application/pdf",
        )
    elif stored:
        st.caption("You changed the settings after the last report. "
                   "Generate it again to update it.")
