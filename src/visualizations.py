import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VISUALIZATION_DIR = os.path.join(
    PROJECT_ROOT,
    "visualizations"
)

os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def correlation_heatmap(correlation):

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="RdYlGn",
        center=0,
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Stock Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "correlation_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )


def risk_return_chart(results):

    plt.figure(figsize=(11, 7))

    plt.scatter(
        results["Annual Volatility"] * 100,
        results["Annual Return"] * 100,
        s=100,
        alpha=0.8
    )

    for ticker in results.index:
        plt.annotate(
            ticker,
            (
                results.loc[ticker, "Annual Volatility"] * 100,
                results.loc[ticker, "Annual Return"] * 100
            ),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    plt.title("Risk vs Return")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "risk_return.png"),
        dpi=300,
        bbox_inches="tight"
    )


def stock_performance_chart(prices):

    normalized_prices = prices / prices.iloc[0] * 100

    plt.figure(figsize=(13, 7))

    for ticker in normalized_prices.columns:
        plt.plot(
            normalized_prices.index,
            normalized_prices[ticker],
            label=ticker
        )

    plt.title("Stock Performance (Starting Value = 100)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "stock_performance.png"),
        dpi=300,
        bbox_inches="tight"
    )


def efficient_frontier_chart(frontier, statistics, portfolios):
    """
    frontier:   DataFrame from portfolio.efficient_frontier()
    statistics: DataFrame from calculations.calculate_statistics()
    portfolios: dict like {"Min Variance": portfolio_dict, ...}
    """

    plt.figure(figsize=(11, 7))

    # The efficient frontier line
    plt.plot(
        frontier["Volatility"] * 100,
        frontier["Return"] * 100,
        linewidth=2.5,
        label="Efficient Frontier"
    )

    # The individual stocks
    plt.scatter(
        statistics["Annual Volatility"] * 100,
        statistics["Annual Return"] * 100,
        s=70,
        alpha=0.7,
        color="gray",
        label="Individual Stocks"
    )

    for ticker in statistics.index:
        plt.annotate(
            ticker,
            (
                statistics.loc[ticker, "Annual Volatility"] * 100,
                statistics.loc[ticker, "Annual Return"] * 100
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9
        )

    # Highlighted portfolios
    markers = ["*", "D", "s", "^"]
    for (name, p), marker in zip(portfolios.items(), markers):
        plt.scatter(
            p["volatility"] * 100,
            p["return"] * 100,
            s=250 if marker == "*" else 110,
            marker=marker,
            edgecolors="black",
            zorder=5,
            label=name
        )

    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "efficient_frontier.png"),
        dpi=300,
        bbox_inches="tight"
    )


def backtest_growth_chart(growth):
    """Value of 100 kr invested at the start of the test period."""

    plt.figure(figsize=(12, 7))

    for name in growth.columns:
        style = "--" if name not in ("Max Sharpe", "Min Variance", "Equal Weight") else "-"
        plt.plot(
            growth.index,
            growth[name],
            linestyle=style,
            linewidth=2.2,
            label=f"{name} ({growth[name].iloc[-1] - 100:+.1f}%)"
        )

    plt.axhline(100, color="gray", linewidth=1, alpha=0.6)
    plt.title("Backtest: Growth of 100 kr in the Test Period")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (kr)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "backtest_growth.png"),
        dpi=300,
        bbox_inches="tight"
    )


def predicted_vs_actual_chart(results):
    """Compare what the optimizer expected with what actually happened."""

    names = [n for n in results.index if not pd.isna(results.loc[n, "Predicted Return"])]
    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, metric in zip(axes, ["Return", "Volatility"]):
        predicted = results.loc[names, f"Predicted {metric}"].astype(float) * 100
        actual = results.loc[names, f"Actual {metric}"].astype(float) * 100

        ax.bar(x - width / 2, predicted, width, label="Predicted (training)")
        ax.bar(x + width / 2, actual, width, label="Actual (test)")

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(f"Annualized {metric} (%)")
        ax.set_title(f"{metric}: Predicted vs Actual")
        ax.axhline(0, color="gray", linewidth=1)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(VISUALIZATION_DIR, "predicted_vs_actual.png"),
        dpi=300,
        bbox_inches="tight"
    )


def show_all():
    """Open every chart in its own window. The program waits until you close them."""
    plt.show()