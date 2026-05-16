"""Plotly charts for backtest results.

Interactive figures, so the same chart code serves the analysis notebook today
and a web dashboard later. Chart-rendering code is not unit-tested (CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from dispatcher_watts.backtest.engine import BacktestResult

# Consistent palette across charts.
_CHARGE_COLOR = "#e76f51"
_DISCHARGE_COLOR = "#2a9d8f"
_NEUTRAL_COLOR = "#264653"

# Distinct line colors for the multi-strategy comparison chart.
_COMPARISON_COLORS = ("#264653", "#e76f51", "#2a9d8f", "#e9c46a")


def _styled(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def soc_chart(result: BacktestResult) -> go.Figure:
    """State of charge over the backtest period."""
    frame = result.frame
    fig = go.Figure()
    fig.add_scatter(
        x=frame["interval_start"].to_list(),
        y=frame["soc_mwh"].to_list(),
        mode="lines",
        name="State of charge",
        line={"color": _DISCHARGE_COLOR, "width": 1},
    )
    return _styled(
        fig, f"State of charge — {result.strategy_name}", "Time", "State of charge (MWh)"
    )


def dispatch_chart(result: BacktestResult) -> go.Figure:
    """Price over time, with charge and discharge intervals marked."""
    frame = result.frame
    charging = frame.filter(pl.col("action") < 0)
    discharging = frame.filter(pl.col("action") > 0)
    fig = go.Figure()
    fig.add_scatter(
        x=frame["interval_start"].to_list(),
        y=frame["price"].to_list(),
        mode="lines",
        name="Price",
        line={"color": _NEUTRAL_COLOR, "width": 1},
    )
    fig.add_scatter(
        x=charging["interval_start"].to_list(),
        y=charging["price"].to_list(),
        mode="markers",
        name="Charging",
        marker={"color": _CHARGE_COLOR, "size": 4},
    )
    fig.add_scatter(
        x=discharging["interval_start"].to_list(),
        y=discharging["price"].to_list(),
        mode="markers",
        name="Discharging",
        marker={"color": _DISCHARGE_COLOR, "size": 4},
    )
    return _styled(fig, f"Dispatch decisions — {result.strategy_name}", "Time", "Price ($/MWh)")


def cumulative_revenue_chart(result: BacktestResult) -> go.Figure:
    """Cumulative revenue over the backtest period."""
    frame = result.frame
    fig = go.Figure()
    fig.add_scatter(
        x=frame["interval_start"].to_list(),
        y=frame["cumulative_revenue"].to_list(),
        mode="lines",
        name="Cumulative revenue",
        line={"color": _DISCHARGE_COLOR, "width": 2},
        fill="tozeroy",
    )
    return _styled(fig, f"Cumulative revenue — {result.strategy_name}", "Time", "Revenue ($)")


def daily_revenue_chart(result: BacktestResult) -> go.Figure:
    """Revenue earned each day across the backtest period."""
    daily = (
        result.frame.group_by(pl.col("interval_start").dt.date().alias("day"))
        .agg(pl.col("revenue").sum().alias("daily_revenue"))
        .sort("day")
    )
    fig = go.Figure()
    fig.add_bar(
        x=daily["day"].to_list(),
        y=daily["daily_revenue"].to_list(),
        name="Daily revenue",
        marker={"color": _DISCHARGE_COLOR},
    )
    return _styled(fig, f"Daily revenue — {result.strategy_name}", "Day", "Revenue ($)")


def cumulative_revenue_comparison_chart(
    results: dict[str, BacktestResult],
) -> go.Figure:
    """Cumulative revenue of several strategies on one figure.

    `results` maps a strategy label to its backtest result; all results should
    cover the same period. This is the chart that makes the gap between a naive
    strategy and the perfect-foresight ceiling visible at a glance.
    """
    fig = go.Figure()
    for (label, result), color in zip(results.items(), _COMPARISON_COLORS, strict=False):
        frame = result.frame
        fig.add_scatter(
            x=frame["interval_start"].to_list(),
            y=frame["cumulative_revenue"].to_list(),
            mode="lines",
            name=label,
            line={"color": color, "width": 2},
        )
    return _styled(fig, "Cumulative revenue by strategy", "Time", "Revenue ($)")


def save_figure(fig: go.Figure, path: Path) -> Path:
    """Write a figure to `path`: an `.html` interactive file, or a static image.

    Static image formats (`.png`, `.svg`, ...) require the `kaleido` package.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".html":
        fig.write_html(path)
    else:
        fig.write_image(path)
    return path
