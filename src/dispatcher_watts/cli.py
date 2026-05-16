"""Command-line interface for dispatcher-watts."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from dotenv import load_dotenv

from dispatcher_watts.backtest.engine import BacktestResult, run_backtest
from dispatcher_watts.backtest.metrics import (
    BacktestMetrics,
    capture_rate,
    compute_metrics,
)
from dispatcher_watts.battery.model import Battery, BatterySpec
from dispatcher_watts.data.ercot import GridstatusERCOTSource
from dispatcher_watts.data.schemas import ERCOT_HUBS, RTM_INTERVAL_MINUTES
from dispatcher_watts.data.store import (
    cache_path,
    is_cached,
    load_prices,
    save_prices,
    summarize_prices,
)
from dispatcher_watts.reporting.charts import (
    cumulative_revenue_chart,
    daily_revenue_chart,
    dispatch_chart,
    save_figure,
    soc_chart,
)
from dispatcher_watts.strategies.base import Strategy
from dispatcher_watts.strategies.perfect_foresight import PerfectForesightStrategy
from dispatcher_watts.strategies.rolling_avg import RollingAverageStrategy
from dispatcher_watts.strategies.threshold import ThresholdStrategy

# Strategies available to the `backtest` and `compare` commands.
_STRATEGIES: tuple[str, ...] = ("threshold", "rolling-average", "perfect-foresight")

# Load GRIDSTATUS_API_KEY (and any other vars) from a local .env file, if one
# exists. Real environment variables always take precedence.
load_dotenv()

app = typer.Typer(
    help="Backtest battery dispatch strategies against ERCOT market data.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Fetch and inspect ERCOT price data.", no_args_is_help=True)
app.add_typer(data_app, name="data")


def _print_summary(hub: str, year: int, summary: dict[str, float]) -> None:
    typer.echo(f"\n{hub} {year} -- real-time settlement-point prices ($/MWh)")
    typer.echo(f"  intervals          : {int(summary['intervals']):,}")
    typer.echo(
        f"  min / mean / max   : {summary['min']:.2f} / "
        f"{summary['mean']:.2f} / {summary['max']:.2f}"
    )
    typer.echo(f"  median / std       : {summary['median']:.2f} / {summary['std']:.2f}")
    typer.echo(f"  negative intervals : {int(summary['negative_intervals']):,}")


@data_app.command("fetch")
def data_fetch(
    year: int = typer.Option(..., help="Calendar year to fetch, e.g. 2025."),
    hub: str = typer.Option(
        "HB_HOUSTON", help=f"ERCOT trading hub. One of: {', '.join(ERCOT_HUBS)}."
    ),
    force: bool = typer.Option(False, help="Re-fetch even if a cached file exists."),
) -> None:
    """Fetch one year of ERCOT real-time prices for a hub; cache it as parquet."""
    if hub not in ERCOT_HUBS:
        raise typer.BadParameter(f"unknown hub {hub!r}; expected one of {', '.join(ERCOT_HUBS)}")
    path = cache_path(year, hub)
    if path.exists() and not force:
        typer.echo(f"cached file already exists: {path} (use --force to re-fetch)")
        df = load_prices(year, hub)
    else:
        typer.echo(f"fetching {hub} {year} from gridstatus.io ...")
        df = GridstatusERCOTSource().get_rtm_prices(year, hub)
        path = save_prices(df, year, hub)
        typer.echo(f"wrote {df.height:,} intervals to {path}")
    _print_summary(hub, year, summarize_prices(df))


@data_app.command("summary")
def data_summary(
    year: int = typer.Option(..., help="Calendar year."),
    hub: str = typer.Option("HB_HOUSTON", help="ERCOT trading hub."),
) -> None:
    """Print a statistical summary of cached prices for a hub-year."""
    _print_summary(hub, year, summarize_prices(load_prices(year, hub)))


def _build_strategy(
    name: str,
    *,
    spec: BatterySpec,
    interval_minutes: int,
    charge_below: float,
    discharge_above: float,
    window_hours: float,
    band: float,
) -> Strategy:
    try:
        if name == "threshold":
            return ThresholdStrategy(charge_below=charge_below, discharge_above=discharge_above)
        if name == "rolling-average":
            return RollingAverageStrategy(
                window_hours=window_hours,
                interval_minutes=interval_minutes,
                band=band,
            )
        if name == "perfect-foresight":
            return PerfectForesightStrategy(spec, interval_minutes)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raise typer.BadParameter(f"unknown strategy {name!r}; v1 supports: {', '.join(_STRATEGIES)}")


def _print_backtest(hub: str, year: int, result: BacktestResult, metrics: BacktestMetrics) -> None:
    spec = result.spec
    typer.echo(f"\nBacktest -- {result.strategy_name} strategy on {hub} {year}")
    typer.echo(
        f"  battery            : {spec.capacity_mwh:g} MWh / {spec.power_mw:g} MW, "
        f"{spec.round_trip_efficiency:.0%} round-trip"
    )
    typer.echo(f"  total revenue      : ${metrics.total_revenue:,.0f}")
    typer.echo(f"  revenue / MWh-year : ${metrics.revenue_per_mwh_year:,.0f}")
    typer.echo(f"  equivalent cycles  : {metrics.equivalent_full_cycles:,.1f}")
    typer.echo(f"  capacity factor    : {metrics.capacity_factor:.1%}")
    typer.echo(
        f"  intervals c/d/idle : {metrics.intervals_charging:,} / "
        f"{metrics.intervals_discharging:,} / {metrics.intervals_idle:,}"
    )


@app.command("backtest")
def backtest(
    year: int = typer.Option(..., help="Calendar year to backtest (must be cached)."),
    hub: str = typer.Option("HB_HOUSTON", help="ERCOT trading hub."),
    strategy: str = typer.Option(
        "threshold", help="Strategy: threshold, rolling-average, or perfect-foresight."
    ),
    charge_below: float = typer.Option(20.0, help="threshold: charge when price <= this ($/MWh)."),
    discharge_above: float = typer.Option(
        50.0, help="threshold: discharge when price >= this ($/MWh)."
    ),
    window_hours: float = typer.Option(
        24.0, help="rolling-average: trailing window length in hours."
    ),
    band: float = typer.Option(
        0.0, help="rolling-average: no-trade band around the average (0-1)."
    ),
    capacity_mwh: float = typer.Option(1.0, help="Battery energy capacity (MWh)."),
    power_mw: float = typer.Option(0.5, help="Battery power rating (MW)."),
    rte: float = typer.Option(0.87, help="Round-trip efficiency, 0-1."),
) -> None:
    """Backtest a dispatch strategy on cached ERCOT prices for one hub-year."""
    prices = load_prices(year, hub)
    spec = BatterySpec(capacity_mwh=capacity_mwh, power_mw=power_mw, round_trip_efficiency=rte)
    strat = _build_strategy(
        strategy,
        spec=spec,
        interval_minutes=RTM_INTERVAL_MINUTES,
        charge_below=charge_below,
        discharge_above=discharge_above,
        window_hours=window_hours,
        band=band,
    )
    result = run_backtest(prices, Battery(spec), strat)
    _print_backtest(hub, year, result, compute_metrics(result))


@app.command("compare")
def compare(
    year: int = typer.Option(..., help="Calendar year to compare (must be cached)."),
    hub: str = typer.Option("HB_HOUSTON", help="ERCOT trading hub."),
    charge_below: float = typer.Option(20.0, help="threshold charge price ($/MWh)."),
    discharge_above: float = typer.Option(50.0, help="threshold discharge price ($/MWh)."),
    window_hours: float = typer.Option(24.0, help="rolling-average window (hours)."),
    band: float = typer.Option(0.0, help="rolling-average no-trade band (0-1)."),
    capacity_mwh: float = typer.Option(1.0, help="Battery energy capacity (MWh)."),
    power_mw: float = typer.Option(0.5, help="Battery power rating (MW)."),
    rte: float = typer.Option(0.87, help="Round-trip efficiency, 0-1."),
) -> None:
    """Run every strategy on one hub-year and report revenue and capture rate."""
    prices = load_prices(year, hub)
    spec = BatterySpec(capacity_mwh=capacity_mwh, power_mw=power_mw, round_trip_efficiency=rte)
    revenue: dict[str, float] = {}
    cycles: dict[str, float] = {}
    for name in _STRATEGIES:
        strat = _build_strategy(
            name,
            spec=spec,
            interval_minutes=RTM_INTERVAL_MINUTES,
            charge_below=charge_below,
            discharge_above=discharge_above,
            window_hours=window_hours,
            band=band,
        )
        metrics = compute_metrics(run_backtest(prices, Battery(spec), strat))
        revenue[name] = metrics.total_revenue
        cycles[name] = metrics.equivalent_full_cycles

    ceiling = revenue["perfect-foresight"]
    typer.echo(f"\nStrategy comparison -- {hub} {year}")
    typer.echo(f"  {'strategy':<18}{'revenue':>13}{'capture':>10}{'cycles':>9}")
    for name in _STRATEGIES:
        money = "$" + format(revenue[name], ",.0f")
        rate = capture_rate(revenue[name], ceiling)
        typer.echo(f"  {name:<18}{money:>13}{rate:>9.1%}{cycles[name]:>9.1f}")


@app.command("reproduce-v1")
def reproduce_v1() -> None:
    """Regenerate every v1 result from scratch.

    Fetches any missing hub-years (needs GRIDSTATUS_API_KEY), backtests all
    three strategies across both years and all four hubs, writes the results
    table to results/v1/results.csv, and saves charts for one representative
    hub-year.
    """
    spec = BatterySpec()
    out_dir = Path("results") / "v1"
    rows: list[dict[str, object]] = []

    for year in (2024, 2025):
        for hub in ERCOT_HUBS:
            if not is_cached(year, hub):
                typer.echo(f"fetching missing data: {hub} {year} ...")
                save_prices(GridstatusERCOTSource().get_rtm_prices(year, hub), year, hub)
            prices = load_prices(year, hub)
            metrics: dict[str, BacktestMetrics] = {}
            for name in _STRATEGIES:
                strat = _build_strategy(
                    name,
                    spec=spec,
                    interval_minutes=RTM_INTERVAL_MINUTES,
                    charge_below=20.0,
                    discharge_above=50.0,
                    window_hours=24.0,
                    band=0.0,
                )
                metrics[name] = compute_metrics(run_backtest(prices, Battery(spec), strat))
            ceiling = metrics["perfect-foresight"].total_revenue
            for name, result_metrics in metrics.items():
                rows.append(
                    {
                        "year": year,
                        "hub": hub,
                        "strategy": name,
                        "revenue": round(result_metrics.total_revenue, 2),
                        "capture_rate": round(
                            capture_rate(result_metrics.total_revenue, ceiling), 4
                        ),
                        "equivalent_cycles": round(result_metrics.equivalent_full_cycles, 1),
                    }
                )
            typer.echo(f"  backtested {hub} {year}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    pl.DataFrame(rows).write_csv(csv_path)
    typer.echo(f"\nwrote results table -> {csv_path}")

    # Charts for one representative hub-year.
    sample = run_backtest(
        load_prices(2024, "HB_HOUSTON"), Battery(spec), PerfectForesightStrategy(spec)
    )
    charts_dir = out_dir / "charts"
    for label, figure in (
        ("dispatch", dispatch_chart(sample)),
        ("soc", soc_chart(sample)),
        ("cumulative_revenue", cumulative_revenue_chart(sample)),
        ("daily_revenue", daily_revenue_chart(sample)),
    ):
        save_figure(figure, charts_dir / f"{label}_HB_HOUSTON_2024.html")
    typer.echo(f"wrote charts -> {charts_dir}")


if __name__ == "__main__":
    app()
