"""Command-line interface for dispatcher-watts."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from dispatcher_watts.backtest.engine import BacktestResult, run_backtest
from dispatcher_watts.backtest.metrics import BacktestMetrics, compute_metrics
from dispatcher_watts.battery.model import Battery, BatterySpec
from dispatcher_watts.data.ercot import GridstatusERCOTSource
from dispatcher_watts.data.schemas import ERCOT_HUBS
from dispatcher_watts.data.store import (
    cache_path,
    load_prices,
    save_prices,
    summarize_prices,
)
from dispatcher_watts.strategies.base import Strategy
from dispatcher_watts.strategies.threshold import ThresholdStrategy

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


def _build_strategy(name: str, charge_below: float, discharge_above: float) -> Strategy:
    if name == "threshold":
        try:
            return ThresholdStrategy(charge_below=charge_below, discharge_above=discharge_above)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    raise typer.BadParameter(f"unknown strategy {name!r}; v1 supports: threshold")


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
    strategy: str = typer.Option("threshold", help="Dispatch strategy. v1: threshold."),
    charge_below: float = typer.Option(20.0, help="threshold: charge when price <= this ($/MWh)."),
    discharge_above: float = typer.Option(
        50.0, help="threshold: discharge when price >= this ($/MWh)."
    ),
    capacity_mwh: float = typer.Option(1.0, help="Battery energy capacity (MWh)."),
    power_mw: float = typer.Option(0.5, help="Battery power rating (MW)."),
    rte: float = typer.Option(0.87, help="Round-trip efficiency, 0-1."),
) -> None:
    """Backtest a dispatch strategy on cached ERCOT prices for one hub-year."""
    prices = load_prices(year, hub)
    battery = Battery(
        BatterySpec(capacity_mwh=capacity_mwh, power_mw=power_mw, round_trip_efficiency=rte)
    )
    strat = _build_strategy(strategy, charge_below, discharge_above)
    result = run_backtest(prices, battery, strat)
    _print_backtest(hub, year, result, compute_metrics(result))


if __name__ == "__main__":
    app()
