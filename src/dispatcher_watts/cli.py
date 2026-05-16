"""Command-line interface for dispatcher-watts."""

from __future__ import annotations

import typer

from dispatcher_watts.data.ercot import GridstatusERCOTSource
from dispatcher_watts.data.schemas import ERCOT_HUBS
from dispatcher_watts.data.store import (
    cache_path,
    load_prices,
    save_prices,
    summarize_prices,
)

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


if __name__ == "__main__":
    app()
