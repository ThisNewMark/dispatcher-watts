"""Command-line interface for dispatcher-watts."""

from __future__ import annotations

import datetime as dt
import shutil
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
from dispatcher_watts.cooptimization.solver import solve_co_optimization
from dispatcher_watts.data.ercot import GridstatusERCOTSource
from dispatcher_watts.data.schemas import ERCOT_HUBS, MCPC_PRODUCTS, RTM_INTERVAL_MINUTES
from dispatcher_watts.data.store import (
    cache_path,
    is_cached,
    load_mcpc,
    load_mcpc_window,
    load_prices,
    load_prices_window,
    mcpc_cache_path,
    save_mcpc,
    save_prices,
    summarize_prices,
)
from dispatcher_watts.finance.economics import BatteryEconomics, summarize_finance
from dispatcher_watts.reporting.charts import (
    cumulative_revenue_chart,
    cumulative_revenue_comparison_chart,
    daily_revenue_chart,
    dispatch_chart,
    revenue_stack_chart,
    rtcb_pct_change_chart,
    rtcb_revenue_comparison_chart,
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


@data_app.command("fetch-as")
def data_fetch_as(
    year: int = typer.Option(..., help="Calendar year to fetch, e.g. 2026."),
    force: bool = typer.Option(False, help="Re-fetch even if a cached file exists."),
) -> None:
    """Fetch one year of ERCOT real-time AS clearing prices (post-RTC+B).

    Only Dec 5, 2025 onward is meaningful; earlier dates return empty or
    sparse data (RTC+B introduced real-time AS clearing).
    """
    path = mcpc_cache_path(year)
    if path.exists() and not force:
        typer.echo(f"cached file already exists: {path} (use --force to re-fetch)")
        df = load_mcpc(year)
    else:
        typer.echo(f"fetching MCPC {year} from gridstatus.io ...")
        df = GridstatusERCOTSource().get_rtm_mcpc(year)
        path = save_mcpc(df, year)
        typer.echo(f"wrote {df.height:,} intervals to {path}")
    if df.is_empty():
        typer.echo("  (frame empty; this is expected for pre-RTC+B years)")
        return
    typer.echo(f"\nMCPC {year} -- mean clearing price per 15-min interval ($/MW)")
    products = ("regup", "regdn", "rrs", "ecrs", "nspin")
    stats = df.select(
        *[pl.col(f"mcpc_{p}").mean().alias(f"mean_{p}") for p in products],
        *[pl.col(f"mcpc_{p}").max().alias(f"max_{p}") for p in products],
    ).row(0, named=True)
    for product in products:
        typer.echo(
            f"  {product:<6}  mean ${stats[f'mean_{product}']:6.2f}   "
            f"max ${stats[f'max_{product}']:8.2f}"
        )


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
    table to results/v1/results.csv, and saves charts for every strategy on
    one representative hub-year (HB_HOUSTON 2024).
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

    # Charts for one representative hub-year (HB_HOUSTON 2024): the four charts
    # for every strategy, in a per-strategy subfolder, plus one comparison
    # chart overlaying all strategies' cumulative revenue.
    charts_dir = out_dir / "charts"
    if charts_dir.exists():
        shutil.rmtree(charts_dir)  # clear stale charts from a previous run
    sample_prices = load_prices(2024, "HB_HOUSTON")
    sample_results: dict[str, BacktestResult] = {}
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
        result = run_backtest(sample_prices, Battery(spec), strat)
        sample_results[name] = result
        for label, figure in (
            ("dispatch", dispatch_chart(result)),
            ("soc", soc_chart(result)),
            ("cumulative_revenue", cumulative_revenue_chart(result)),
            ("daily_revenue", daily_revenue_chart(result)),
        ):
            save_figure(figure, charts_dir / name / f"{label}_HB_HOUSTON_2024.html")
    save_figure(
        cumulative_revenue_comparison_chart(sample_results),
        charts_dir / "comparison_cumulative_revenue_HB_HOUSTON_2024.html",
    )
    typer.echo(f"wrote charts -> {charts_dir}")


# ERCOT's RTC+B (Real-Time Co-optimization plus Batteries) market redesign went
# live on this date -- the boundary between the "old" and "new" regimes.
RTCB_GO_LIVE: dt.date = dt.date(2025, 12, 5)


def _backtest_window(
    start: dt.date,
    end: dt.date,
    hub: str,
    spec: BatterySpec,
    *,
    charge_below: float,
    discharge_above: float,
    window_hours: float,
    band: float,
) -> dict[str, BacktestMetrics]:
    """Run every strategy on `[start, end)` for one hub; return metrics by name."""
    prices = load_prices_window(start, end, hub)
    if prices.is_empty():
        raise typer.BadParameter(
            f"no cached prices for {hub} in [{start}, {end}); "
            f"run `dispatcher-watts data fetch` for the missing year(s)"
        )
    metrics: dict[str, BacktestMetrics] = {}
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
        metrics[name] = compute_metrics(run_backtest(prices, Battery(spec), strat))
    return metrics


@app.command("rtcb-compare")
def rtcb_compare(
    end_date: str | None = typer.Option(
        None, help="End of the post-RTC+B window (UTC, exclusive). Default: today."
    ),
    capacity_mwh: float = typer.Option(1.0, help="Battery energy capacity (MWh)."),
    power_mw: float = typer.Option(0.5, help="Battery power rating (MW)."),
    rte: float = typer.Option(0.87, help="Round-trip efficiency, 0-1."),
    charge_below: float = typer.Option(20.0, help="threshold charge price ($/MWh)."),
    discharge_above: float = typer.Option(50.0, help="threshold discharge price ($/MWh)."),
    window_hours: float = typer.Option(24.0, help="rolling-average window (hours)."),
    band: float = typer.Option(0.0, help="rolling-average no-trade band (0-1)."),
) -> None:
    """Compare energy-arbitrage revenue in matched windows pre and post RTC+B.

    The post-RTC+B window runs from RTC+B go-live (Dec 5, 2025) to `--end-date`.
    The pre-RTC+B window is the same calendar window one year earlier, so the
    two are matched on season and length. All three strategies are backtested
    on both windows for every ERCOT trading hub.

    Caveat: any pre→post delta blends the RTC+B effect with continuing battery
    saturation. The v1 baseline showed ~20% year-over-year compression from
    saturation alone, before any regime change. Interpret deltas accordingly.
    """
    end = dt.date.fromisoformat(end_date) if end_date else dt.date.today()
    if end <= RTCB_GO_LIVE:
        raise typer.BadParameter(f"--end-date {end} must be after RTC+B go-live ({RTCB_GO_LIVE})")
    post_start, post_end = RTCB_GO_LIVE, end
    pre_start = RTCB_GO_LIVE - dt.timedelta(days=365)
    pre_end = end - dt.timedelta(days=365)
    spec = BatterySpec(capacity_mwh=capacity_mwh, power_mw=power_mw, round_trip_efficiency=rte)

    typer.echo(f"\nRTC+B comparison ({capacity_mwh:g} MWh / {power_mw:g} MW battery)")
    typer.echo(f"  pre-RTC+B  : [{pre_start}, {pre_end})  -- {(pre_end - pre_start).days} days")
    typer.echo(f"  post-RTC+B : [{post_start}, {post_end}) -- {(post_end - post_start).days} days")

    rows: list[dict[str, object]] = []
    pf_pre: dict[str, float] = {}
    pf_post: dict[str, float] = {}
    for hub in ERCOT_HUBS:
        pre_metrics = _backtest_window(
            pre_start,
            pre_end,
            hub,
            spec,
            charge_below=charge_below,
            discharge_above=discharge_above,
            window_hours=window_hours,
            band=band,
        )
        post_metrics = _backtest_window(
            post_start,
            post_end,
            hub,
            spec,
            charge_below=charge_below,
            discharge_above=discharge_above,
            window_hours=window_hours,
            band=band,
        )
        pre_days = (pre_end - pre_start).days
        post_days = (post_end - post_start).days
        for window_label, window_metrics, days in (
            ("pre", pre_metrics, pre_days),
            ("post", post_metrics, post_days),
        ):
            ceiling = window_metrics["perfect-foresight"].total_revenue
            for name in _STRATEGIES:
                m = window_metrics[name]
                rows.append(
                    {
                        "window": window_label,
                        "hub": hub,
                        "strategy": name,
                        "days": days,
                        "revenue": round(m.total_revenue, 2),
                        "revenue_per_day": round(m.total_revenue / days, 2),
                        "capture_rate": round(capture_rate(m.total_revenue, ceiling), 4),
                        "equivalent_cycles": round(m.equivalent_full_cycles, 1),
                    }
                )
        pf_pre[hub] = pre_metrics["perfect-foresight"].total_revenue / pre_days
        pf_post[hub] = post_metrics["perfect-foresight"].total_revenue / post_days
        typer.echo(
            f"  {hub:<12} PF $/day  pre {pf_pre[hub]:>7,.0f}  "
            f"post {pf_post[hub]:>7,.0f}  "
            f"({(pf_post[hub] - pf_pre[hub]) / pf_pre[hub] * 100:+.1f}%)"
        )

    out_dir = Path("results") / "rtcb-v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    pl.DataFrame(rows).write_csv(csv_path)
    typer.echo(f"\nwrote results table -> {csv_path}")

    charts_dir = out_dir / "charts"
    if charts_dir.exists():
        shutil.rmtree(charts_dir)
    save_figure(
        rtcb_revenue_comparison_chart(pf_pre, pf_post),
        charts_dir / "pre_vs_post_pf_revenue.html",
    )
    save_figure(
        rtcb_pct_change_chart(pf_pre, pf_post),
        charts_dir / "pct_change_pf_revenue.html",
    )
    typer.echo(f"wrote charts -> {charts_dir}")


@app.command("revenue-stack")
def revenue_stack(
    start: str = typer.Option(
        "2025-12-05", help="Window start (UTC, inclusive). Default: RTC+B go-live."
    ),
    end: str | None = typer.Option(None, help="Window end (UTC, exclusive). Default: today."),
    hub: str = typer.Option("HB_HOUSTON", help="ERCOT trading hub."),
    capacity_mwh: float = typer.Option(10.0, help="Battery energy capacity (MWh)."),
    power_mw: float = typer.Option(2.5, help="Battery power rating (MW)."),
    rte: float = typer.Option(0.87, help="Round-trip efficiency, 0-1."),
) -> None:
    """Co-optimize energy and AS revenue for a post-RTC+B window.

    Solves one LP that allocates the battery's power and state of charge
    across real-time energy and all five AS products jointly, with full
    foresight. The result is the post-RTC+B revenue ceiling -- the
    co-optimized analogue of the v1 perfect-foresight benchmark.
    """
    start_date = dt.date.fromisoformat(start)
    end_date = dt.date.fromisoformat(end) if end else dt.date.today()
    prices = load_prices_window(start_date, end_date, hub)
    mcpc = load_mcpc_window(start_date, end_date)
    spec = BatterySpec(capacity_mwh=capacity_mwh, power_mw=power_mw, round_trip_efficiency=rte)
    days = (end_date - start_date).days
    econ = BatteryEconomics()

    typer.echo(f"\nRevenue stack -- {capacity_mwh:g} MWh / {power_mw:g} MW @ {hub}")
    typer.echo(f"  window     : [{start_date}, {end_date})  -- {days} days")
    typer.echo(f"  energy obs : {prices.height:,}  AS obs: {mcpc.height:,}")
    typer.echo(
        f"  solving 2 LPs: gross-max and degradation-aware "
        f"(${econ.degradation_cost_per_mwh:.0f}/MWh) ..."
    )

    out_dir = Path("results") / "rtcb-v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []
    for mode_label, deg_in_lp in (
        ("gross-max", 0.0),
        ("degradation-aware", econ.degradation_cost_per_mwh),
    ):
        typer.echo(f"\n  === {mode_label} LP ===")
        _show_one_mode(
            prices,
            mcpc,
            spec,
            days,
            econ,
            mode_label=mode_label,
            degradation_in_lp=deg_in_lp,
            out_dir=out_dir,
            hub=hub,
            start_date=start_date,
            end_date=end_date,
            summary_out=summary_lines,
        )

    typer.echo("\n  === Summary: same finance assumptions, different LP objectives ===")
    typer.echo(f"  {'mode':<22}{'gross':>14}{'net FCF/yr':>16}{'payback':>12}{'CoC':>8}")
    typer.echo(f"  {'-' * 22}{'-' * 14:>14}{'-' * 16:>16}{'-' * 12:>12}{'-' * 8:>8}")
    for line in summary_lines:
        typer.echo("  " + line)


def _show_one_mode(
    prices: pl.DataFrame,
    mcpc: pl.DataFrame,
    spec: BatterySpec,
    days: int,
    econ: BatteryEconomics,
    *,
    mode_label: str,
    degradation_in_lp: float,
    out_dir: Path,
    hub: str,
    start_date: dt.date,
    end_date: dt.date,
    summary_out: list[str],
) -> None:
    """Solve one LP variant, print its stack + finance summary, save its chart."""
    result = solve_co_optimization(prices, mcpc, spec, degradation_cost_per_mwh=degradation_in_lp)
    annualization = 365.0 / days
    typer.echo(f"  {'source':<10}{'revenue':>14}{'$/day':>10}{'$/MWh-year':>14}{'share':>8}")
    typer.echo(f"  {'-' * 10}{'-' * 14:>14}{'-' * 10:>10}{'-' * 14:>14}{'-' * 8:>8}")
    for source in ("energy", *MCPC_PRODUCTS):
        amount = result.revenue_by_source[source]
        per_mwh_year = amount / spec.capacity_mwh * annualization
        share = amount / result.total_revenue if result.total_revenue else 0.0
        typer.echo(
            f"  {source:<10}{'$' + format(amount, ',.0f'):>14}"
            f"{'$' + format(amount / days, ',.0f'):>10}"
            f"{'$' + format(per_mwh_year, ',.0f'):>14}{share:>8.1%}"
        )
    total = result.total_revenue
    typer.echo(f"  {'-' * 10}{'-' * 14:>14}{'-' * 10:>10}{'-' * 14:>14}{'-' * 8:>8}")
    typer.echo(
        f"  {'TOTAL':<10}{'$' + format(total, ',.0f'):>14}"
        f"{'$' + format(total / days, ',.0f'):>10}"
        f"{'$' + format(total / spec.capacity_mwh * annualization, ',.0f'):>14}"
    )

    throughput_mwh = float(
        result.frame.select(
            (pl.col("charge_mwh").sum() + pl.col("discharge_mwh").sum()).alias("t")
        ).item()
    )
    fin = summarize_finance(
        gross_revenue=total,
        throughput_mwh=throughput_mwh,
        capacity_mwh=spec.capacity_mwh,
        power_mw=spec.power_mw,
        days_in_window=days,
        econ=econ,
    )
    typer.echo("  Project finance:")
    typer.echo(f"    gross                          : ${fin.gross_revenue:>14,.0f}")
    typer.echo(f"    - availability (5%)            : ${fin.availability_haircut:>14,.0f}")
    typer.echo(f"    - degradation                  : ${fin.degradation_cost:>14,.0f}")
    typer.echo(f"    - QSE fee (2%)                 : ${fin.qse_fee:>14,.0f}")
    typer.echo(f"    = net operating                : ${fin.net_operating_revenue:>14,.0f}")
    typer.echo(
        f"    annualized                     : ${fin.net_operating_revenue_annualized:>14,.0f}/yr"
    )
    typer.echo(f"    - fixed O&M                    : ${fin.fixed_annual_costs:>14,.0f}/yr")
    typer.echo(
        f"    = net free cash flow           : ${fin.net_free_cash_flow_annualized:>14,.0f}/yr"
    )
    typer.echo(f"    capex after 30% ITC            : ${fin.capex_after_itc:>14,.0f}")
    if fin.net_free_cash_flow_annualized > 0:
        typer.echo(f"    simple payback                 : {fin.simple_payback_years:>14.1f} years")
        typer.echo(f"    cash-on-cash return            : {fin.cash_on_cash_return_pct:>14.1f}%")
        payback_str = f"{fin.simple_payback_years:.1f} yrs"
        coc_str = f"{fin.cash_on_cash_return_pct:.1f}%"
    else:
        typer.echo("    payback / return               : (free cash flow is negative)")
        payback_str = "n/a"
        coc_str = "n/a"

    summary_out.append(
        f"{mode_label:<22}{'$' + format(total, ',.0f'):>14}"
        f"{'$' + format(fin.net_free_cash_flow_annualized, ',.0f'):>16}"
        f"{payback_str:>12}{coc_str:>8}"
    )

    chart_path = out_dir / f"revenue_stack_{mode_label.replace('-', '_')}.html"
    save_figure(
        revenue_stack_chart(
            result.revenue_by_source,
            title=(
                f"Revenue stack ({mode_label}) -- "
                f"{spec.capacity_mwh:g} MWh / {spec.power_mw:g} MW @ {hub}, "
                f"[{start_date}, {end_date})"
            ),
        ),
        chart_path,
    )


if __name__ == "__main__":
    app()
