"""Look-back analysis of a live run against its perfect-foresight ceiling.

The live strategy decides with only present information; the co-optimization LP
(``cooptimization/solver.py``) decides with full foresight over the same data.
Their ratio is the capture rate -- the headline honesty metric for the live
experiment: "of the revenue that was theoretically on the table, how much did a
deployable rule actually get?"

Both sides are computed over the *captured* market data the live run observed
(``data/live/``), so the comparison is apples-to-apples -- including the shared
limitation that AS uses indicative, not settled, MCPC (see ``ercot_direct``).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dispatcher_watts.backtest.metrics import capture_rate
from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.cooptimization.solver import solve_co_optimization
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, RTM_INTERVAL_MINUTES


@dataclass
class CaptureRateResult:
    """Live actual vs. foresight ceiling over the intervals the LP covers."""

    intervals: int
    actual_revenue: float
    ceiling_revenue: float
    capture_rate: float
    actual_by_source: dict[str, float]
    ceiling_by_source: dict[str, float]


def live_capture_rate(
    decisions: pl.DataFrame,
    prices: pl.DataFrame,
    mcpc: pl.DataFrame,
    spec: BatterySpec,
    interval_minutes: int = RTM_INTERVAL_MINUTES,
    degradation_cost_per_mwh: float = 0.0,
) -> CaptureRateResult:
    """Compare the live decision log to the foresight LP over captured data.

    The LP defines the comparable interval set (the overlap of `prices` and
    `mcpc`); the live actuals are summed over exactly those intervals so neither
    side is credited for intervals the other could not see.
    """
    ceiling = solve_co_optimization(
        prices, mcpc, spec, interval_minutes, degradation_cost_per_mwh=degradation_cost_per_mwh
    )
    covered = ceiling.frame["interval_start"]
    actual_rows = decisions.filter(pl.col("interval_start").is_in(covered))

    actual_by_source = _actual_by_source(actual_rows)
    actual_revenue = sum(actual_by_source.values())
    return CaptureRateResult(
        intervals=actual_rows.height,
        actual_revenue=actual_revenue,
        ceiling_revenue=ceiling.total_revenue,
        capture_rate=capture_rate(actual_revenue, ceiling.total_revenue),
        actual_by_source=actual_by_source,
        ceiling_by_source=ceiling.revenue_by_source,
    )


def _actual_by_source(decisions: pl.DataFrame) -> dict[str, float]:
    """Split realized live revenue into energy + per-AS-product buckets."""
    by_source = {"energy": float(decisions["energy_revenue"].sum())}
    for product in MCPC_PRODUCTS:
        product_revenue = decisions.filter(pl.col("as_product") == product)["as_revenue"].sum()
        by_source[product] = float(product_revenue)
    return by_source


__all__ = ["CaptureRateResult", "live_capture_rate"]
