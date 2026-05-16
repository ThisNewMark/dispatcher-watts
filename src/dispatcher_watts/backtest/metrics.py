"""Backtest metrics: revenue, equivalent cycles, capacity factor, SoC distribution."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dispatcher_watts.backtest.engine import BacktestResult

# Hours in a (non-leap) year, used to annualize revenue from a partial period.
HOURS_PER_YEAR: float = 8760.0


@dataclass
class BacktestMetrics:
    """Headline numbers from a finished backtest."""

    total_revenue: float
    revenue_per_mwh_year: float
    equivalent_full_cycles: float
    capacity_factor: float
    intervals_charging: int
    intervals_discharging: int
    intervals_idle: int
    # Fraction of intervals spent in each 10%-wide state-of-charge band.
    soc_distribution: dict[str, float]


def compute_metrics(result: BacktestResult) -> BacktestMetrics:
    """Summarize a finished backtest."""
    frame = result.frame
    spec = result.spec
    battery = result.final_battery
    hours = result.interval_minutes / 60.0
    total_hours = frame.height * hours

    total_revenue = float(frame["revenue"].sum())
    annualization = HOURS_PER_YEAR / total_hours if total_hours > 0 else 0.0
    revenue_per_mwh_year = total_revenue / spec.capacity_mwh * annualization

    capacity_factor = (
        battery.energy_discharged_mwh / (spec.power_mw * total_hours) if total_hours > 0 else 0.0
    )

    actions = frame["action"]
    return BacktestMetrics(
        total_revenue=total_revenue,
        revenue_per_mwh_year=revenue_per_mwh_year,
        equivalent_full_cycles=battery.equivalent_full_cycles,
        capacity_factor=capacity_factor,
        intervals_charging=int((actions < 0).sum()),
        intervals_discharging=int((actions > 0).sum()),
        intervals_idle=int((actions == 0).sum()),
        soc_distribution=_soc_distribution(frame, spec.capacity_mwh),
    )


def capture_rate(strategy_revenue: float, perfect_foresight_revenue: float) -> float:
    """Strategy revenue as a fraction of the perfect-foresight ceiling.

    Returns 0.0 when the ceiling is non-positive -- i.e. no profitable
    arbitrage was available, so there is nothing to capture.
    """
    if perfect_foresight_revenue <= 0:
        return 0.0
    return strategy_revenue / perfect_foresight_revenue


def _soc_distribution(frame: pl.DataFrame, capacity_mwh: float) -> dict[str, float]:
    """Fraction of intervals spent in each 10%-wide state-of-charge band."""
    n = frame.height
    if n == 0:
        return {}
    soc_fraction = frame["soc_mwh"] / capacity_mwh
    distribution: dict[str, float] = {}
    for band in range(10):
        low, high = band / 10, (band + 1) / 10
        # The top band is closed on the right so a full battery is counted.
        in_band = (soc_fraction >= low) & (
            soc_fraction <= high if band == 9 else soc_fraction < high
        )
        distribution[f"{band * 10}-{band * 10 + 10}%"] = int(in_band.sum()) / n
    return distribution
