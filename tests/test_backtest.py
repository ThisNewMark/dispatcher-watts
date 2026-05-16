"""Tests for the backtest engine and metrics."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from dispatcher_watts.backtest.engine import run_backtest
from dispatcher_watts.backtest.metrics import capture_rate, compute_metrics
from dispatcher_watts.battery.model import Battery, BatterySpec
from dispatcher_watts.strategies.perfect_foresight import PerfectForesightStrategy
from dispatcher_watts.strategies.threshold import ThresholdStrategy


def _prices(values: list[float]) -> pl.DataFrame:
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    return pl.DataFrame(
        {
            "interval_start": [start + dt.timedelta(hours=i) for i in range(len(values))],
            "price": values,
        }
    ).cast({"interval_start": pl.Datetime(time_unit="us", time_zone="UTC")})


def _lossless_battery() -> Battery:
    return Battery(BatterySpec(capacity_mwh=1.0, power_mw=0.5, round_trip_efficiency=1.0))


def test_run_backtest_hand_computed_revenue() -> None:
    # Lossless 1 MWh / 0.5 MW battery, hourly intervals. Prices [10, 10, 100, 100]:
    #   charge    0.5 MWh x $10  twice  -> -$10
    #   discharge 0.5 MWh x $100 twice  -> +$100  => total revenue $90
    result = run_backtest(
        _prices([10.0, 10.0, 100.0, 100.0]),
        _lossless_battery(),
        ThresholdStrategy(charge_below=20.0, discharge_above=50.0),
        interval_minutes=60,
    )
    assert result.frame["revenue"].sum() == pytest.approx(90.0)
    assert result.frame["soc_mwh"].to_list() == pytest.approx([0.5, 1.0, 0.5, 0.0])
    assert result.frame["cumulative_revenue"].to_list() == pytest.approx([-5.0, -10.0, 40.0, 90.0])


def test_metrics_counts_and_cycles() -> None:
    result = run_backtest(
        _prices([10.0, 10.0, 100.0, 100.0]),
        _lossless_battery(),
        ThresholdStrategy(charge_below=20.0, discharge_above=50.0),
        interval_minutes=60,
    )
    metrics = compute_metrics(result)
    assert metrics.total_revenue == pytest.approx(90.0)
    assert metrics.intervals_charging == 2
    assert metrics.intervals_discharging == 2
    assert metrics.intervals_idle == 0
    assert metrics.equivalent_full_cycles == pytest.approx(1.0)
    assert sum(metrics.soc_distribution.values()) == pytest.approx(1.0)


def test_run_backtest_rejects_bad_schema() -> None:
    bad = pl.DataFrame({"interval_start": [1, 2], "price": [3.0, 4.0]})
    with pytest.raises(ValueError, match="RTM_PRICE_SCHEMA"):
        run_backtest(bad, _lossless_battery(), ThresholdStrategy(20.0, 50.0))


def test_capture_rate() -> None:
    assert capture_rate(50.0, 100.0) == pytest.approx(0.5)
    assert capture_rate(50.0, 0.0) == 0.0  # no ceiling -> nothing to capture
    assert capture_rate(10.0, -5.0) == 0.0


def test_perfect_foresight_is_a_revenue_ceiling() -> None:
    # Perfect foresight must earn at least as much as any other strategy on the
    # same prices, so the capture rate is bounded by 1.
    prices = _prices([10.0, 80.0, 15.0, 5.0, 120.0, 30.0])
    spec = BatterySpec(capacity_mwh=1.0, power_mw=0.5, round_trip_efficiency=0.9)

    threshold_revenue = compute_metrics(
        run_backtest(prices, Battery(spec), ThresholdStrategy(20.0, 50.0), interval_minutes=60)
    ).total_revenue
    perfect_revenue = compute_metrics(
        run_backtest(
            prices,
            Battery(spec),
            PerfectForesightStrategy(spec, interval_minutes=60),
            interval_minutes=60,
        )
    ).total_revenue

    assert perfect_revenue >= threshold_revenue - 1e-6
    assert capture_rate(threshold_revenue, perfect_revenue) <= 1.0 + 1e-6
