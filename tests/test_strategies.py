"""Tests for dispatch strategies."""

from __future__ import annotations

import pytest

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.strategies.perfect_foresight import solve_perfect_foresight_dispatch
from dispatcher_watts.strategies.rolling_avg import RollingAverageStrategy
from dispatcher_watts.strategies.threshold import ThresholdStrategy

# --- threshold ---------------------------------------------------------------


def test_threshold_decisions() -> None:
    strategy = ThresholdStrategy(charge_below=20.0, discharge_above=50.0)
    prices = [10.0, 20.0, 35.0, 50.0, 80.0]
    decisions = [strategy.decide(i, prices, soc_fraction=0.5) for i in range(len(prices))]
    # <=20 charge, >=50 discharge, in between idle. Boundaries are inclusive.
    assert decisions == [-1.0, -1.0, 0.0, 1.0, 1.0]


def test_threshold_rejects_overlapping_thresholds() -> None:
    with pytest.raises(ValueError, match="must be less than"):
        ThresholdStrategy(charge_below=50.0, discharge_above=20.0)


def test_threshold_name() -> None:
    assert ThresholdStrategy(10.0, 20.0).name == "threshold"


# --- rolling average ---------------------------------------------------------


def test_rolling_average_charges_below_and_discharges_above() -> None:
    # window_hours 1 at 60-min intervals -> a one-interval trailing window.
    strategy = RollingAverageStrategy(window_hours=1.0, interval_minutes=60)
    prices = [50.0, 10.0, 90.0]
    strategy.prepare(prices)
    assert strategy.decide(0, prices, 0.5) == 0.0  # no history yet
    assert strategy.decide(1, prices, 0.5) == -1.0  # 10 < trailing avg 50
    assert strategy.decide(2, prices, 0.5) == 1.0  # 90 > trailing avg 10


def test_rolling_average_band_creates_a_no_trade_zone() -> None:
    strategy = RollingAverageStrategy(window_hours=1.0, interval_minutes=60, band=0.5)
    prices = [100.0, 120.0]  # trailing avg at step 1 is 100; band is +/-50
    strategy.prepare(prices)
    assert strategy.decide(1, prices, 0.5) == 0.0  # 120 is inside [50, 150]


def test_rolling_average_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="window_hours"):
        RollingAverageStrategy(window_hours=0)
    with pytest.raises(ValueError, match="band"):
        RollingAverageStrategy(band=1.5)


# --- perfect foresight (linear program) --------------------------------------


def test_perfect_foresight_lossless_round_trip() -> None:
    # Lossless 1 MWh / 1 MW battery, hourly. Prices [10, 50]:
    # charge 1 MWh at $10, discharge 1 MWh at $50  ->  net dispatch [-1, +1].
    spec = BatterySpec(capacity_mwh=1.0, power_mw=1.0, round_trip_efficiency=1.0)
    net = solve_perfect_foresight_dispatch([10.0, 50.0], spec, hours=1.0)
    assert net == pytest.approx([-1.0, 1.0], abs=1e-4)


def test_perfect_foresight_idles_when_no_profit_exists() -> None:
    # Descending prices: any round trip loses money, so the LP stays idle.
    spec = BatterySpec(capacity_mwh=1.0, power_mw=1.0, round_trip_efficiency=1.0)
    net = solve_perfect_foresight_dispatch([50.0, 10.0], spec, hours=1.0)
    assert net == pytest.approx([0.0, 0.0], abs=1e-4)


def test_perfect_foresight_respects_round_trip_loss() -> None:
    # RTE 0.81: charging 1 MWh of grid energy stores 0.9 and later delivers
    # 0.9 * 0.9 = 0.81. The LP discharges exactly that.
    spec = BatterySpec(capacity_mwh=1.0, power_mw=1.0, round_trip_efficiency=0.81)
    net = solve_perfect_foresight_dispatch([10.0, 100.0], spec, hours=1.0)
    assert net[0] == pytest.approx(-1.0, abs=1e-4)
    assert net[1] == pytest.approx(0.81, abs=1e-4)
