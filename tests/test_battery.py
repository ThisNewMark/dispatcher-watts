"""Tests for the battery model: state transitions, constraints, edge cases."""

from __future__ import annotations

import pytest

from dispatcher_watts.battery.model import Battery, BatterySpec


def _lossless(capacity: float = 1.0, power: float = 0.5) -> Battery:
    """A battery with RTE 1.0, so grid energy equals state-of-charge change."""
    return Battery(BatterySpec(capacity_mwh=capacity, power_mw=power, round_trip_efficiency=1.0))


def test_spec_one_way_efficiency_is_sqrt_rte() -> None:
    assert BatterySpec(round_trip_efficiency=0.81).one_way_efficiency == pytest.approx(0.9)


def test_spec_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="capacity_mwh"):
        BatterySpec(capacity_mwh=0)
    with pytest.raises(ValueError, match="power_mw"):
        BatterySpec(power_mw=-1)
    with pytest.raises(ValueError, match="round_trip_efficiency"):
        BatterySpec(round_trip_efficiency=1.5)


def test_battery_rejects_initial_soc_out_of_range() -> None:
    with pytest.raises(ValueError, match="initial_soc_mwh"):
        Battery(BatterySpec(capacity_mwh=1.0), initial_soc_mwh=2.0)


def test_charge_limited_by_power_rating() -> None:
    battery = _lossless(capacity=10.0, power=0.5)
    # One hour at 0.5 MW -> at most 0.5 MWh, even when far more is requested.
    assert battery.charge(99.0, hours=1.0) == pytest.approx(0.5)
    assert battery.soc_mwh == pytest.approx(0.5)


def test_charge_limited_by_capacity_headroom() -> None:
    battery = _lossless(capacity=1.0, power=100.0)  # power huge; headroom binds
    assert battery.charge(99.0, hours=1.0) == pytest.approx(1.0)
    assert battery.soc_mwh == pytest.approx(1.0)


def test_charge_to_exact_capacity_then_no_more() -> None:
    battery = _lossless(capacity=1.0, power=100.0)
    battery.charge(1.0, hours=1.0)
    assert battery.soc_mwh == pytest.approx(1.0)
    assert battery.max_charge_mwh(hours=1.0) == pytest.approx(0.0)
    assert battery.charge(1.0, hours=1.0) == pytest.approx(0.0)


def test_discharge_limited_by_state_of_charge() -> None:
    battery = _lossless(capacity=10.0, power=100.0)
    battery.charge(2.0, hours=1.0)
    assert battery.discharge(99.0, hours=1.0) == pytest.approx(2.0)
    assert battery.soc_mwh == pytest.approx(0.0)


def test_discharge_from_empty_delivers_nothing() -> None:
    battery = _lossless()
    assert battery.discharge(1.0, hours=1.0) == pytest.approx(0.0)
    assert battery.soc_mwh == pytest.approx(0.0)


def test_round_trip_efficiency_loss() -> None:
    # RTE 0.81: charging 1 MWh stores 0.9; discharging it all returns 0.81.
    battery = Battery(BatterySpec(capacity_mwh=10.0, power_mw=100.0, round_trip_efficiency=0.81))
    battery.charge(1.0, hours=1.0)
    assert battery.soc_mwh == pytest.approx(0.9)
    assert battery.discharge(99.0, hours=1.0) == pytest.approx(0.81)
    assert battery.soc_mwh == pytest.approx(0.0)


def test_negative_request_raises() -> None:
    battery = _lossless()
    with pytest.raises(ValueError, match="non-negative"):
        battery.charge(-1.0, hours=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        battery.discharge(-1.0, hours=1.0)


def test_equivalent_full_cycles_and_throughput() -> None:
    battery = _lossless(capacity=1.0, power=100.0)
    for _ in range(2):  # two full charge/discharge round trips
        battery.charge(1.0, hours=1.0)
        battery.discharge(1.0, hours=1.0)
    assert battery.equivalent_full_cycles == pytest.approx(2.0)
    assert battery.energy_charged_mwh == pytest.approx(2.0)
    assert battery.energy_discharged_mwh == pytest.approx(2.0)
