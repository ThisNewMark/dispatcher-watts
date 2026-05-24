"""Tests for live simulator state persistence (live/state.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.live.state import (
    DECISION_LOG_SCHEMA,
    DecisionRecord,
    LiveState,
    append_decisions,
    load_decisions,
    load_state,
    save_state,
    state_exists,
)
from dispatcher_watts.strategies.live import ASCommitment

T0 = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.UTC)


def _state() -> LiveState:
    return LiveState(
        hub="HB_HOUSTON",
        spec=BatterySpec(capacity_mwh=10.0, power_mw=2.5, round_trip_efficiency=0.87),
        strategy_name="follow-the-leader",
        strategy_config={"charge_below": 20.0, "discharge_above": 50.0},
        interval_minutes=15,
    )


def _record(at: dt.datetime, *, product: str | None = "rrs") -> DecisionRecord:
    return DecisionRecord(
        interval_start=at,
        price=80.0,
        mcpc={"regup": 4.0, "rrs": 9.0},
        energy_mw=1.25,
        energy_mwh=0.3125,
        as_product=product,
        as_mw=1.25,
        energy_revenue=25.0,
        as_revenue=11.25,
        soc_mwh_after=5.0,
        reason="discharge; AS rrs",
    )


# --- LiveState round-trip ----------------------------------------------------


def test_save_load_round_trips_scalar_state(tmp_path: Path) -> None:
    state = _state()
    state.soc_mwh = 6.0
    state.revenue_by_source["energy"] = 123.45
    state.revenue_by_source["rrs"] = 67.0
    state.last_commitment = ASCommitment(product="rrs", mw=1.25, committed_at=T0)
    state.last_processed_interval = T0
    save_state(state, data_dir=tmp_path)

    loaded = load_state(data_dir=tmp_path)
    assert loaded.hub == "HB_HOUSTON"
    assert loaded.spec == state.spec
    assert loaded.soc_mwh == 6.0
    assert loaded.revenue_by_source["energy"] == 123.45
    assert loaded.revenue_by_source["rrs"] == 67.0
    assert loaded.last_commitment == state.last_commitment
    assert loaded.last_processed_interval == T0


def test_round_trips_null_commitment_and_interval(tmp_path: Path) -> None:
    save_state(_state(), data_dir=tmp_path)
    loaded = load_state(data_dir=tmp_path)
    assert loaded.last_commitment is None
    assert loaded.last_processed_interval is None


def test_battery_counters_survive_round_trip(tmp_path: Path) -> None:
    state = _state()
    battery = state.to_battery()
    battery.charge(2.0, hours=0.25)
    battery.discharge(1.0, hours=0.25)
    state.adopt_battery(battery)
    save_state(state, data_dir=tmp_path)

    loaded = load_state(data_dir=tmp_path)
    assert loaded.soc_mwh == pytest.approx(state.soc_mwh)
    assert loaded.throughput_internal_mwh == pytest.approx(state.throughput_internal_mwh)
    assert loaded.equivalent_full_cycles == pytest.approx(state.equivalent_full_cycles)


def test_to_battery_restores_soc_and_throughput(tmp_path: Path) -> None:
    state = _state()
    state.soc_mwh = 4.0
    state.throughput_internal_mwh = 30.0
    battery = state.to_battery()
    assert battery.soc_mwh == 4.0
    assert battery.equivalent_full_cycles == pytest.approx(3.0)  # 30 / 10


def test_total_revenue_sums_sources(tmp_path: Path) -> None:
    state = _state()
    state.revenue_by_source["energy"] = 100.0
    state.revenue_by_source["rrs"] = 50.0
    assert state.total_revenue == 150.0


def test_load_state_missing_raises(tmp_path: Path) -> None:
    assert not state_exists(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="no live state"):
        load_state(data_dir=tmp_path)


# --- decision log ------------------------------------------------------------


def test_append_and_load_decisions(tmp_path: Path) -> None:
    append_decisions([_record(T0)], data_dir=tmp_path)
    append_decisions([_record(T0 + dt.timedelta(minutes=15))], data_dir=tmp_path)
    log = load_decisions(data_dir=tmp_path)
    assert log.schema == DECISION_LOG_SCHEMA
    assert log.height == 2
    assert log["mcpc_rrs"].to_list() == [9.0, 9.0]
    assert log["as_product"].to_list() == ["rrs", "rrs"]


def test_decision_log_dedups_on_interval(tmp_path: Path) -> None:
    append_decisions([_record(T0, product="rrs")], data_dir=tmp_path)
    append_decisions([_record(T0, product="ecrs")], data_dir=tmp_path)  # reprocess same interval
    log = load_decisions(data_dir=tmp_path)
    assert log.height == 1
    assert log["as_product"].to_list() == ["ecrs"]  # latest wins


def test_decision_log_handles_null_as_product(tmp_path: Path) -> None:
    append_decisions([_record(T0, product=None)], data_dir=tmp_path)
    log = load_decisions(data_dir=tmp_path)
    assert log["as_product"].to_list() == [None]


def test_load_decisions_empty_when_absent(tmp_path: Path) -> None:
    log = load_decisions(data_dir=tmp_path)
    assert log.is_empty()
    assert log.schema == DECISION_LOG_SCHEMA


def test_append_empty_records_is_noop(tmp_path: Path) -> None:
    append_decisions([], data_dir=tmp_path)
    assert load_decisions(data_dir=tmp_path).is_empty()
