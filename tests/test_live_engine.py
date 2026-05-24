"""Tests for the pure live-simulator step (live/engine.py)."""

from __future__ import annotations

import datetime as dt

import pytest

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.live.engine import step
from dispatcher_watts.live.state import LiveState
from dispatcher_watts.strategies.live import (
    ASCommitment,
    FollowTheLeaderStrategy,
    LiveDecision,
    LiveStrategy,
    MarketSnapshot,
)

T0 = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.UTC)


class _FixedStrategy(LiveStrategy):
    """Returns a preset decision -- lets a test pin exact energy/AS values."""

    name = "fixed"

    def __init__(self, decision: LiveDecision) -> None:
        self._decision = decision

    def decide(self, snapshot, soc_fraction, power_mw, held):  # type: ignore[no-untyped-def]
        return self._decision


def _state(soc_mwh: float = 5.0, rte: float = 1.0) -> LiveState:
    # Lossless by default so MWh<->revenue math is exact and easy to assert.
    return LiveState(
        hub="HB_HOUSTON",
        spec=BatterySpec(capacity_mwh=10.0, power_mw=2.5, round_trip_efficiency=rte),
        strategy_name="fixed",
        strategy_config={},
        interval_minutes=15,
        soc_mwh=soc_mwh,
    )


def _snapshot(price: float, mcpc: dict[str, float] | None = None) -> MarketSnapshot:
    return MarketSnapshot(timestamp=T0, price=price, mcpc=mcpc or {})


def _fixed(energy_mw: float, as_product: str | None = None, as_mw: float = 0.0) -> _FixedStrategy:
    return _FixedStrategy(
        LiveDecision(
            energy_mw=energy_mw,
            as_product=as_product,
            as_mw=as_mw,
            as_committed_at=T0,
            reason="test",
        )
    )


# --- energy leg --------------------------------------------------------------


def test_discharge_banks_revenue_and_lowers_soc() -> None:
    # 2.5 MW for 15 min = 0.625 MWh grid; lossless -> SoC drops 0.625.
    state, record = step(_state(soc_mwh=5.0), _snapshot(80.0), _fixed(energy_mw=2.5))
    assert record.energy_mwh == pytest.approx(0.625)
    assert record.energy_revenue == pytest.approx(0.625 * 80.0)
    assert record.soc_mwh_after == pytest.approx(5.0 - 0.625)
    assert state.revenue_by_source["energy"] == pytest.approx(0.625 * 80.0)


def test_charge_is_a_cost_and_raises_soc() -> None:
    state, record = step(_state(soc_mwh=5.0), _snapshot(10.0), _fixed(energy_mw=-2.5))
    assert record.energy_mwh == pytest.approx(-0.625)
    assert record.energy_revenue == pytest.approx(-0.625 * 10.0)  # negative = money out
    assert record.soc_mwh_after == pytest.approx(5.0 + 0.625)


def test_discharge_clamped_by_available_charge() -> None:
    # Almost empty: can only deliver what little SoC remains, not the full ask.
    state, record = step(_state(soc_mwh=0.1), _snapshot(80.0), _fixed(energy_mw=2.5))
    assert record.energy_mwh == pytest.approx(0.1)
    assert record.soc_mwh_after == pytest.approx(0.0)


def test_idle_banks_nothing_on_energy() -> None:
    state, record = step(_state(), _snapshot(35.0), _fixed(energy_mw=0.0))
    assert record.energy_mwh == 0.0
    assert record.energy_revenue == 0.0
    assert state.revenue_by_source["energy"] == 0.0


# --- AS leg ------------------------------------------------------------------


def test_as_standby_revenue_accrues_to_its_bucket() -> None:
    state, record = step(
        _state(),
        _snapshot(35.0, {"rrs": 9.0}),
        _fixed(energy_mw=0.0, as_product="rrs", as_mw=1.25),
    )
    assert record.as_revenue == pytest.approx(1.25 * 9.0)
    assert state.revenue_by_source["rrs"] == pytest.approx(1.25 * 9.0)
    assert state.revenue_by_source["energy"] == 0.0


def test_no_as_product_means_no_as_revenue() -> None:
    state, record = step(_state(), _snapshot(80.0), _fixed(energy_mw=2.5, as_product=None))
    assert record.as_revenue == 0.0
    assert sum(v for k, v in state.revenue_by_source.items() if k != "energy") == 0.0


def test_both_legs_bank_in_the_same_interval() -> None:
    state, record = step(
        _state(soc_mwh=5.0),
        _snapshot(80.0, {"regup": 6.0}),
        _fixed(energy_mw=1.25, as_product="regup", as_mw=1.25),
    )
    assert record.energy_revenue > 0
    assert record.as_revenue == pytest.approx(1.25 * 6.0)
    assert state.total_revenue == pytest.approx(record.energy_revenue + record.as_revenue)


# --- state progression -------------------------------------------------------


def test_commitment_and_interval_carry_into_state() -> None:
    state, _ = step(
        _state(),
        _snapshot(35.0, {"ecrs": 5.0}),
        _fixed(energy_mw=0.0, as_product="ecrs", as_mw=1.25),
    )
    assert state.last_commitment == ASCommitment(product="ecrs", mw=1.25, committed_at=T0)
    assert state.last_processed_interval == T0


def test_throughput_accumulates_across_steps() -> None:
    state = _state(soc_mwh=5.0)
    step(state, _snapshot(80.0), _fixed(energy_mw=2.5))
    first = state.throughput_internal_mwh
    step(state, _snapshot(80.0), _fixed(energy_mw=2.5))
    assert state.throughput_internal_mwh > first
    assert state.equivalent_full_cycles == pytest.approx(state.throughput_internal_mwh / 10.0)


def test_integrates_with_follow_the_leader_strategy() -> None:
    # End-to-end with the real strategy: high price + a positive MCPC ->
    # discharge on the energy leg and commit the leader on the AS leg.
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.5)
    state, record = step(_state(soc_mwh=5.0), _snapshot(80.0, {"regup": 4.0, "rrs": 9.0}), strategy)
    assert record.as_product == "rrs"
    assert record.as_mw == pytest.approx(1.25)  # 0.5 * 2.5
    assert record.energy_mwh > 0  # discharging into a high price
    assert record.as_revenue == pytest.approx(1.25 * 9.0)
