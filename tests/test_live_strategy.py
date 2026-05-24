"""Tests for the deployable live strategies (strategies/live.py)."""

from __future__ import annotations

import datetime as dt

import pytest

from dispatcher_watts.strategies.live import (
    ASCommitment,
    FollowTheLeaderStrategy,
    MarketSnapshot,
)

T0 = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.UTC)


def _snapshot(
    price: float,
    mcpc: dict[str, float] | None = None,
    *,
    at: dt.datetime = T0,
) -> MarketSnapshot:
    return MarketSnapshot(timestamp=at, price=price, mcpc=mcpc or {})


# --- AS leg: leader selection ------------------------------------------------


def test_picks_highest_mcpc_product() -> None:
    strategy = FollowTheLeaderStrategy(charge_below=20.0, discharge_above=50.0)
    snap = _snapshot(35.0, {"regup": 4.0, "rrs": 9.0, "ecrs": 2.0})
    decision = strategy.decide(snap, soc_fraction=0.5, power_mw=2.5, held=None)
    assert decision.as_product == "rrs"


def test_as_mw_sized_by_capacity_fraction() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.4)
    snap = _snapshot(35.0, {"regup": 5.0})
    decision = strategy.decide(snap, 0.5, power_mw=2.5, held=None)
    assert decision.as_product == "regup"
    assert decision.as_mw == pytest.approx(1.0)  # 0.4 * 2.5


def test_no_positive_mcpc_means_no_as_leg() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.5)
    snap = _snapshot(35.0, {"regup": 0.0, "rrs": -1.0})
    decision = strategy.decide(snap, 0.5, power_mw=2.5, held=None)
    assert decision.as_product is None
    assert decision.as_mw == 0.0


def test_empty_mcpc_means_no_as_leg() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0)
    decision = strategy.decide(_snapshot(35.0, {}), 0.5, power_mw=2.5, held=None)
    assert decision.as_product is None


def test_tie_breaks_toward_earlier_product_in_canonical_order() -> None:
    # regup precedes rrs in MCPC_PRODUCTS; equal quotes -> regup wins.
    strategy = FollowTheLeaderStrategy(20.0, 50.0)
    snap = _snapshot(35.0, {"rrs": 7.0, "regup": 7.0})
    decision = strategy.decide(snap, 0.5, power_mw=2.5, held=None)
    assert decision.as_product == "regup"


def test_unknown_products_in_mcpc_are_ignored() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0)
    snap = _snapshot(35.0, {"made_up": 99.0, "regup": 3.0})
    decision = strategy.decide(snap, 0.5, power_mw=2.5, held=None)
    assert decision.as_product == "regup"


# --- energy leg: threshold ---------------------------------------------------


def test_energy_charges_below_threshold() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.0)
    decision = strategy.decide(_snapshot(15.0), 0.5, power_mw=2.5, held=None)
    assert decision.energy_mw == pytest.approx(-2.5)


def test_energy_discharges_above_threshold() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.0)
    decision = strategy.decide(_snapshot(80.0), 0.5, power_mw=2.5, held=None)
    assert decision.energy_mw == pytest.approx(2.5)


def test_energy_idles_inside_band() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.0)
    decision = strategy.decide(_snapshot(35.0), 0.5, power_mw=2.5, held=None)
    assert decision.energy_mw == 0.0


def test_energy_power_is_reduced_by_as_reservation() -> None:
    # 0.6 of 2.5 MW reserved for AS -> 1.0 MW left for energy.
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.6)
    snap = _snapshot(80.0, {"regup": 5.0})
    decision = strategy.decide(snap, 0.5, power_mw=2.5, held=None)
    assert decision.as_mw == pytest.approx(1.5)
    assert decision.energy_mw == pytest.approx(1.0)


def test_energy_gets_full_power_when_no_as_committed() -> None:
    # MCPC all non-positive -> no AS reservation -> energy uses full rating
    # even though as_capacity_fraction would otherwise reserve half.
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.5)
    snap = _snapshot(80.0, {"regup": 0.0})
    decision = strategy.decide(snap, 0.5, power_mw=2.5, held=None)
    assert decision.energy_mw == pytest.approx(2.5)


# --- allocation interval -----------------------------------------------------


def test_option1_repicks_leader_every_tick() -> None:
    # Default 5-min interval == one tick: a new leader is adopted immediately.
    strategy = FollowTheLeaderStrategy(20.0, 50.0, allocation_interval_minutes=5.0)
    held = ASCommitment(product="regup", mw=1.25, committed_at=T0)
    later = _snapshot(35.0, {"regup": 1.0, "rrs": 9.0}, at=T0 + dt.timedelta(minutes=5))
    decision = strategy.decide(later, 0.5, power_mw=2.5, held=held)
    assert decision.as_product == "rrs"
    assert decision.as_committed_at == later.timestamp


def test_option2_holds_leader_within_allocation_interval() -> None:
    # 60-min interval: 30 min later we still hold regup even though rrs now leads.
    strategy = FollowTheLeaderStrategy(20.0, 50.0, allocation_interval_minutes=60.0)
    held = ASCommitment(product="regup", mw=1.25, committed_at=T0)
    later = _snapshot(35.0, {"regup": 1.0, "rrs": 9.0}, at=T0 + dt.timedelta(minutes=30))
    decision = strategy.decide(later, 0.5, power_mw=2.5, held=held)
    assert decision.as_product == "regup"
    assert decision.as_mw == pytest.approx(1.25)
    assert decision.as_committed_at == T0  # carried, not refreshed


def test_repicks_once_allocation_interval_elapses() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, allocation_interval_minutes=60.0)
    held = ASCommitment(product="regup", mw=1.25, committed_at=T0)
    later = _snapshot(35.0, {"regup": 1.0, "rrs": 9.0}, at=T0 + dt.timedelta(minutes=60))
    decision = strategy.decide(later, 0.5, power_mw=2.5, held=held)
    assert decision.as_product == "rrs"
    assert decision.as_committed_at == later.timestamp


def test_held_none_product_does_not_block_repick() -> None:
    # A held leg with product None is not a real commitment; re-pick freely.
    strategy = FollowTheLeaderStrategy(20.0, 50.0, allocation_interval_minutes=60.0)
    held = ASCommitment(product=None, mw=0.0, committed_at=T0)
    later = _snapshot(35.0, {"rrs": 9.0}, at=T0 + dt.timedelta(minutes=1))
    decision = strategy.decide(later, 0.5, power_mw=2.5, held=held)
    assert decision.as_product == "rrs"


def test_commitment_round_trips_into_next_tick() -> None:
    strategy = FollowTheLeaderStrategy(20.0, 50.0, allocation_interval_minutes=60.0)
    first = strategy.decide(_snapshot(35.0, {"ecrs": 6.0}), 0.5, 2.5, held=None)
    held = first.commitment()
    assert held.product == "ecrs"
    assert held.committed_at == T0


# --- validation --------------------------------------------------------------


def test_rejects_overlapping_thresholds() -> None:
    with pytest.raises(ValueError, match="must be less than"):
        FollowTheLeaderStrategy(charge_below=50.0, discharge_above=20.0)


def test_rejects_capacity_fraction_out_of_range() -> None:
    with pytest.raises(ValueError, match="as_capacity_fraction"):
        FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=1.5)


def test_rejects_nonpositive_allocation_interval() -> None:
    with pytest.raises(ValueError, match="allocation_interval_minutes"):
        FollowTheLeaderStrategy(20.0, 50.0, allocation_interval_minutes=0)


def test_name() -> None:
    assert FollowTheLeaderStrategy(20.0, 50.0).name == "follow-the-leader"
