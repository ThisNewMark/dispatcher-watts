"""Tests for the deliverability clamp (competition/feasibility.py)."""

from __future__ import annotations

import datetime as dt

import pytest

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.competition.feasibility import apply_as_deployment, clamp_to_feasible
from dispatcher_watts.strategies.live import LiveDecision

T0 = dt.datetime(2026, 5, 23, 13, 0, tzinfo=dt.UTC)
# Lossless 10 MWh / 2.5 MW so the arithmetic is clean (eff = 1.0).
SPEC = BatterySpec(capacity_mwh=10.0, power_mw=2.5, round_trip_efficiency=1.0)


def _decision(energy_mw: float, as_product: str | None = None, as_mw: float = 0.0) -> LiveDecision:
    return LiveDecision(
        energy_mw=energy_mw,
        as_product=as_product,
        as_mw=as_mw,
        as_committed_at=T0,
        reason="x",
    )


def _clamp(decision: LiveDecision, soc_mwh: float) -> LiveDecision:
    return clamp_to_feasible(decision, soc_mwh, SPEC, interval_minutes=15)


# --- SoC reservation ---------------------------------------------------------


def test_empty_battery_cannot_commit_discharge_as() -> None:
    # The degenerate exploit: stack full AS on an empty battery -> clamped away.
    out = _clamp(_decision(0.0, "regup", 2.5), soc_mwh=0.0)
    assert out.as_product is None
    assert out.as_mw == 0.0


def test_full_battery_cannot_commit_regdn() -> None:
    # RegDn needs headroom to absorb; a full battery has none.
    out = _clamp(_decision(0.0, "regdn", 2.5), soc_mwh=10.0)
    assert out.as_product is None


def test_regup_clamped_to_deliverable_soc() -> None:
    # RegUp duration 1h, lossless: deliverable MW = soc / 1h. soc=1.5 -> 1.5 MW.
    out = _clamp(_decision(0.0, "regup", 2.5), soc_mwh=1.5)
    assert out.as_product == "regup"
    assert out.as_mw == pytest.approx(1.5)


def test_short_duration_product_barely_binds_on_soc() -> None:
    # RRS duration 0.25h: deliverable MW = soc / 0.25 = 4*soc. soc=1.0 -> 4 MW,
    # but power caps it at 2.5, so the full request stands.
    out = _clamp(_decision(0.0, "rrs", 2.5), soc_mwh=1.0)
    assert out.as_mw == pytest.approx(2.5)


# --- power sharing -----------------------------------------------------------


def test_full_discharge_leaves_no_power_for_discharge_as() -> None:
    out = _clamp(_decision(2.5, "rrs", 2.5), soc_mwh=10.0)
    assert out.as_product is None  # 2.5 MW energy uses the whole envelope


def test_partial_discharge_shares_power_with_as() -> None:
    # 1.0 MW discharge leaves 1.5 MW for the AS leg (SoC is ample).
    out = _clamp(_decision(1.0, "rrs", 2.5), soc_mwh=10.0)
    assert out.energy_mw == pytest.approx(1.0)
    assert out.as_mw == pytest.approx(1.5)


def test_charging_does_not_block_discharge_direction_as() -> None:
    # Charging uses the charge envelope; RRS (discharge-direction) is limited by
    # SoC/power on the discharge side, not by the charge power. soc=2.0 leaves
    # room to charge and enough charge to back the RRS reservation.
    out = _clamp(_decision(-2.5, "rrs", 2.5), soc_mwh=2.0)
    assert out.energy_mw == pytest.approx(-2.5)
    assert out.as_mw == pytest.approx(2.5)


def test_charging_shares_power_with_regdn() -> None:
    out = _clamp(_decision(-2.0, "regdn", 2.5), soc_mwh=0.0)
    assert out.energy_mw == pytest.approx(-2.0)
    assert out.as_mw == pytest.approx(0.5)  # 2.5 - 2.0 charge


# --- energy clamping ---------------------------------------------------------


def test_energy_discharge_clamped_to_available_charge() -> None:
    # soc 0.1 MWh, 15 min -> max discharge = 0.1/0.25 = 0.4 MW.
    out = _clamp(_decision(2.5), soc_mwh=0.1)
    assert out.energy_mw == pytest.approx(0.4)


def test_feasible_decision_passes_through_unchanged() -> None:
    out = _clamp(_decision(1.0, "rrs", 1.0), soc_mwh=8.0)
    assert out.energy_mw == pytest.approx(1.0)
    assert out.as_product == "rrs"
    assert out.as_mw == pytest.approx(1.0)


# --- AS deployment -----------------------------------------------------------

_FRACTIONS = {"regup": 0.2, "regdn": 0.2, "rrs": 0.1, "ecrs": 0.1, "nspin": 0.1}


def test_deployment_discharges_for_discharge_direction_as() -> None:
    # Idle energy + 2.0 MW RegUp at 20% deployment -> +0.4 MW forced discharge.
    out = apply_as_deployment(_decision(0.0, "regup", 2.0), _FRACTIONS)
    assert out.energy_mw == pytest.approx(0.4)
    assert out.as_mw == pytest.approx(2.0)  # capacity commitment unchanged


def test_deployment_charges_for_regdn() -> None:
    out = apply_as_deployment(_decision(0.0, "regdn", 2.0), _FRACTIONS)
    assert out.energy_mw == pytest.approx(-0.4)  # forced charge (absorb)


def test_deployment_offsets_existing_energy() -> None:
    # Charging -1.0 with RegUp deployed +0.4 -> net -0.6 (curtailed charge).
    out = apply_as_deployment(_decision(-1.0, "regup", 2.0), _FRACTIONS)
    assert out.energy_mw == pytest.approx(-0.6)


def test_no_as_means_no_deployment() -> None:
    out = apply_as_deployment(_decision(1.0, None, 0.0), _FRACTIONS)
    assert out.energy_mw == pytest.approx(1.0)
