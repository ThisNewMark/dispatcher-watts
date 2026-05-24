"""Clamp a submitted decision to what the battery could physically deliver.

Applied to every participant at execution time so the leaderboard rewards only
real-world-plausible strategies. Two constraints, mirroring the perfect-foresight
LP (``cooptimization/solver.py``):

* **Power sharing** -- energy dispatch and the AS reservation draw on one power
  rating: discharge + discharge-direction AS <= power, charge + RegDn <= power.
* **State-of-charge reservation** -- an AS commitment is limited to what the
  current charge could actually back if called for its required duration
  (discharge-direction products), or what headroom could absorb (RegDn).

Without this, a bot could be paid to "stand ready" with capacity it has no
charge to deliver -- free money no real battery earns. The clamp makes holding
an AS reserve cost the state of charge it ties up, which is the real
co-optimization tension.
"""

from __future__ import annotations

from dataclasses import replace

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.cooptimization.solver import AS_DURATION_HOURS
from dispatcher_watts.strategies.live import LiveDecision

# RegDn is the only charge-direction product (it absorbs energy if called); the
# rest discharge.
_CHARGE_DIRECTION_AS = "regdn"


def clamp_to_feasible(
    decision: LiveDecision,
    soc_mwh: float,
    spec: BatterySpec,
    interval_minutes: int,
    as_durations: dict[str, float] | None = None,
) -> LiveDecision:
    """Return `decision` reduced to a physically deliverable energy + AS pair.

    Energy is clamped to the battery's power and available charge/headroom; the
    AS reservation is then clamped to the power left over and the state of charge
    (after this interval's energy) that could back it for its delivery duration.
    """
    as_durations = as_durations or AS_DURATION_HOURS
    hours = interval_minutes / 60.0
    eff = spec.one_way_efficiency
    p_max = spec.power_mw
    cap = spec.capacity_mwh

    # 1. Feasible energy + the resulting end-of-interval state of charge.
    if decision.energy_mw >= 0:  # discharging (or idle)
        max_discharge_mw = min(p_max, soc_mwh * eff / hours) if hours > 0 else 0.0
        energy_mw = min(decision.energy_mw, max_discharge_mw)
        soc_after = soc_mwh - energy_mw * hours / eff
    else:  # charging
        max_charge_mw = min(p_max, (cap - soc_mwh) / eff / hours) if hours > 0 else 0.0
        energy_mw = -min(-decision.energy_mw, max_charge_mw)
        soc_after = soc_mwh + (-energy_mw) * hours * eff
    soc_after = min(cap, max(0.0, soc_after))

    # 2. Clamp the AS reservation to the shared power envelope + deliverable SoC.
    as_product = decision.as_product
    as_mw = decision.as_mw
    if as_product is not None and as_mw > 0.0:
        duration = as_durations.get(as_product, 1.0)
        if as_product == _CHARGE_DIRECTION_AS:
            power_room = p_max - max(0.0, -energy_mw)
            soc_room = (cap - soc_after) / (duration * eff) if duration > 0 else float("inf")
        else:
            power_room = p_max - max(0.0, energy_mw)
            soc_room = soc_after * eff / duration if duration > 0 else float("inf")
        as_mw = min(as_mw, max(0.0, power_room), max(0.0, soc_room))
        if as_mw <= 1e-9:
            as_product, as_mw = None, 0.0

    return LiveDecision(
        energy_mw=energy_mw,
        as_product=as_product,
        as_mw=as_mw,
        as_committed_at=decision.as_committed_at,
        reason=decision.reason,
    )


def apply_as_deployment(
    decision: LiveDecision, deployment_fractions: dict[str, float]
) -> LiveDecision:
    """Fold expected AS deployment into the energy leg.

    A committed AS product is "called" a fixed fraction of the time; the deployed
    power moves the battery (discharge-direction products discharge it, RegDn
    charges it). Folding that into the energy leg means the battery model moves
    the charge, settles the deployed energy at the interval price, and accrues
    the throughput the wear fee is charged on -- so committing AS now costs
    charge instead of being free standby income.
    """
    if decision.as_product is None or decision.as_mw <= 0.0:
        return decision
    fraction = deployment_fractions.get(decision.as_product, 0.0)
    if fraction <= 0.0:
        return decision
    deployed_mw = decision.as_mw * fraction
    if decision.as_product == _CHARGE_DIRECTION_AS:
        return replace(decision, energy_mw=decision.energy_mw - deployed_mw)
    return replace(decision, energy_mw=decision.energy_mw + deployed_mw)


__all__ = ["apply_as_deployment", "clamp_to_feasible"]
