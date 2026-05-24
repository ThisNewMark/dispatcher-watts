"""The live simulator's per-interval step -- a pure function.

``step`` is the live analogue of ``backtest/engine.py``'s loop body, but for one
interval at a time and with two revenue legs instead of one. It does no I/O: it
takes the current state plus a market snapshot, asks the strategy for a
decision, executes it on a battery rebuilt from state, banks revenue from both
energy and ancillary services, and returns the updated state plus a record of
what happened. The caller (the CLI) is responsible for fetching the snapshot and
persisting the results.

Revenue conventions match the co-optimization LP (``cooptimization/solver.py``)
so a live run and its perfect-foresight ceiling are directly comparable:

    energy_revenue = grid_energy_mwh * price        (+discharge / -charge)
    as_revenue     = as_mw * mcpc[as_product]       (per-interval standby pay)
"""

from __future__ import annotations

from dispatcher_watts.live.state import DecisionRecord, LiveState
from dispatcher_watts.strategies.live import LiveStrategy, MarketSnapshot


def step(
    state: LiveState,
    snapshot: MarketSnapshot,
    strategy: LiveStrategy,
) -> tuple[LiveState, DecisionRecord]:
    """Advance `state` by one interval against `snapshot`; return it plus a record.

    `state` is mutated in place and also returned for convenience. The battery's
    physical constraints are enforced by reusing the battery model, so the energy
    leg is clamped to what is actually feasible (e.g. a discharge request larger
    than the available charge delivers only what is there).
    """
    hours = state.interval_minutes / 60.0
    battery = state.to_battery()
    decision = strategy.decide(
        snapshot,
        soc_fraction=battery.soc_fraction,
        power_mw=state.spec.power_mw,
        held=state.last_commitment,
    )

    # Energy leg: signed average power -> grid-side MWh, clamped by the battery.
    if decision.energy_mw < 0:
        energy_mwh = -battery.charge(-decision.energy_mw * hours, hours)
    elif decision.energy_mw > 0:
        energy_mwh = battery.discharge(decision.energy_mw * hours, hours)
    else:
        energy_mwh = 0.0
    energy_revenue = energy_mwh * snapshot.price

    # AS leg: a per-interval standby payment for the committed capacity, earned
    # whether or not the product is actually called.
    if decision.as_product is not None:
        as_revenue = decision.as_mw * snapshot.mcpc.get(decision.as_product, 0.0)
    else:
        as_revenue = 0.0

    state.adopt_battery(battery)
    state.revenue_by_source["energy"] += energy_revenue
    if decision.as_product is not None:
        state.revenue_by_source[decision.as_product] += as_revenue
    state.last_commitment = decision.commitment()
    state.last_processed_interval = snapshot.timestamp

    record = DecisionRecord(
        interval_start=snapshot.timestamp,
        price=snapshot.price,
        mcpc=snapshot.mcpc,
        energy_mw=decision.energy_mw,
        energy_mwh=energy_mwh,
        as_product=decision.as_product,
        as_mw=decision.as_mw,
        energy_revenue=energy_revenue,
        as_revenue=as_revenue,
        soc_mwh_after=battery.soc_mwh,
        reason=decision.reason,
    )
    return state, record


__all__ = ["step"]
