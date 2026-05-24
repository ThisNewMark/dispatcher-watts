"""Advance the competition market: step every participant over the shared tape.

This is the multi-participant generalization of ``live/runner.run_once``. One
shared ERCOT fetch per catch-up; then, for each newly-settled interval, every
participant's battery is stepped using their pre-cutoff queued decision -- or
the default-fill (idle energy, hold the existing AS commitment) when they have
no valid one.

The anti-cheat rule lives here: a queued decision for interval T is honored only
if it was submitted strictly before T's start (``submitted_at < interval_start``),
regardless of when the server actually processes T. You cannot move on an
interval whose price you could already know.

Per-interval execution is delegated to the single-player engine's ``step`` (and
thus the battery model), so the physics and revenue accounting are identical to
the rest of the project.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.competition import (
    COMPETITION_AS_DEPLOYMENT_FRACTION,
    COMPETITION_HUB,
    COMPETITION_SPEC,
)
from dispatcher_watts.competition.feasibility import apply_as_deployment, clamp_to_feasible
from dispatcher_watts.competition.store import CompetitionStore, Participant
from dispatcher_watts.cooptimization.solver import AS_DURATION_HOURS
from dispatcher_watts.data.live_capture import DEFAULT_LIVE_DIR, append_mcpc, append_prices
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, RTM_INTERVAL_MINUTES
from dispatcher_watts.live.engine import step
from dispatcher_watts.live.runner import LiveDataSource
from dispatcher_watts.live.state import LiveState
from dispatcher_watts.strategies.live import (
    ASCommitment,
    LiveDecision,
    LiveStrategy,
    MarketSnapshot,
)


@dataclass
class AdvanceSummary:
    """Outcome of one ``advance_market`` catch-up."""

    window_start: dt.datetime
    window_end: dt.datetime
    intervals_processed: int
    participants: int


class _FixedDecisionStrategy(LiveStrategy):
    """Returns a pre-resolved decision -- the participant already decided."""

    name = "fixed"

    def __init__(self, decision: LiveDecision) -> None:
        self._decision = decision

    def decide(
        self,
        snapshot: MarketSnapshot,
        soc_fraction: float,
        power_mw: float,
        held: ASCommitment | None,
    ) -> LiveDecision:
        return self._decision


def advance_market(
    store: CompetitionStore,
    source: LiveDataSource,
    *,
    hub: str = COMPETITION_HUB,
    spec: BatterySpec = COMPETITION_SPEC,
    interval_minutes: int = RTM_INTERVAL_MINUTES,
    lookback_hours: float = 1.0,
    data_dir: Path = DEFAULT_LIVE_DIR,
    enforce_deliverability: bool = True,
    as_durations: dict[str, float] | None = None,
    model_as_deployment: bool = True,
    deployment_fractions: dict[str, float] | None = None,
    now: dt.datetime | None = None,
) -> AdvanceSummary:
    """Process every interval newly settled since the last catch-up.

    Fetches the shared market tape once, captures it, and steps all participants
    over each new interval. The market clock advances regardless of how many
    participants there are, so the captured tape and leaderboard windows stay
    current even during quiet periods.
    """
    now = now or dt.datetime.now(dt.UTC)
    durations = as_durations or AS_DURATION_HOURS
    deployment = deployment_fractions or COMPETITION_AS_DEPLOYMENT_FRACTION
    interval = dt.timedelta(minutes=interval_minutes)
    last_processed = store.get_last_processed_interval()
    window_start = (
        last_processed + interval
        if last_processed is not None
        else now - dt.timedelta(hours=lookback_hours)
    )
    if window_start >= now:
        return AdvanceSummary(window_start, now, 0, len(store.list_participants()))

    prices = source.get_rtm_prices_window(window_start, now, hub)
    mcpc = source.get_indicative_mcpc_window(window_start, now)
    append_prices(prices, hub, data_dir)
    append_mcpc(mcpc, data_dir)

    if last_processed is not None:
        prices = prices.filter(pl.col("interval_start") > last_processed)
    prices = prices.sort("interval_start")

    participants = store.list_participants()
    states = {p.id: _state_from_participant(p, hub, spec, interval_minutes) for p in participants}
    mcpc_lookup = _mcpc_on_grid(mcpc, interval_minutes)

    processed = 0
    last_interval: dt.datetime | None = None
    for row in prices.iter_rows(named=True):
        ts = row["interval_start"]
        snapshot = MarketSnapshot(timestamp=ts, price=row["price"], mcpc=mcpc_lookup.get(ts, {}))
        for participant in participants:
            state = states[participant.id]
            decision, was_default = _resolve_decision(
                store, participant.id, ts, state.last_commitment
            )
            if enforce_deliverability:
                decision = clamp_to_feasible(
                    decision, state.soc_mwh, spec, interval_minutes, durations
                )
            if model_as_deployment:
                # Expected AS calls move charge -- folded into the energy leg so
                # the battery settles it and the wear fee applies to the recharge.
                decision = apply_as_deployment(decision, deployment)
            _, record = step(state, snapshot, _FixedDecisionStrategy(decision))
            store.record_decision(
                participant_id=participant.id,
                interval_start=ts,
                price=record.price,
                energy_mw=record.energy_mw,
                energy_mwh=record.energy_mwh,
                as_product=record.as_product,
                as_mw=record.as_mw,
                energy_revenue=record.energy_revenue,
                as_revenue=record.as_revenue,
                soc_mwh_after=record.soc_mwh_after,
                was_default=was_default,
                reason=record.reason,
            )
        processed += 1
        last_interval = ts

    for participant in participants:
        _write_back_state(participant, states[participant.id])
        store.save_participant_state(participant)
    if last_interval is not None:
        store.set_last_processed_interval(last_interval)

    return AdvanceSummary(window_start, now, processed, len(participants))


def _resolve_decision(
    store: CompetitionStore,
    participant_id: str,
    interval_start: dt.datetime,
    held: ASCommitment | None,
) -> tuple[LiveDecision, bool]:
    """The participant's valid queued decision, or the default-fill.

    Returns ``(decision, was_default)``. A queued decision counts only if it was
    submitted strictly before the interval started (the information cutoff).
    """
    queued = store.get_queued_decision(participant_id, interval_start)
    if queued is not None and queued.submitted_at < interval_start:
        return (
            LiveDecision(
                energy_mw=queued.energy_mw,
                as_product=queued.as_product,
                as_mw=queued.as_mw,
                as_committed_at=interval_start,
                reason="queued",
            ),
            False,
        )
    # Default-fill: idle energy, hold whatever AS commitment already stands.
    return (
        LiveDecision(
            energy_mw=0.0,
            as_product=held.product if held else None,
            as_mw=held.mw if held else 0.0,
            as_committed_at=held.committed_at if held else interval_start,
            reason="default-fill: idle energy, hold AS",
        ),
        True,
    )


def _state_from_participant(
    participant: Participant, hub: str, spec: BatterySpec, interval_minutes: int
) -> LiveState:
    held = (
        ASCommitment(
            participant.held_as_product,
            participant.held_as_mw,
            participant.held_committed_at or participant.created_at,
        )
        if participant.held_as_product is not None
        else None
    )
    return LiveState(
        hub=hub,
        spec=spec,
        strategy_name="competition",
        strategy_config={},
        interval_minutes=interval_minutes,
        soc_mwh=participant.soc_mwh,
        energy_charged_mwh=participant.energy_charged_mwh,
        energy_discharged_mwh=participant.energy_discharged_mwh,
        throughput_internal_mwh=participant.throughput_internal_mwh,
        last_commitment=held,
    )


def _write_back_state(participant: Participant, state: LiveState) -> None:
    participant.soc_mwh = state.soc_mwh
    participant.energy_charged_mwh = state.energy_charged_mwh
    participant.energy_discharged_mwh = state.energy_discharged_mwh
    participant.throughput_internal_mwh = state.throughput_internal_mwh
    commitment = state.last_commitment
    participant.held_as_product = commitment.product if commitment else None
    participant.held_as_mw = commitment.mw if commitment else 0.0
    participant.held_committed_at = commitment.committed_at if commitment else None


def _mcpc_on_grid(mcpc: pl.DataFrame, interval_minutes: int) -> dict[dt.datetime, dict[str, float]]:
    """Index indicative MCPCs onto the settlement-interval grid (see runner)."""
    on_grid = mcpc.filter((pl.col("interval_start").dt.minute() % interval_minutes) == 0)
    return {
        row["interval_start"]: {p: row[f"mcpc_{p}"] for p in MCPC_PRODUCTS}
        for row in on_grid.iter_rows(named=True)
    }


__all__ = ["AdvanceSummary", "advance_market"]
