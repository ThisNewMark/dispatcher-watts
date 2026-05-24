"""The reference "house bot" -- follow-the-leader, run as a real participant.

So every leaderboard has a baseline to beat from day one, we run our deployable
follow-the-leader strategy as an ordinary competitor. It is not special-cased in
the engine: a driver simply computes its decision from the latest market info
and queues it for the next open interval, exactly as an external agent would.

(The *perfect-foresight* reference is the leaderboard's ceiling -- everyone's
capture rate is measured against it -- not a competitor: it cannot run live
because it would have to see future prices.)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.competition import COMPETITION_HUB, COMPETITION_SPEC
from dispatcher_watts.competition.store import CompetitionStore, Participant, QueuedDecision
from dispatcher_watts.data.live_capture import (
    DEFAULT_LIVE_DIR,
    load_captured_mcpc_window,
    load_captured_prices_window,
)
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, RTM_INTERVAL_MINUTES
from dispatcher_watts.strategies.live import (
    ASCommitment,
    FollowTheLeaderStrategy,
    LiveStrategy,
    MarketSnapshot,
)

HOUSE_BOT_NAME = "house: follow-the-leader"
_HOUSE_BOT_META_KEY = "house_bot_id"


def ensure_house_bot(store: CompetitionStore) -> Participant:
    """Return the house-bot participant, registering it once if needed."""
    participant_id = store.get_market_meta(_HOUSE_BOT_META_KEY)
    if participant_id is not None:
        existing = store.get_participant(participant_id)
        if existing is not None:
            return existing
    bot = store.register_participant(HOUSE_BOT_NAME)
    store.set_market_meta(_HOUSE_BOT_META_KEY, bot.id)
    return bot


def drive_house_bot(
    store: CompetitionStore,
    *,
    hub: str = COMPETITION_HUB,
    spec: BatterySpec = COMPETITION_SPEC,
    interval_minutes: int = RTM_INTERVAL_MINUTES,
    data_dir: Path = DEFAULT_LIVE_DIR,
    now: dt.datetime,
    strategy: LiveStrategy | None = None,
) -> dt.datetime | None:
    """Queue the house bot's decision for the next open interval.

    Reads the latest captured price + indicative MCPC, runs the strategy, and
    queues the result (submitted before the interval's cutoff). Returns the
    interval it queued for, or ``None`` if no market data is available yet.
    """
    bot = ensure_house_bot(store)
    strategy = strategy or FollowTheLeaderStrategy(charge_below=20.0, discharge_above=50.0)
    snapshot = _latest_snapshot(hub, interval_minutes, data_dir, now)
    if snapshot is None:
        return None

    held = (
        ASCommitment(bot.held_as_product, bot.held_as_mw, bot.held_committed_at or bot.created_at)
        if bot.held_as_product is not None
        else None
    )
    decision = strategy.decide(snapshot, bot.soc_mwh / spec.capacity_mwh, spec.power_mw, held)
    target = next_open_interval(now, interval_minutes)
    store.queue_decision(
        QueuedDecision(
            participant_id=bot.id,
            interval_start=target,
            energy_mw=decision.energy_mw,
            as_product=decision.as_product,
            as_mw=decision.as_mw,
            submitted_at=now,
        )
    )
    return target


def next_open_interval(now: dt.datetime, interval_minutes: int) -> dt.datetime:
    """Start of the next interval a decision may still be submitted for."""
    minutes = (now.minute // interval_minutes + 1) * interval_minutes
    return now.replace(minute=0, second=0, microsecond=0) + dt.timedelta(minutes=minutes)


def _latest_snapshot(
    hub: str, interval_minutes: int, data_dir: Path, now: dt.datetime
) -> MarketSnapshot | None:
    window_start = now - dt.timedelta(hours=2)
    prices = load_captured_prices_window(window_start, now, hub, data_dir)
    if prices.is_empty():
        return None
    price_row = prices.sort("interval_start").row(-1, named=True)

    mcpc = load_captured_mcpc_window(window_start, now, data_dir)
    on_grid = mcpc.filter((pl.col("interval_start").dt.minute() % interval_minutes) == 0)
    mcpc_dict: dict[str, float] = {}
    if not on_grid.is_empty():
        mcpc_row = on_grid.sort("interval_start").row(-1, named=True)
        mcpc_dict = {p: mcpc_row[f"mcpc_{p}"] for p in MCPC_PRODUCTS}

    return MarketSnapshot(timestamp=now, price=price_row["price"], mcpc=mcpc_dict)


__all__ = ["HOUSE_BOT_NAME", "drive_house_bot", "ensure_house_bot", "next_open_interval"]
