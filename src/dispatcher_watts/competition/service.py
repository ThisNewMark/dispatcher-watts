"""Transport-agnostic competition service -- the logic the MCP server wraps.

Every participant action goes through here: register, observe, submit a
decision, read state, read the leaderboard. The methods take/return plain
dicts and raise ``AuthError`` / ``DecisionError`` for the caller to translate,
so the same service backs an MCP server today and a REST API or web UI later
(mirroring how ``run_once`` backs the live CLI).

The service owns the lazy market advance: ``get_observation`` and
``get_leaderboard`` first catch the market up to now (fetch newly-settled
intervals, step every participant), so a participant always sees current state
without a separate always-on worker. A periodic heartbeat call keeps the market
current during quiet periods.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.competition import COMPETITION_HUB, COMPETITION_SPEC
from dispatcher_watts.competition.engine import AdvanceSummary, advance_market
from dispatcher_watts.competition.house_bot import drive_house_bot
from dispatcher_watts.competition.leaderboard import SORT_KEYS, WINDOWS, compute_leaderboard
from dispatcher_watts.competition.store import CompetitionStore, Participant, QueuedDecision
from dispatcher_watts.data.live_capture import (
    DEFAULT_LIVE_DIR,
    load_captured_mcpc_window,
    load_captured_prices_window,
)
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, RTM_INTERVAL_MINUTES
from dispatcher_watts.live.runner import LiveDataSource


class AuthError(Exception):
    """Raised when a token does not match any participant."""


class DecisionError(Exception):
    """Raised when a submitted decision is invalid or past its cutoff."""


class CompetitionService:
    """Application logic for the competition, independent of any transport."""

    def __init__(
        self,
        store: CompetitionStore,
        source: LiveDataSource,
        *,
        hub: str = COMPETITION_HUB,
        spec: BatterySpec = COMPETITION_SPEC,
        interval_minutes: int = RTM_INTERVAL_MINUTES,
        data_dir: Path = DEFAULT_LIVE_DIR,
        lookback_hours: float = 1.0,
    ) -> None:
        self._store = store
        self._source = source
        self._hub = hub
        self._spec = spec
        self._interval_minutes = interval_minutes
        self._data_dir = data_dir
        self._lookback_hours = lookback_hours

    # --- registration / auth ----------------------------------------------

    def register(self, display_name: str, email: str | None = None) -> dict[str, object]:
        """Create a participant; return their id + secret token (shown once)."""
        participant = self._store.register_participant(display_name, email)
        return {
            "participant_id": participant.id,
            "display_name": participant.display_name,
            "token": participant.token,
            "battery": _spec_dict(self._spec),
            "hub": self._hub,
        }

    def _authenticate(self, token: str) -> Participant:
        participant = self._store.get_participant_by_token(token)
        if participant is None:
            raise AuthError("invalid token")
        return participant

    # --- market advance ----------------------------------------------------

    def catch_up(self, now: dt.datetime | None = None) -> AdvanceSummary:
        """Advance the market over any intervals that settled since last time."""
        return advance_market(
            self._store,
            self._source,
            hub=self._hub,
            spec=self._spec,
            interval_minutes=self._interval_minutes,
            data_dir=self._data_dir,
            lookback_hours=self._lookback_hours,
            now=now,
        )

    def run_heartbeat(self, now: dt.datetime | None = None) -> AdvanceSummary:
        """One scheduler tick: advance the market, then drive the house bot.

        Meant to be called periodically (Railway cron / a worker loop). The
        lazy catch-up keeps the market current; driving the house bot here keeps
        the reference baseline submitting decisions even when no one is calling.
        """
        now = now or dt.datetime.now(dt.UTC)
        summary = self.catch_up(now)
        drive_house_bot(
            self._store,
            hub=self._hub,
            spec=self._spec,
            interval_minutes=self._interval_minutes,
            data_dir=self._data_dir,
            now=now,
        )
        return summary

    # --- participant actions ----------------------------------------------

    def submit_decision(
        self,
        token: str,
        interval_start: dt.datetime,
        energy_mw: float,
        as_product: str | None = None,
        as_mw: float = 0.0,
        now: dt.datetime | None = None,
    ) -> dict[str, object]:
        """Queue a decision for an upcoming interval (must clear the cutoff)."""
        participant = self._authenticate(token)
        now = now or dt.datetime.now(dt.UTC)
        self._validate_decision(interval_start, energy_mw, as_product, as_mw, now)
        self._store.queue_decision(
            QueuedDecision(
                participant_id=participant.id,
                interval_start=interval_start,
                energy_mw=energy_mw,
                as_product=as_product,
                as_mw=as_mw,
                submitted_at=now,
            )
        )
        return {"accepted": True, "interval_start": interval_start.isoformat()}

    def get_observation(self, token: str, now: dt.datetime | None = None) -> dict[str, object]:
        """Current market + own state + next decision deadline."""
        participant = self._authenticate(token)
        now = now or dt.datetime.now(dt.UTC)
        self.catch_up(now)
        participant = self._authenticate(token)  # reload post-advance state
        latest = self._latest_market_row(now)
        next_interval = self._next_open_interval(now)
        return {
            "now": now.isoformat(),
            "latest_settled": latest,
            "next_decision_interval": next_interval.isoformat(),
            "battery": {
                "soc_mwh": participant.soc_mwh,
                "capacity_mwh": self._spec.capacity_mwh,
                "power_mw": self._spec.power_mw,
                "held_as_product": participant.held_as_product,
                "held_as_mw": participant.held_as_mw,
            },
        }

    def get_my_state(self, token: str) -> dict[str, object]:
        """Battery state + cumulative revenue-by-source for the caller."""
        participant = self._authenticate(token)
        revenue = self._revenue_by_source(participant.id)
        return {
            "participant_id": participant.id,
            "display_name": participant.display_name,
            "soc_mwh": participant.soc_mwh,
            "equivalent_full_cycles": participant.throughput_internal_mwh / self._spec.capacity_mwh,
            "intervals": self._store.count_decisions(participant.id),
            "revenue_by_source": revenue,
            "total_revenue": sum(revenue.values()),
        }

    def get_leaderboard(
        self,
        window: str = "all-time",
        sort_by: str = "revenue",
        now: dt.datetime | None = None,
    ) -> dict[str, object]:
        """Ranked standings for a window (no token required -- public)."""
        if window not in WINDOWS:
            raise DecisionError(f"unknown window {window!r}; expected one of {WINDOWS}")
        if sort_by not in SORT_KEYS:
            raise DecisionError(f"unknown sort_by {sort_by!r}; expected one of {SORT_KEYS}")
        now = now or dt.datetime.now(dt.UTC)
        self.catch_up(now)
        prices, mcpc = self._captured_tape(window, now)
        board = compute_leaderboard(
            self._store,
            prices,
            mcpc,
            window=window,
            now=now,
            spec=self._spec,
            interval_minutes=self._interval_minutes,
            sort_by=sort_by,
        )
        return {
            "window": board.window,
            "start": board.start.isoformat(),
            "end": board.end.isoformat(),
            "ceiling_revenue": board.ceiling_revenue,
            "sort_by": sort_by,
            "entries": [
                {
                    "rank": i + 1,
                    "display_name": e.display_name,
                    "revenue": e.revenue,
                    "gross_revenue": e.gross_revenue,
                    "degradation_cost": e.degradation_cost,
                    "capture_rate": e.capture_rate,
                    "equivalent_cycles": e.equivalent_cycles,
                    "energy_revenue": e.energy_revenue,
                    "as_revenue": e.as_revenue,
                    "intervals": e.intervals,
                }
                for i, e in enumerate(board.entries)
            ],
        }

    # --- helpers -----------------------------------------------------------

    def _validate_decision(
        self,
        interval_start: dt.datetime,
        energy_mw: float,
        as_product: str | None,
        as_mw: float,
        now: dt.datetime,
    ) -> None:
        if interval_start.tzinfo is None:
            raise DecisionError("interval_start must be timezone-aware")
        if now >= interval_start:
            raise DecisionError(
                "interval has already started (or passed) -- decisions must be "
                "submitted before the interval's information cutoff"
            )
        if abs(energy_mw) > self._spec.power_mw + 1e-9:
            raise DecisionError(f"|energy_mw| exceeds power rating {self._spec.power_mw}")
        if as_product is not None and as_product not in MCPC_PRODUCTS:
            raise DecisionError(
                f"unknown as_product {as_product!r}; expected one of {MCPC_PRODUCTS}"
            )
        if as_mw < 0 or as_mw > self._spec.power_mw + 1e-9:
            raise DecisionError(f"as_mw must be in [0, {self._spec.power_mw}]")

    def _next_open_interval(self, now: dt.datetime) -> dt.datetime:
        """Start of the next interval a decision may still be submitted for."""
        minutes = (now.minute // self._interval_minutes + 1) * self._interval_minutes
        floored = now.replace(minute=0, second=0, microsecond=0)
        return floored + dt.timedelta(minutes=minutes)

    def _latest_market_row(self, now: dt.datetime) -> dict[str, object] | None:
        prices = load_captured_prices_window(
            now - dt.timedelta(hours=2), now, self._hub, self._data_dir
        )
        if prices.is_empty():
            return None
        row = prices.sort("interval_start").row(-1, named=True)
        return {"interval_start": row["interval_start"].isoformat(), "price": row["price"]}

    def _captured_tape(self, window: str, now: dt.datetime) -> tuple[pl.DataFrame, pl.DataFrame]:
        from dispatcher_watts.competition.leaderboard import window_bounds

        start, end = window_bounds(window, now)
        prices = load_captured_prices_window(start, end, self._hub, self._data_dir)
        mcpc = load_captured_mcpc_window(start, end, self._data_dir)
        return prices, mcpc

    def _revenue_by_source(self, participant_id: str) -> dict[str, float]:
        cursor = self._store.connection.execute(
            "SELECT COALESCE(SUM(energy_revenue), 0) AS e, COALESCE(SUM(as_revenue), 0) AS a "
            "FROM decision_log WHERE participant_id = ?",
            (participant_id,),
        )
        row = cursor.fetchone()
        return {"energy": float(row["e"]), "ancillary": float(row["a"])}


def _spec_dict(spec: BatterySpec) -> dict[str, float]:
    return {
        "capacity_mwh": spec.capacity_mwh,
        "power_mw": spec.power_mw,
        "round_trip_efficiency": spec.round_trip_efficiency,
    }


__all__ = ["AuthError", "CompetitionService", "DecisionError"]
