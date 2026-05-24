"""Tests for the transport-agnostic competition service (competition/service.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from dispatcher_watts.competition import COMPETITION_SPEC
from dispatcher_watts.competition.service import (
    AuthError,
    CompetitionService,
    DecisionError,
)
from dispatcher_watts.competition.store import CompetitionStore, QueuedDecision
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA

NOW = dt.datetime(2026, 5, 22, 13, 0, tzinfo=dt.UTC)


class _FakeSource:
    def __init__(self, prices: pl.DataFrame, mcpc: pl.DataFrame) -> None:
        self._prices = prices
        self._mcpc = mcpc

    def get_rtm_prices_window(self, start: dt.datetime, end: dt.datetime, hub: str) -> pl.DataFrame:
        return self._prices.filter(
            (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
        )

    def get_indicative_mcpc_window(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        return self._mcpc.filter(
            (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
        )


def _intervals(n: int) -> list[dt.datetime]:
    base = NOW - dt.timedelta(minutes=15 * n)
    return [base + dt.timedelta(minutes=15 * i) for i in range(n)]


def _source() -> _FakeSource:
    starts = _intervals(4)
    cols = {f"mcpc_{p}": [0.0] * 4 for p in MCPC_PRODUCTS}
    cols["mcpc_rrs"] = [9.0] * 4
    return _FakeSource(
        pl.DataFrame({"interval_start": starts, "price": [80.0] * 4}, schema=RTM_PRICE_SCHEMA),
        pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA),
    )


def _service(tmp_path: Path) -> CompetitionService:
    store = CompetitionStore(tmp_path / "comp.db")
    return CompetitionService(store, _source(), data_dir=tmp_path / "live", lookback_hours=1.0)


# --- registration / auth -----------------------------------------------------


def test_register_returns_token_and_battery(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    out = svc.register("alice-bot", email="alice@example.com")
    assert out["token"]
    assert out["battery"]["capacity_mwh"] == COMPETITION_SPEC.capacity_mwh


def test_submit_with_bad_token_raises(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    with pytest.raises(AuthError):
        svc.submit_decision("nope", NOW + dt.timedelta(minutes=15), 1.0, now=NOW)


# --- submit_decision validation ----------------------------------------------


def test_submit_valid_decision_is_queued(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    token = str(svc.register("bot")["token"])
    target = NOW + dt.timedelta(minutes=15)
    out = svc.submit_decision(token, target, energy_mw=2.0, as_product="rrs", as_mw=1.0, now=NOW)
    assert out["accepted"] is True


def test_submit_for_started_interval_is_rejected(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    token = str(svc.register("bot")["token"])
    # Interval already started (== now) -> past the cutoff.
    with pytest.raises(DecisionError, match="cutoff"):
        svc.submit_decision(token, NOW, energy_mw=1.0, now=NOW)


def test_submit_rejects_energy_over_power(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    token = str(svc.register("bot")["token"])
    target = NOW + dt.timedelta(minutes=15)
    with pytest.raises(DecisionError, match="power rating"):
        svc.submit_decision(token, target, energy_mw=999.0, now=NOW)


def test_submit_rejects_unknown_as_product(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    token = str(svc.register("bot")["token"])
    target = NOW + dt.timedelta(minutes=15)
    with pytest.raises(DecisionError, match="as_product"):
        svc.submit_decision(token, target, 0.0, as_product="made_up", as_mw=1.0, now=NOW)


# --- observation / state / leaderboard ---------------------------------------


def _seed_revenue(svc: CompetitionService, store: CompetitionStore, name: str) -> str:
    """Register a participant (with charge to back AS) and queue a winning AS decision."""
    out = svc.register(name)
    pid = str(out["participant_id"])
    participant = store.get_participant(pid)
    assert participant is not None
    participant.soc_mwh = 5.0  # enough charge to back the RRS reservation
    store.save_participant_state(participant)
    first = _intervals(4)[0]
    store.queue_decision(
        QueuedDecision(pid, first, 0.0, "rrs", 1.25, submitted_at=first - dt.timedelta(minutes=5))
    )
    return str(out["token"])


def test_get_my_state_reports_revenue_after_catch_up(tmp_path: Path) -> None:
    store = CompetitionStore(tmp_path / "comp.db")
    svc = CompetitionService(store, _source(), data_dir=tmp_path / "live", lookback_hours=1.0)
    token = _seed_revenue(svc, store, "earner")
    svc.catch_up(now=NOW)

    state = svc.get_my_state(token)
    assert state["total_revenue"] > 0
    assert state["revenue_by_source"]["ancillary"] > 0
    assert state["intervals"] == 4


def test_get_observation_returns_market_and_deadline(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    token = str(svc.register("bot")["token"])
    obs = svc.get_observation(token, now=NOW)
    assert obs["latest_settled"] is not None
    # Next open interval is the one after the current clock interval.
    assert obs["next_decision_interval"] == (NOW + dt.timedelta(minutes=15)).isoformat()
    assert obs["battery"]["capacity_mwh"] == COMPETITION_SPEC.capacity_mwh


def test_get_leaderboard_ranks_participants(tmp_path: Path) -> None:
    store = CompetitionStore(tmp_path / "comp.db")
    svc = CompetitionService(store, _source(), data_dir=tmp_path / "live", lookback_hours=1.0)
    _seed_revenue(svc, store, "earner")
    svc.register("idler")
    svc.catch_up(now=NOW)

    board = svc.get_leaderboard(window="day", sort_by="revenue", now=NOW)
    entries = board["entries"]
    assert entries[0]["display_name"] == "earner"
    assert entries[0]["rank"] == 1
    assert entries[0]["revenue"] > 0


def test_get_leaderboard_rejects_unknown_window(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    with pytest.raises(DecisionError, match="unknown window"):
        svc.get_leaderboard(window="century", now=NOW)
