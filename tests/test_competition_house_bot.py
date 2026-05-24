"""Tests for the reference house bot + heartbeat (house_bot.py, service.run_heartbeat)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from dispatcher_watts.competition import COMPETITION_SPEC
from dispatcher_watts.competition.house_bot import (
    HOUSE_BOT_NAME,
    drive_house_bot,
    ensure_house_bot,
    next_open_interval,
)
from dispatcher_watts.competition.service import CompetitionService
from dispatcher_watts.competition.store import CompetitionStore
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA

NOW = dt.datetime(2026, 5, 23, 13, 3, tzinfo=dt.UTC)


class _FakeSource:
    def __init__(self, price: float = 80.0, rrs: float = 9.0) -> None:
        self._price, self._rrs = price, rrs

    @staticmethod
    def _grid(start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
        # 15-min-grid-aligned timestamps in [start, end), like real ERCOT data.
        t = start.replace(minute=(start.minute // 15) * 15, second=0, microsecond=0)
        out: list[dt.datetime] = []
        while t < end:
            if t >= start:
                out.append(t)
            t += dt.timedelta(minutes=15)
        return out

    def get_rtm_prices_window(self, start: dt.datetime, end: dt.datetime, hub: str) -> pl.DataFrame:
        starts = self._grid(start, end)
        return pl.DataFrame(
            {"interval_start": starts, "price": [self._price] * len(starts)},
            schema=RTM_PRICE_SCHEMA,
        )

    def get_indicative_mcpc_window(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        starts = self._grid(start, end)
        cols = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
        cols["mcpc_rrs"] = [self._rrs] * len(starts)
        return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA)


def _service(tmp_path: Path) -> CompetitionService:
    store = CompetitionStore(tmp_path / "comp.db")
    return CompetitionService(store, _FakeSource(), data_dir=tmp_path / "live", lookback_hours=1.0)


# --- ensure / next interval --------------------------------------------------


def test_ensure_house_bot_is_idempotent(tmp_path: Path) -> None:
    store = CompetitionStore(tmp_path / "comp.db")
    first = ensure_house_bot(store)
    second = ensure_house_bot(store)
    assert first.id == second.id
    assert first.display_name == HOUSE_BOT_NAME
    assert len(store.list_participants()) == 1


def test_next_open_interval_rounds_up() -> None:
    # 13:03 on a 15-min grid -> next open interval is 13:15.
    assert next_open_interval(NOW, 15) == dt.datetime(2026, 5, 23, 13, 15, tzinfo=dt.UTC)


def test_next_open_interval_on_boundary() -> None:
    boundary = dt.datetime(2026, 5, 23, 13, 0, tzinfo=dt.UTC)
    assert next_open_interval(boundary, 15) == dt.datetime(2026, 5, 23, 13, 15, tzinfo=dt.UTC)


# --- driving -----------------------------------------------------------------


def test_drive_queues_decision_for_next_interval(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.catch_up(now=NOW)  # capture some market data first
    target = drive_house_bot(svc._store, data_dir=tmp_path / "live", now=NOW, spec=COMPETITION_SPEC)
    assert target == dt.datetime(2026, 5, 23, 13, 15, tzinfo=dt.UTC)
    bot = ensure_house_bot(svc._store)
    queued = svc._store.get_queued_decision(bot.id, target)
    assert queued is not None
    # High price + positive RRS -> discharge energy + commit to RRS.
    assert queued.as_product == "rrs"
    assert queued.submitted_at == NOW  # before the interval's cutoff


def test_drive_without_market_data_is_noop(tmp_path: Path) -> None:
    store = CompetitionStore(tmp_path / "comp.db")
    # No catch_up, so the capture log is empty.
    target = drive_house_bot(store, data_dir=tmp_path / "live", now=NOW)
    assert target is None


# --- heartbeat ---------------------------------------------------------------


def test_run_heartbeat_advances_and_drives(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    summary = svc.run_heartbeat(now=NOW)
    assert summary.intervals_processed >= 1  # market advanced
    # House bot exists and has a decision queued for the next interval.
    bot = ensure_house_bot(svc._store)
    target = next_open_interval(NOW, 15)
    assert svc._store.get_queued_decision(bot.id, target) is not None


def test_house_bot_appears_on_leaderboard_after_heartbeats(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    # First heartbeat seeds the bot + queues a decision for the next interval.
    svc.run_heartbeat(now=NOW)
    # A later heartbeat settles that interval, so the bot has a logged decision.
    later = NOW + dt.timedelta(minutes=30)
    svc.run_heartbeat(now=later)
    board = svc.get_leaderboard(window="day", now=later)
    names = [e["display_name"] for e in board["entries"]]
    assert HOUSE_BOT_NAME in names
