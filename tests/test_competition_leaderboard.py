"""Tests for time-windowed leaderboards (competition/leaderboard.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from dispatcher_watts.competition import COMPETITION_SPEC
from dispatcher_watts.competition.engine import advance_market
from dispatcher_watts.competition.leaderboard import (
    compute_leaderboard,
    window_bounds,
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


def _prices(starts: list[dt.datetime], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"interval_start": starts, "price": values}, schema=RTM_PRICE_SCHEMA)


def _mcpc(starts: list[dt.datetime], rrs: list[float]) -> pl.DataFrame:
    cols = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
    cols["mcpc_rrs"] = rrs
    return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA)


def _setup(tmp_path: Path) -> tuple[CompetitionStore, dict[str, str], pl.DataFrame, pl.DataFrame]:
    """A store with an active trader + an idle participant, advanced over 4 intervals."""
    starts = _intervals(4)
    # Alternating cheap/expensive prices so energy arbitrage is possible.
    prices = _prices(starts, [10.0, 90.0, 10.0, 90.0])
    mcpc = _mcpc(starts, [9.0, 9.0, 9.0, 9.0])
    store = CompetitionStore(tmp_path / "comp.db")
    trader = store.register_participant("trader").id
    store.register_participant("idle")
    ids = {p.display_name: p.id for p in store.list_participants()}

    # Trader: discharge into the high prices, charge on the cheap ones, plus RRS.
    for s, price in zip(starts, [10.0, 90.0, 10.0, 90.0], strict=True):
        energy = COMPETITION_SPEC.power_mw if price >= 50 else -COMPETITION_SPEC.power_mw
        store.queue_decision(
            QueuedDecision(trader, s, energy, "rrs", 1.25, submitted_at=s - dt.timedelta(minutes=5))
        )

    advance_market(
        store, _FakeSource(prices, mcpc), now=NOW, lookback_hours=1.0, data_dir=tmp_path / "live"
    )
    return store, ids, prices, mcpc


# --- window bounds -----------------------------------------------------------


def test_window_bounds_hour() -> None:
    start, end = window_bounds("hour", NOW)
    assert start == dt.datetime(2026, 5, 22, 13, 0, tzinfo=dt.UTC)
    assert end == NOW


def test_window_bounds_day() -> None:
    start, _ = window_bounds("day", NOW)
    assert start == dt.datetime(2026, 5, 22, 0, 0, tzinfo=dt.UTC)


def test_window_bounds_month() -> None:
    start, _ = window_bounds("month", NOW)
    assert start == dt.datetime(2026, 5, 1, 0, 0, tzinfo=dt.UTC)


def test_window_bounds_week_is_monday() -> None:
    # 2026-05-22 is a Friday; the ISO week starts Monday 2026-05-18.
    start, _ = window_bounds("week", NOW)
    assert start == dt.datetime(2026, 5, 18, 0, 0, tzinfo=dt.UTC)


def test_window_bounds_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown window"):
        window_bounds("decade", NOW)


# --- ranking -----------------------------------------------------------------


def test_active_trader_outranks_idle_by_revenue(tmp_path: Path) -> None:
    store, ids, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(store, prices, mcpc, window="day", now=NOW, sort_by="revenue")
    assert [e.display_name for e in board.entries] == ["trader", "idle"]
    assert board.entries[0].revenue > board.entries[1].revenue
    assert board.entries[1].revenue == 0.0  # idle earned nothing


def test_capture_rate_is_bounded_and_uses_ceiling(tmp_path: Path) -> None:
    store, _, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(store, prices, mcpc, window="day", now=NOW, sort_by="capture_rate")
    assert board.ceiling_revenue > 0
    top = board.entries[0]
    # Cannot beat perfect foresight (epsilon for floating-point: an optimal
    # play lands at exactly 1.0 and may overshoot by a rounding bit).
    assert 0.0 <= top.capture_rate <= 1.0 + 1e-6
    assert top.revenue <= board.ceiling_revenue + 1e-6


def test_sort_by_capture_rate_orders_descending(tmp_path: Path) -> None:
    store, _, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(store, prices, mcpc, window="day", now=NOW, sort_by="capture_rate")
    rates = [e.capture_rate for e in board.entries]
    assert rates == sorted(rates, reverse=True)


def test_entry_reports_cycles_and_split(tmp_path: Path) -> None:
    store, _, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(store, prices, mcpc, window="day", now=NOW)
    trader = next(e for e in board.entries if e.display_name == "trader")
    assert trader.equivalent_cycles > 0  # it discharged
    assert trader.as_revenue > 0  # earned RRS standby
    assert trader.gross_revenue == pytest.approx(trader.energy_revenue + trader.as_revenue)
    assert trader.intervals == 4


def test_empty_window_has_zero_ceiling_and_revenue(tmp_path: Path) -> None:
    store, _, prices, mcpc = _setup(tmp_path)
    # An hour window in the far future: no intervals, no tape.
    future = NOW + dt.timedelta(days=400)
    board = compute_leaderboard(store, prices, mcpc, window="hour", now=future)
    assert board.ceiling_revenue == 0.0
    assert all(e.revenue == 0.0 for e in board.entries)
    assert all(e.capture_rate == 0.0 for e in board.entries)


def test_wear_fee_reduces_score_below_gross(tmp_path: Path) -> None:
    # The trader cycles, so its score is gross minus a wear fee on throughput.
    store, ids, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(store, prices, mcpc, window="day", now=NOW)
    trader = next(e for e in board.entries if e.display_name == "trader")
    assert trader.degradation_cost > 0  # it moved energy
    assert trader.revenue == pytest.approx(trader.gross_revenue - trader.degradation_cost)
    assert trader.revenue < trader.gross_revenue


def test_idle_participant_pays_no_wear_fee(tmp_path: Path) -> None:
    store, ids, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(store, prices, mcpc, window="day", now=NOW)
    idle = next(e for e in board.entries if e.display_name == "idle")
    assert idle.degradation_cost == 0.0  # moved no energy
    assert idle.revenue == idle.gross_revenue


def test_high_wear_fee_sinks_a_thin_spread_cycler(tmp_path: Path) -> None:
    # With a punishing wear fee, the cycling trader's score goes negative even
    # though its gross is positive -- exactly the over-cycling we want to deter.
    store, ids, prices, mcpc = _setup(tmp_path)
    board = compute_leaderboard(
        store, prices, mcpc, window="day", now=NOW, degradation_cost_per_mwh=1000.0
    )
    trader = next(e for e in board.entries if e.display_name == "trader")
    assert trader.gross_revenue > 0
    assert trader.revenue < 0


def test_rejects_unknown_sort_key(tmp_path: Path) -> None:
    store, _, prices, mcpc = _setup(tmp_path)
    with pytest.raises(ValueError, match="unknown sort_by"):
        compute_leaderboard(store, prices, mcpc, window="day", now=NOW, sort_by="vibes")
