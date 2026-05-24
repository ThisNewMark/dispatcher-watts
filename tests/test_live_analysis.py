"""Tests for live capture-rate analysis (live/analysis.py)."""

from __future__ import annotations

import datetime as dt

import polars as pl

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA
from dispatcher_watts.live.analysis import live_capture_rate
from dispatcher_watts.live.engine import step
from dispatcher_watts.live.state import LiveState, records_to_frame
from dispatcher_watts.strategies.live import FollowTheLeaderStrategy, MarketSnapshot

T0 = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.UTC)
SPEC = BatterySpec(capacity_mwh=10.0, power_mw=2.5, round_trip_efficiency=0.9)


def _intervals(n: int) -> list[dt.datetime]:
    return [T0 + dt.timedelta(minutes=15 * i) for i in range(n)]


def _prices(starts: list[dt.datetime], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"interval_start": starts, "price": values}, schema=RTM_PRICE_SCHEMA)


def _mcpc(starts: list[dt.datetime], rrs: list[float]) -> pl.DataFrame:
    cols = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
    cols["mcpc_rrs"] = rrs
    return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA)


def _run_live(prices: pl.DataFrame, mcpc: pl.DataFrame) -> pl.DataFrame:
    """Replay the live strategy over the data and return its decision log."""
    state = LiveState(
        hub="HB_HOUSTON",
        spec=SPEC,
        strategy_name="follow-the-leader",
        strategy_config={},
        interval_minutes=15,
        soc_mwh=5.0,
    )
    strategy = FollowTheLeaderStrategy(20.0, 50.0, as_capacity_fraction=0.5)
    mcpc_lookup = {row["interval_start"]: row for row in mcpc.iter_rows(named=True)}
    records = []
    for row in prices.iter_rows(named=True):
        ts = row["interval_start"]
        m = mcpc_lookup.get(ts, {})
        snap = MarketSnapshot(ts, row["price"], {p: m.get(f"mcpc_{p}", 0.0) for p in MCPC_PRODUCTS})
        _, record = step(state, snap, strategy)
        records.append(record)
    return records_to_frame(records)


def test_capture_rate_is_between_zero_and_one() -> None:
    starts = _intervals(8)
    # Alternating cheap/expensive prices so arbitrage exists; steady RRS.
    prices = _prices(starts, [10.0, 90.0] * 4)
    mcpc = _mcpc(starts, [8.0] * 8)
    decisions = _run_live(prices, mcpc)

    result = live_capture_rate(decisions, prices, mcpc, SPEC)
    # The live rule cannot beat perfect foresight.
    assert result.actual_revenue <= result.ceiling_revenue + 1e-6
    assert 0.0 <= result.capture_rate <= 1.0
    assert result.intervals == 8


def test_by_source_buckets_sum_to_totals() -> None:
    starts = _intervals(8)
    prices = _prices(starts, [10.0, 90.0] * 4)
    mcpc = _mcpc(starts, [8.0] * 8)
    decisions = _run_live(prices, mcpc)

    result = live_capture_rate(decisions, prices, mcpc, SPEC)
    assert sum(result.actual_by_source.values()) == result.actual_revenue
    assert "energy" in result.actual_by_source
    assert "rrs" in result.actual_by_source


def test_capture_rate_zero_when_no_arbitrage() -> None:
    # Descending prices, no AS value: foresight earns nothing, so there is
    # nothing to capture and the rate is defined as 0.
    starts = _intervals(4)
    prices = _prices(starts, [90.0, 70.0, 50.0, 30.0])
    mcpc = _mcpc(starts, [0.0] * 4)
    decisions = _run_live(prices, mcpc)

    result = live_capture_rate(decisions, prices, mcpc, SPEC)
    assert result.capture_rate == 0.0
