"""Tests for the live-run orchestration (live/runner.py), with a fake source."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA
from dispatcher_watts.live.runner import LiveConfig, make_strategy, run_once
from dispatcher_watts.live.state import load_decisions, load_state
from dispatcher_watts.strategies.live import FollowTheLeaderStrategy

NOW = dt.datetime(2026, 5, 22, 13, 0, tzinfo=dt.UTC)


class _FakeSource:
    """Serves canned price + indicative-MCPC frames, recording the windows asked."""

    def __init__(self, prices: pl.DataFrame, mcpc: pl.DataFrame) -> None:
        self._prices = prices
        self._mcpc = mcpc
        self.price_calls: list[tuple[dt.datetime, dt.datetime, str]] = []

    def get_rtm_prices_window(self, start: dt.datetime, end: dt.datetime, hub: str) -> pl.DataFrame:
        self.price_calls.append((start, end, hub))
        return self._prices.filter(
            (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
        )

    def get_indicative_mcpc_window(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        return self._mcpc.filter(
            (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
        )


def _prices_at(starts: list[dt.datetime], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"interval_start": starts, "price": values}, schema=RTM_PRICE_SCHEMA)


def _mcpc_at(starts: list[dt.datetime], rrs: list[float]) -> pl.DataFrame:
    cols = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
    cols["mcpc_rrs"] = rrs
    return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA)


def _intervals(n: int) -> list[dt.datetime]:
    base = NOW - dt.timedelta(minutes=15 * n)
    return [base + dt.timedelta(minutes=15 * i) for i in range(n)]


def _config(**overrides: object) -> LiveConfig:
    base = LiveConfig(
        hub="HB_HOUSTON",
        spec=BatterySpec(capacity_mwh=10.0, power_mw=2.5, round_trip_efficiency=1.0),
        strategy_name="follow-the-leader",
        strategy_config={
            "charge_below": 20.0,
            "discharge_above": 50.0,
            "as_capacity_fraction": 0.5,
        },
        lookback_hours=1.0,
        initial_soc_mwh=5.0,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_first_run_processes_lookback_window(tmp_path: Path) -> None:
    starts = _intervals(4)  # 1h lookback at 15-min = 4 intervals
    source = _FakeSource(_prices_at(starts, [80.0] * 4), _mcpc_at(starts, [9.0] * 4))
    summary = run_once(
        source, _config(), state_dir=tmp_path / "s", data_dir=tmp_path / "d", now=NOW
    )

    assert summary.intervals_processed == 4
    assert summary.state.last_processed_interval == starts[-1]
    # High prices + positive RRS -> energy and AS both earned.
    assert summary.state.revenue_by_source["energy"] > 0
    assert summary.state.revenue_by_source["rrs"] > 0


def test_state_and_decisions_persist(tmp_path: Path) -> None:
    starts = _intervals(4)
    source = _FakeSource(_prices_at(starts, [80.0] * 4), _mcpc_at(starts, [9.0] * 4))
    run_once(source, _config(), state_dir=tmp_path / "s", data_dir=tmp_path / "d", now=NOW)

    reloaded = load_state(tmp_path / "s")
    assert reloaded.last_processed_interval == starts[-1]
    log = load_decisions(tmp_path / "s")
    assert log.height == 4


def test_captured_data_is_written(tmp_path: Path) -> None:
    starts = _intervals(4)
    source = _FakeSource(_prices_at(starts, [80.0] * 4), _mcpc_at(starts, [9.0] * 4))
    run_once(source, _config(), state_dir=tmp_path / "s", data_dir=tmp_path / "d", now=NOW)
    # The capture log under data_dir should now hold the day's price file.
    captured = list((tmp_path / "d").rglob("*.parquet"))
    assert captured, "expected captured market data on disk"


def test_second_run_resumes_without_reprocessing(tmp_path: Path) -> None:
    starts = _intervals(4)
    source = _FakeSource(_prices_at(starts, [80.0] * 4), _mcpc_at(starts, [9.0] * 4))
    state_dir, data_dir = tmp_path / "s", tmp_path / "d"
    run_once(source, _config(), state_dir=state_dir, data_dir=data_dir, now=NOW)

    # 30 min later, two new intervals arrive.
    later = NOW + dt.timedelta(minutes=30)
    new_starts = [NOW, NOW + dt.timedelta(minutes=15)]
    all_starts = starts + new_starts
    source2 = _FakeSource(_prices_at(all_starts, [80.0] * 6), _mcpc_at(all_starts, [9.0] * 6))
    summary = run_once(source2, _config(), state_dir=state_dir, data_dir=data_dir, now=later)
    assert summary.intervals_processed == 2  # only the two new ones
    assert load_decisions(state_dir).height == 6  # 4 + 2, no duplicates
    # The resume fetch starts just after the last processed interval.
    assert source2.price_calls[0][0] == starts[-1] + dt.timedelta(minutes=15)


def test_no_new_intervals_is_a_noop(tmp_path: Path) -> None:
    starts = _intervals(4)
    source = _FakeSource(_prices_at(starts, [80.0] * 4), _mcpc_at(starts, [9.0] * 4))
    state_dir, data_dir = tmp_path / "s", tmp_path / "d"
    run_once(source, _config(), state_dir=state_dir, data_dir=data_dir, now=NOW)
    # Re-run at the same instant: nothing new to process.
    summary = run_once(source, _config(), state_dir=state_dir, data_dir=data_dir, now=NOW)
    assert summary.intervals_processed == 0


def test_make_strategy_builds_follow_the_leader() -> None:
    strat = make_strategy(
        "follow-the-leader",
        {"charge_below": 20.0, "discharge_above": 50.0, "allocation_interval_minutes": 60.0},
    )
    assert isinstance(strat, FollowTheLeaderStrategy)
    assert strat.allocation_interval_minutes == 60.0


def test_make_strategy_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown live strategy"):
        make_strategy("nope", {})
