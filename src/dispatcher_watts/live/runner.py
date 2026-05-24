"""Orchestration for one `simulate-live` invocation.

`run_once` is the testable core the CLI wraps: given a data source, it figures
out which intervals are new since the last run, fetches their prices and
indicative MCPCs, captures the raw observations, steps the engine over each new
interval, and persists the updated state and decision log. An external scheduler
calling this every few minutes is the whole live loop -- state on disk is what
makes that safe across process restarts.

The data source must expose the two windowed live fetches
(`get_rtm_prices_window`, `get_indicative_mcpc_window`); `ErcotDirectSource`
does. The simulator deliberately uses ERCOT-direct for everything to avoid the
gridstatus row budget.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.data.live_capture import append_mcpc, append_prices
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, RTM_INTERVAL_MINUTES
from dispatcher_watts.live.engine import step
from dispatcher_watts.live.state import (
    DecisionRecord,
    LiveState,
    append_decisions,
    load_state,
    save_state,
    state_exists,
)
from dispatcher_watts.strategies.live import (
    FollowTheLeaderStrategy,
    LiveStrategy,
    MarketSnapshot,
)


class LiveDataSource(Protocol):
    """The two windowed fetches the live loop needs (ErcotDirectSource fits)."""

    def get_rtm_prices_window(
        self, start: dt.datetime, end: dt.datetime, hub: str
    ) -> pl.DataFrame: ...

    def get_indicative_mcpc_window(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame: ...


@dataclass
class LiveConfig:
    """Everything needed to start (or, on resume, identify) a live run."""

    hub: str
    spec: BatterySpec
    strategy_name: str
    strategy_config: dict[str, float]
    interval_minutes: int = RTM_INTERVAL_MINUTES
    lookback_hours: float = 3.0
    initial_soc_mwh: float = 0.0


@dataclass
class RunSummary:
    """Outcome of one `run_once` call, for the CLI to print."""

    intervals_processed: int
    window_start: dt.datetime
    window_end: dt.datetime
    state: LiveState


def make_strategy(name: str, config: dict[str, float]) -> LiveStrategy:
    """Build a live strategy from its persisted name + config."""
    if name == FollowTheLeaderStrategy.name:
        return FollowTheLeaderStrategy(
            charge_below=config["charge_below"],
            discharge_above=config["discharge_above"],
            as_capacity_fraction=config.get("as_capacity_fraction", 0.5),
            allocation_interval_minutes=config.get("allocation_interval_minutes", 5.0),
        )
    raise ValueError(f"unknown live strategy {name!r}")


def _init_state(config: LiveConfig) -> LiveState:
    return LiveState(
        hub=config.hub,
        spec=config.spec,
        strategy_name=config.strategy_name,
        strategy_config=config.strategy_config,
        interval_minutes=config.interval_minutes,
        soc_mwh=config.initial_soc_mwh,
    )


def _mcpc_lookup(mcpc: pl.DataFrame, interval_minutes: int) -> dict[dt.datetime, dict[str, float]]:
    """Index indicative MCPCs onto the settlement-interval grid.

    The indicative feed is 5-minute; energy settles (and the LP clears) on the
    15-minute grid. We keep the rows that land on a settlement-interval start
    and use each as that interval's AS price -- the same row a 15-min/5-min
    inner join would select, so a live run and its LP ceiling stay consistent.
    """
    on_grid = mcpc.filter((pl.col("interval_start").dt.minute() % interval_minutes) == 0)
    lookup: dict[dt.datetime, dict[str, float]] = {}
    for row in on_grid.iter_rows(named=True):
        lookup[row["interval_start"]] = {p: row[f"mcpc_{p}"] for p in MCPC_PRODUCTS}
    return lookup


def run_once(
    source: LiveDataSource,
    config: LiveConfig,
    *,
    state_dir: Path,
    data_dir: Path,
    now: dt.datetime | None = None,
) -> RunSummary:
    """Fetch new intervals, step the engine over them, and persist results."""
    now = now or dt.datetime.now(dt.UTC)
    state = load_state(state_dir) if state_exists(state_dir) else _init_state(config)
    interval = dt.timedelta(minutes=state.interval_minutes)

    if state.last_processed_interval is not None:
        window_start = state.last_processed_interval + interval
    else:
        window_start = now - dt.timedelta(hours=config.lookback_hours)

    if window_start >= now:
        return RunSummary(0, window_start, now, state)

    prices = source.get_rtm_prices_window(window_start, now, config.hub)
    mcpc = source.get_indicative_mcpc_window(window_start, now)

    # Capture every raw observation before doing anything else, so the run
    # leaves a complete replayable record even if stepping later changes.
    append_prices(prices, config.hub, data_dir)
    append_mcpc(mcpc, data_dir)

    if state.last_processed_interval is not None:
        prices = prices.filter(pl.col("interval_start") > state.last_processed_interval)
    prices = prices.sort("interval_start")

    strategy = make_strategy(state.strategy_name, state.strategy_config)
    mcpc_lookup = _mcpc_lookup(mcpc, state.interval_minutes)

    records: list[DecisionRecord] = []
    for row in prices.iter_rows(named=True):
        ts = row["interval_start"]
        snapshot = MarketSnapshot(timestamp=ts, price=row["price"], mcpc=mcpc_lookup.get(ts, {}))
        _, record = step(state, snapshot, strategy)
        records.append(record)

    append_decisions(records, state_dir)
    save_state(state, state_dir)
    return RunSummary(len(records), window_start, now, state)


__all__ = [
    "LiveConfig",
    "LiveDataSource",
    "RunSummary",
    "make_strategy",
    "run_once",
]
