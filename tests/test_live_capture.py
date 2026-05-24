"""Tests for the append-only live-capture log (data/live_capture.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from dispatcher_watts.data.live_capture import (
    append_mcpc,
    append_prices,
    load_captured_mcpc_window,
    load_captured_prices_window,
    prices_dir,
)
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA


def _prices(rows: list[tuple[dt.datetime, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {"interval_start": [r[0] for r in rows], "price": [r[1] for r in rows]},
        schema=RTM_PRICE_SCHEMA,
    )


def _mcpc(starts: list[dt.datetime], regup: list[float]) -> pl.DataFrame:
    cols: dict[str, list[float]] = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
    cols["mcpc_regup"] = regup
    return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA)


T0 = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.UTC)
WINDOW_END = dt.datetime(2026, 5, 24, 0, 0, tzinfo=dt.UTC)
WINDOW_START = dt.datetime(2026, 5, 22, 0, 0, tzinfo=dt.UTC)


def test_append_then_load_round_trips(tmp_path: Path) -> None:
    df = _prices([(T0, 30.0), (T0 + dt.timedelta(minutes=15), 45.0)])
    append_prices(df, "HB_HOUSTON", data_dir=tmp_path)
    loaded = load_captured_prices_window(WINDOW_START, WINDOW_END, "HB_HOUSTON", data_dir=tmp_path)
    assert loaded.schema == RTM_PRICE_SCHEMA
    assert loaded["price"].to_list() == [30.0, 45.0]


def test_append_partitions_across_utc_day_boundary(tmp_path: Path) -> None:
    day1 = dt.datetime(2026, 5, 22, 23, 45, tzinfo=dt.UTC)
    day2 = dt.datetime(2026, 5, 23, 0, 0, tzinfo=dt.UTC)
    append_prices(_prices([(day1, 10.0), (day2, 20.0)]), "HB_NORTH", data_dir=tmp_path)
    files = sorted(p.name for p in prices_dir("HB_NORTH", tmp_path).glob("*.parquet"))
    assert files == ["2026-05-22.parquet", "2026-05-23.parquet"]


def test_reappending_same_interval_keeps_latest(tmp_path: Path) -> None:
    # Indicative MCPC: a later capture of the same interval is a fresher
    # projection and must overwrite the earlier one.
    append_mcpc(_mcpc([T0], regup=[5.0]), data_dir=tmp_path)
    append_mcpc(_mcpc([T0], regup=[9.0]), data_dir=tmp_path)
    loaded = load_captured_mcpc_window(WINDOW_START, WINDOW_END, data_dir=tmp_path)
    assert loaded.height == 1
    assert loaded["mcpc_regup"].to_list() == [9.0]


def test_load_window_is_half_open(tmp_path: Path) -> None:
    inside = T0
    on_end = WINDOW_END
    append_prices(_prices([(inside, 1.0), (on_end, 2.0)]), "HB_SOUTH", data_dir=tmp_path)
    loaded = load_captured_prices_window(WINDOW_START, WINDOW_END, "HB_SOUTH", data_dir=tmp_path)
    # [start, end): the row exactly on `end` is excluded.
    assert loaded["price"].to_list() == [1.0]


def test_load_window_skips_missing_days(tmp_path: Path) -> None:
    # Only one day captured; querying a multi-day window must not raise.
    append_prices(_prices([(T0, 7.0)]), "HB_WEST", data_dir=tmp_path)
    wide_end = dt.datetime(2026, 5, 30, 0, 0, tzinfo=dt.UTC)
    loaded = load_captured_prices_window(WINDOW_START, wide_end, "HB_WEST", data_dir=tmp_path)
    assert loaded["price"].to_list() == [7.0]


def test_load_window_empty_when_nothing_captured(tmp_path: Path) -> None:
    loaded = load_captured_prices_window(WINDOW_START, WINDOW_END, "HB_HOUSTON", data_dir=tmp_path)
    assert loaded.is_empty()
    assert loaded.schema == RTM_PRICE_SCHEMA


def test_append_empty_frame_is_noop(tmp_path: Path) -> None:
    append_prices(_prices([]), "HB_HOUSTON", data_dir=tmp_path)
    assert not prices_dir("HB_HOUSTON", tmp_path).exists()


def test_load_window_rejects_naive_datetimes(tmp_path: Path) -> None:
    naive = dt.datetime(2026, 5, 22, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        load_captured_prices_window(naive, WINDOW_END, "HB_HOUSTON", data_dir=tmp_path)


def test_load_window_rejects_inverted_range(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be before"):
        load_captured_prices_window(WINDOW_END, WINDOW_START, "HB_HOUSTON", data_dir=tmp_path)
