"""Tests for the ERCOT data layer: schema, normalization, and parquet cache."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from dispatcher_watts.data.ercot import GridstatusERCOTSource, normalize_rtm_frame
from dispatcher_watts.data.schemas import RTM_PRICE_SCHEMA, validate_rtm_frame
from dispatcher_watts.data.store import (
    is_cached,
    load_prices,
    save_prices,
    summarize_prices,
)


def _raw_gridstatus_frame() -> pl.DataFrame:
    """A frame shaped like gridstatus.io's `ercot_spp_real_time_15_min` output."""
    starts = [
        dt.datetime(2025, 1, 1, 6, 0, tzinfo=dt.UTC),
        dt.datetime(2025, 1, 1, 6, 15, tzinfo=dt.UTC),
        dt.datetime(2025, 1, 1, 6, 30, tzinfo=dt.UTC),
    ]
    return pl.DataFrame({"interval_start_utc": starts, "spp": [22.5, -8.0, 410.25]})


class _FakeClient:
    """Stand-in for `gridstatusio.GridStatusClient` -- avoids network calls."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame
        self.calls: list[dict[str, object]] = []

    def get_dataset(self, **kwargs: object) -> pl.DataFrame:
        self.calls.append(kwargs)
        return self._frame


def test_normalize_rtm_frame_matches_schema() -> None:
    out = normalize_rtm_frame(_raw_gridstatus_frame())
    assert out.schema == RTM_PRICE_SCHEMA
    assert out.columns == ["interval_start", "price"]
    assert out.height == 3


def test_normalize_rtm_frame_sorts_by_interval() -> None:
    out = normalize_rtm_frame(_raw_gridstatus_frame().reverse())
    assert out["interval_start"].is_sorted()
    assert out["price"].to_list() == [22.5, -8.0, 410.25]


def test_get_rtm_prices_calls_client_and_normalizes() -> None:
    client = _FakeClient(_raw_gridstatus_frame())
    df = GridstatusERCOTSource(client=client).get_rtm_prices(2025, "HB_HOUSTON")
    assert df.schema == RTM_PRICE_SCHEMA
    assert client.calls[0]["dataset"] == "ercot_spp_real_time_15_min"
    assert client.calls[0]["filter_value"] == "HB_HOUSTON"


def test_get_rtm_prices_rejects_unknown_hub() -> None:
    source = GridstatusERCOTSource(client=_FakeClient(_raw_gridstatus_frame()))
    with pytest.raises(ValueError, match="unknown ERCOT hub"):
        source.get_rtm_prices(2025, "HB_NOWHERE")


def test_validate_rtm_frame_rejects_bad_schema() -> None:
    bad = pl.DataFrame({"interval_start": [1, 2], "price": [3.0, 4.0]})
    with pytest.raises(ValueError, match="RTM_PRICE_SCHEMA"):
        validate_rtm_frame(bad)


def test_store_roundtrip(tmp_path: Path) -> None:
    df = normalize_rtm_frame(_raw_gridstatus_frame())
    assert not is_cached(2025, "HB_HOUSTON", tmp_path)
    path = save_prices(df, 2025, "HB_HOUSTON", tmp_path)
    assert path.exists()
    assert is_cached(2025, "HB_HOUSTON", tmp_path)
    assert load_prices(2025, "HB_HOUSTON", tmp_path).equals(df)


def test_load_prices_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data fetch"):
        load_prices(2025, "HB_WEST", tmp_path)


def test_summarize_prices() -> None:
    summary = summarize_prices(normalize_rtm_frame(_raw_gridstatus_frame()))
    assert summary["intervals"] == 3
    assert summary["min"] == -8.0
    assert summary["max"] == 410.25
    assert summary["negative_intervals"] == 1
