"""Append-only capture of everything the live simulator observes.

The live loop fetches a real-time price and an indicative MCPC snapshot every
few minutes. Each observation is appended here so the live experiment leaves a
complete, replayable record on disk: later we can recompute the LP revenue
ceiling, measure capture rate, or try a new strategy over exactly the data the
live run saw.

Layout (gitignored, under ``data/``)::

    data/live/{hub}/prices/{YYYY-MM-DD}.parquet     RTM settlement prices
    data/live/mcpc_indicative/{YYYY-MM-DD}.parquet  indicative AS clearing prices

Frames are written in the *same* schemas as the historical caches
(``RTM_PRICE_SCHEMA`` / ``MCPC_SCHEMA``), so the existing backtest engine and
co-optimization LP read them with no special-casing. Files are partitioned by
UTC day so each append touches a small file, and rows are de-duplicated on
``interval_start`` keeping the most recently captured value -- which, for the
indicative MCPC, is the freshest projection.

Distinct from ``store.py``: that caches *settled, final* historical data fetched
in bulk; this records *what we observed live*, including forward-looking
projections. Keeping the two trees separate keeps their provenance clean.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

import polars as pl

from dispatcher_watts.data.schemas import (
    MCPC_SCHEMA,
    RTM_PRICE_SCHEMA,
    validate_mcpc_frame,
    validate_rtm_frame,
)
from dispatcher_watts.paths import home_dir

# Anchored to the project home (not the process cwd) so an unattended run
# always reads and writes the same capture log. Sits under data/ (gitignored).
# Overridable so tests can use a temp directory.
DEFAULT_LIVE_DIR: Path = home_dir() / "data" / "live"


def prices_dir(hub: str, data_dir: Path = DEFAULT_LIVE_DIR) -> Path:
    """Directory holding one hub's captured price day-files."""
    return data_dir / hub / "prices"


def mcpc_dir(data_dir: Path = DEFAULT_LIVE_DIR) -> Path:
    """Directory holding captured indicative-MCPC day-files (system-wide)."""
    return data_dir / "mcpc_indicative"


def append_prices(df: pl.DataFrame, hub: str, data_dir: Path = DEFAULT_LIVE_DIR) -> None:
    """Append captured RTM prices, partitioning by UTC day and de-duplicating."""
    validate_rtm_frame(df)
    _append_partitioned(df, prices_dir(hub, data_dir), validate_rtm_frame)


def append_mcpc(df: pl.DataFrame, data_dir: Path = DEFAULT_LIVE_DIR) -> None:
    """Append captured indicative MCPCs, partitioning by UTC day and de-duplicating."""
    validate_mcpc_frame(df)
    _append_partitioned(df, mcpc_dir(data_dir), validate_mcpc_frame)


def load_captured_prices_window(
    start: dt.datetime,
    end: dt.datetime,
    hub: str,
    data_dir: Path = DEFAULT_LIVE_DIR,
) -> pl.DataFrame:
    """Load captured prices with ``interval_start`` in ``[start, end)`` (UTC)."""
    return _load_window(start, end, prices_dir(hub, data_dir), RTM_PRICE_SCHEMA, validate_rtm_frame)


def load_captured_mcpc_window(
    start: dt.datetime,
    end: dt.datetime,
    data_dir: Path = DEFAULT_LIVE_DIR,
) -> pl.DataFrame:
    """Load captured indicative MCPCs with ``interval_start`` in ``[start, end)`` (UTC)."""
    return _load_window(start, end, mcpc_dir(data_dir), MCPC_SCHEMA, validate_mcpc_frame)


def _day_file(directory: Path, day: dt.date) -> Path:
    return directory / f"{day.isoformat()}.parquet"


def _append_partitioned(
    df: pl.DataFrame,
    directory: Path,
    validate: Callable[[pl.DataFrame], pl.DataFrame],
) -> None:
    """Merge `df`'s rows into per-UTC-day files under `directory`.

    For each day touched, concatenates any existing rows with the new ones,
    keeps the last occurrence per ``interval_start`` (new rows win, so a fresher
    capture overwrites a stale one), sorts, and rewrites that day's file.
    """
    if df.is_empty():
        return
    directory.mkdir(parents=True, exist_ok=True)
    by_day = df.with_columns(_day=pl.col("interval_start").dt.date()).partition_by(
        "_day", as_dict=True, include_key=False
    )
    for (day,), new_rows in by_day.items():
        path = _day_file(directory, day)
        if path.exists():
            new_rows = pl.concat([pl.read_parquet(path), new_rows])
        merged = new_rows.unique(subset=["interval_start"], keep="last").sort("interval_start")
        validate(merged).write_parquet(path)


def _load_window(
    start: dt.datetime,
    end: dt.datetime,
    directory: Path,
    schema: pl.Schema,
    validate: Callable[[pl.DataFrame], pl.DataFrame],
) -> pl.DataFrame:
    """Read and concatenate day-files overlapping ``[start, end)``.

    Unlike the historical loaders, missing day-files are skipped silently: live
    capture is naturally sparse (gaps whenever the simulator was not running),
    so an absent day is expected, not an error.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware datetimes")
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end})")
    frames: list[pl.DataFrame] = []
    day = start.astimezone(dt.UTC).date()
    last_day = end.astimezone(dt.UTC).date()
    while day <= last_day:
        path = _day_file(directory, day)
        if path.exists():
            frames.append(pl.read_parquet(path))
        day += dt.timedelta(days=1)
    combined = pl.concat(frames) if frames else pl.DataFrame(schema=schema)
    sliced = combined.filter(
        (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
    ).sort("interval_start")
    return validate(sliced)


__all__ = [
    "DEFAULT_LIVE_DIR",
    "append_mcpc",
    "append_prices",
    "load_captured_mcpc_window",
    "load_captured_prices_window",
    "mcpc_dir",
    "prices_dir",
]
