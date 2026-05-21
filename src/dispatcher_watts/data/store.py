"""Local parquet cache for ERCOT price data.

gridstatus.io enforces a monthly row budget, so every fetch is written to disk
and re-read from there. Layout (gitignored):

    data/ercot/{year}/{hub}.parquet
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from dispatcher_watts.data.schemas import (
    MCPC_SCHEMA,
    RTM_PRICE_SCHEMA,
    validate_mcpc_frame,
    validate_rtm_frame,
)

# Repo-root-relative default. Overridable so tests can use a temp directory.
DEFAULT_DATA_DIR: Path = Path("data") / "ercot"


def cache_path(year: int, hub: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Return the parquet path for one hub-year."""
    return data_dir / str(year) / f"{hub}.parquet"


def is_cached(year: int, hub: str, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """Return whether a parquet file already exists for this hub-year."""
    return cache_path(year, hub, data_dir).exists()


def save_prices(df: pl.DataFrame, year: int, hub: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Validate `df` against the schema and write it to the parquet cache."""
    validate_rtm_frame(df)
    path = cache_path(year, hub, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def load_prices(year: int, hub: str, data_dir: Path = DEFAULT_DATA_DIR) -> pl.DataFrame:
    """Load and schema-validate cached prices for one hub-year."""
    path = cache_path(year, hub, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached data at {path}; run `dispatcher-watts data fetch` first"
        )
    return validate_rtm_frame(pl.read_parquet(path))


def mcpc_cache_path(year: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Return the parquet path for one year of RT AS clearing prices."""
    return data_dir / str(year) / "mcpc_rt_15min.parquet"


def is_mcpc_cached(year: int, data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    return mcpc_cache_path(year, data_dir).exists()


def save_mcpc(df: pl.DataFrame, year: int, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Validate `df` against MCPC_SCHEMA and write it to the parquet cache."""
    validate_mcpc_frame(df)
    path = mcpc_cache_path(year, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def load_mcpc(year: int, data_dir: Path = DEFAULT_DATA_DIR) -> pl.DataFrame:
    """Load and schema-validate cached AS clearing prices for one year."""
    path = mcpc_cache_path(year, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached MCPC at {path}; run `dispatcher-watts data fetch-as` first"
        )
    return validate_mcpc_frame(pl.read_parquet(path))


def load_mcpc_window(
    start_date: dt.date,
    end_date: dt.date,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pl.DataFrame:
    """Load cached AS clearing prices for `[start_date, end_date)` UTC days.

    Same window semantics as ``load_prices_window``. AS prices are system-wide,
    so there is no hub parameter.
    """
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
    start_dt = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.UTC)
    end_dt = dt.datetime.combine(end_date, dt.time.min, tzinfo=dt.UTC)
    frames: list[pl.DataFrame] = []
    for year in range(start_date.year, end_date.year + 1):
        year_start = dt.date(year, 1, 1)
        year_end = dt.date(year + 1, 1, 1)
        if not (start_date < year_end and end_date > year_start):
            continue
        path = mcpc_cache_path(year, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"no cached MCPC at {path}; "
                f"run `dispatcher-watts data fetch-as --year {year}` first"
            )
        frames.append(pl.read_parquet(path))
    combined = pl.concat(frames) if frames else pl.DataFrame(schema=MCPC_SCHEMA)
    sliced = combined.filter(
        (pl.col("interval_start") >= start_dt) & (pl.col("interval_start") < end_dt)
    ).sort("interval_start")
    return validate_mcpc_frame(sliced)


def load_prices_window(
    start_date: dt.date,
    end_date: dt.date,
    hub: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> pl.DataFrame:
    """Load cached prices for one hub across `[start_date, end_date)` UTC days.

    Concatenates whatever year-files the window touches and returns the
    schema-validated slice. The end date is exclusive (interpreted as 00:00 UTC
    of that day). Every year that overlaps the window must already be cached.
    """
    if start_date >= end_date:
        raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
    start_dt = dt.datetime.combine(start_date, dt.time.min, tzinfo=dt.UTC)
    end_dt = dt.datetime.combine(end_date, dt.time.min, tzinfo=dt.UTC)
    frames: list[pl.DataFrame] = []
    for year in range(start_date.year, end_date.year + 1):
        # Skip years that don't actually overlap [start_date, end_date).
        year_start = dt.date(year, 1, 1)
        year_end = dt.date(year + 1, 1, 1)
        if not (start_date < year_end and end_date > year_start):
            continue
        path = cache_path(year, hub, data_dir)
        if not path.exists():
            raise FileNotFoundError(
                f"no cached data at {path}; "
                f"run `dispatcher-watts data fetch --year {year} --hub {hub}` first"
            )
        frames.append(pl.read_parquet(path))
    combined = pl.concat(frames) if frames else pl.DataFrame(schema=RTM_PRICE_SCHEMA)
    sliced = combined.filter(
        (pl.col("interval_start") >= start_dt) & (pl.col("interval_start") < end_dt)
    ).sort("interval_start")
    return validate_rtm_frame(sliced)


def summarize_prices(df: pl.DataFrame) -> dict[str, float]:
    """Compute a basic statistical summary of a price series ($/MWh)."""
    validate_rtm_frame(df)
    if df.is_empty():
        raise ValueError("cannot summarize an empty price frame")
    # Aggregate inside polars, then pull a single typed row -- avoids juggling
    # the wide union type of bare `Series.min()` etc.
    stats = df.select(
        pl.col("price").min().alias("min"),
        pl.col("price").mean().alias("mean"),
        pl.col("price").median().alias("median"),
        pl.col("price").max().alias("max"),
        pl.col("price").std().alias("std"),
        (pl.col("price") < 0).sum().alias("negative_intervals"),
    ).row(0, named=True)
    return {
        "intervals": float(df.height),
        "min": float(stats["min"]),
        "mean": float(stats["mean"]),
        "median": float(stats["median"]),
        "max": float(stats["max"]),
        "std": float(stats["std"]),
        "negative_intervals": float(stats["negative_intervals"]),
    }
