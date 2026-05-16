"""Local parquet cache for ERCOT price data.

gridstatus.io enforces a monthly row budget, so every fetch is written to disk
and re-read from there. Layout (gitignored):

    data/ercot/{year}/{hub}.parquet
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dispatcher_watts.data.schemas import validate_rtm_frame

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
