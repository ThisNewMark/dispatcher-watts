"""ERCOT real-time price data via the gridstatus.io hosted API.

`GridstatusERCOTSource` is the v1 `MarketDataSource`. It fetches ERCOT
real-time settlement-point prices (the `ercot_spp_real_time_15_min` dataset)
for a single trading hub and normalizes them to `RTM_PRICE_SCHEMA`.

The longer-term plan is to replace this with a direct ERCOT API client -- see
`ercot_direct.py`. Keeping all gridstatus-specific code in this one file makes
that swap a one-file change.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import polars as pl

from dispatcher_watts.data.base import MarketDataSource
from dispatcher_watts.data.schemas import (
    ERCOT_HUBS,
    MCPC_PRODUCTS,
    MCPC_RAW_TO_COLUMN,
    MCPC_SCHEMA,
    validate_mcpc_frame,
    validate_rtm_frame,
)

# gridstatus.io dataset holding ERCOT real-time (15-minute) settlement-point
# prices. The price column is `spp` ($/MWh); a hub is selected by filtering the
# `location` column (e.g. "HB_HOUSTON").
RTM_DATASET: str = "ercot_spp_real_time_15_min"

# gridstatus.io dataset holding ERCOT real-time (15-minute) market clearing
# prices for capacity -- i.e. the per-product AS clearing prices co-optimized
# with energy under RTC+B (Dec 5, 2025 onward). System-wide; not hub-specific.
MCPC_DATASET: str = "ercot_mcpc_real_time_15_min"

API_KEY_ENV_VAR: str = "GRIDSTATUS_API_KEY"


class _DatasetClient(Protocol):
    """The slice of `gridstatusio.GridStatusClient` this module relies on.

    Declaring it as a Protocol lets tests inject a fake client -- no network
    call and no API key required.
    """

    def get_dataset(self, dataset: str, **kwargs: Any) -> Any: ...


class GridstatusERCOTSource(MarketDataSource):
    """ERCOT real-time prices sourced from the gridstatus.io hosted API."""

    def __init__(
        self,
        api_key: str | None = None,
        client: _DatasetClient | None = None,
    ) -> None:
        """Create a source.

        Args:
            api_key: gridstatus.io API key. Falls back to the
                ``GRIDSTATUS_API_KEY`` environment variable. Ignored when
                ``client`` is supplied.
            client: a pre-built client, primarily for tests. When omitted, a
                real ``gridstatusio.GridStatusClient`` is constructed lazily on
                first use.
        """
        self._api_key = api_key
        self._client: _DatasetClient | None = client

    def _get_client(self) -> _DatasetClient:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get(API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(
                f"no gridstatus.io API key: pass api_key=... or set ${API_KEY_ENV_VAR}"
            )
        # Imported lazily so tests that inject a fake client need neither the
        # dependency configured nor a network connection.
        from gridstatusio import GridStatusClient

        client: _DatasetClient = GridStatusClient(api_key=api_key)
        self._client = client
        return client

    def get_rtm_prices(self, year: int, hub: str) -> pl.DataFrame:
        if hub not in ERCOT_HUBS:
            raise ValueError(f"unknown ERCOT hub {hub!r}; expected one of {ERCOT_HUBS}")
        raw = self._get_client().get_dataset(
            dataset=RTM_DATASET,
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
            filter_column="location",
            filter_value=hub,
        )
        return normalize_rtm_frame(_ensure_polars(raw))

    def get_rtm_mcpc(self, year: int) -> pl.DataFrame:
        """Real-time AS clearing prices for `year`, wide-format by product.

        Only post-RTC+B (Dec 5, 2025 onward) is meaningful; pre-RTC+B the
        dataset is empty or sparse. The returned frame conforms to
        ``MCPC_SCHEMA``.
        """
        raw = self._get_client().get_dataset(
            dataset=MCPC_DATASET,
            start=f"{year}-01-01",
            end=f"{year + 1}-01-01",
        )
        return normalize_mcpc_frame(_ensure_polars(raw))


def _ensure_polars(frame: Any) -> pl.DataFrame:
    """Coerce a gridstatus result to a polars DataFrame.

    The released gridstatusio (0.15.1) returns a pandas DataFrame; test fakes
    return polars directly. Either way the rest of the module works in polars.
    """
    if isinstance(frame, pl.DataFrame):
        return frame
    return pl.from_pandas(frame)


def normalize_rtm_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Convert a raw gridstatus SPP frame into one matching `RTM_PRICE_SCHEMA`.

    gridstatus returns an `interval_start_utc` column and an `spp` price
    column (among others); this renames them to the canonical names, casts to
    the canonical dtypes, drops everything else, and sorts ascending by
    interval.
    """
    normalized = raw.select(
        interval_start=pl.col("interval_start_utc").cast(
            pl.Datetime(time_unit="us", time_zone="UTC")
        ),
        price=pl.col("spp").cast(pl.Float64),
    ).sort("interval_start")
    return validate_rtm_frame(normalized)


def normalize_mcpc_frame(raw: pl.DataFrame) -> pl.DataFrame:
    """Convert a raw gridstatus MCPC frame into one matching `MCPC_SCHEMA`.

    Input is long-format (one row per (interval, as_type) pair) with columns
    `interval_start_utc`, `as_type`, and `mcpc`. Output is wide-format with one
    row per interval and one ``mcpc_<product>`` column per AS product.
    """
    if raw.is_empty():
        return pl.DataFrame(schema=MCPC_SCHEMA)
    # Map the raw as_type labels to our column suffixes; pivot to wide.
    mapping = pl.DataFrame(
        {
            "as_type": list(MCPC_RAW_TO_COLUMN),
            "product": list(MCPC_RAW_TO_COLUMN.values()),
        }
    )
    long = raw.select(
        pl.col("interval_start_utc").alias("interval_start"),
        pl.col("as_type"),
        pl.col("mcpc").cast(pl.Float64),
    ).join(mapping, on="as_type", how="inner")
    wide = long.pivot(values="mcpc", index="interval_start", on="product").rename(
        {p: f"mcpc_{p}" for p in MCPC_PRODUCTS}
    )
    # Ensure every product column exists and the dtype matches the schema.
    wide = wide.with_columns(
        pl.col("interval_start").cast(pl.Datetime(time_unit="us", time_zone="UTC"))
    )
    for product in MCPC_PRODUCTS:
        col = f"mcpc_{product}"
        if col not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(col))
        wide = wide.with_columns(pl.col(col).cast(pl.Float64))
    wide = wide.select(["interval_start", *(f"mcpc_{p}" for p in MCPC_PRODUCTS)]).sort(
        "interval_start"
    )
    return validate_mcpc_frame(wide)
