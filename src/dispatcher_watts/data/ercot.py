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
from typing import Protocol

import polars as pl

from dispatcher_watts.data.base import MarketDataSource
from dispatcher_watts.data.schemas import ERCOT_HUBS, validate_rtm_frame

# gridstatus.io dataset holding ERCOT real-time (15-minute) settlement-point
# prices. The price column is `spp` ($/MWh); a hub is selected by filtering the
# `location` column (e.g. "HB_HOUSTON").
RTM_DATASET: str = "ercot_spp_real_time_15_min"

API_KEY_ENV_VAR: str = "GRIDSTATUS_API_KEY"


class _DatasetClient(Protocol):
    """The slice of `gridstatusio.GridStatusClient` this module relies on.

    Declaring it as a Protocol lets tests inject a fake client -- no network
    call and no API key required.
    """

    def get_dataset(
        self,
        dataset: str,
        start: str,
        end: str,
        filter_column: str,
        filter_value: str,
        columns: list[str],
        return_format: str,
    ) -> pl.DataFrame: ...


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
            columns=["interval_start_utc", "spp"],
            return_format="polars",
        )
        return normalize_rtm_frame(raw)


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
