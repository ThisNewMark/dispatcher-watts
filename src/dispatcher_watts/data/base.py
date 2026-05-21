"""Abstract market-data interface.

Every historical price access in the project goes through `MarketDataSource`,
so the backtest never depends on a concrete provider. v1 ships
`GridstatusERCOTSource` (`ercot.py`); the migration target is a direct ERCOT
API client (`ercot_direct.py`). Swapping implementations is a one-line change
at the call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl


class MarketDataSource(ABC):
    """A source of historical wholesale electricity prices."""

    @abstractmethod
    def get_rtm_prices(self, year: int, hub: str) -> pl.DataFrame:
        """Return real-time settlement-point prices for one hub and year.

        The returned frame conforms to `RTM_PRICE_SCHEMA` (see `schemas.py`):
        a UTC `interval_start` column and a `price` column in $/MWh, sorted
        ascending by `interval_start`.
        """
        raise NotImplementedError

    def get_rtm_mcpc(self, year: int) -> pl.DataFrame:
        """Return post-RTC+B real-time AS clearing prices for one year.

        The returned frame conforms to ``MCPC_SCHEMA`` (see `schemas.py`):
        one row per 15-min interval, one column per AS product, system-wide.

        Default implementation raises ``NotImplementedError`` so that sources
        without MCPC support trigger fallback in the composite source.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement get_rtm_mcpc")
