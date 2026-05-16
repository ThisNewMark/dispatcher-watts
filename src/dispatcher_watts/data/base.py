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
