"""Pick a market-data source: ERCOT direct first, gridstatus as fallback.

The project's preferred source is `ErcotDirectSource` (first-party, no quota).
gridstatus stays useful for two reasons: (1) anything not yet wired into the
direct client (MCPC, etc.) falls back to it transparently, and (2) if
credentials or network are missing for ERCOT, gridstatus may still work.
"""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

from dispatcher_watts.data.base import MarketDataSource
from dispatcher_watts.data.ercot import GridstatusERCOTSource
from dispatcher_watts.data.ercot_direct import ErcotDirectSource

_log = logging.getLogger(__name__)

# Exceptions for which falling back to the secondary source is sane: missing
# credentials, missing endpoints, network failure. Anything else (e.g. bad hub
# name, schema validation error) is a real bug and should propagate.
_FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    NotImplementedError,
    RuntimeError,
    ConnectionError,
    TimeoutError,
)


class CompositeMarketDataSource(MarketDataSource):
    """Try `primary`; if it raises a fallback-class error, try `secondary`."""

    def __init__(
        self,
        primary: MarketDataSource,
        secondary: MarketDataSource,
    ) -> None:
        self.primary = primary
        self.secondary = secondary

    def get_rtm_prices(self, year: int, hub: str) -> pl.DataFrame:
        return self._try_then_fall_back("get_rtm_prices", year, hub)

    def get_rtm_mcpc(self, year: int) -> pl.DataFrame:
        return self._try_then_fall_back("get_rtm_mcpc", year)

    def _try_then_fall_back(self, method_name: str, *args: Any) -> pl.DataFrame:
        primary_fn = getattr(self.primary, method_name, None)
        if primary_fn is not None:
            try:
                return primary_fn(*args)
            except _FALLBACK_EXCEPTIONS as exc:
                _log.info(
                    "primary source %s failed; falling back to %s. (%s: %s)",
                    type(self.primary).__name__,
                    type(self.secondary).__name__,
                    type(exc).__name__,
                    exc,
                )
        return getattr(self.secondary, method_name)(*args)


def default_source() -> CompositeMarketDataSource:
    """ERCOT direct first, gridstatus second. The CLI uses this by default."""
    return CompositeMarketDataSource(
        primary=ErcotDirectSource(),
        secondary=GridstatusERCOTSource(),
    )
