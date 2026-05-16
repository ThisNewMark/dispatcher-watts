"""Direct ERCOT API client -- migration target, NOT implemented in v1.

v1 sources data through `GridstatusERCOTSource` (`ercot.py`). The plan is to
replace gridstatus.io with a first-party ERCOT API client: no third-party
dependency, no shared rate limits, and a stronger first-principles credibility
story when the project is published.

This file is a deliberate placeholder so the migration target is explicit from
day one. When implemented, `ErcotDirectSource` must satisfy `MarketDataSource`
and return frames matching `RTM_PRICE_SCHEMA`, so swapping it in is a one-line
change at the call site.

ERCOT public data products: https://www.ercot.com/mp/data-products
"""

from __future__ import annotations

import polars as pl

from dispatcher_watts.data.base import MarketDataSource


class ErcotDirectSource(MarketDataSource):
    """Placeholder for a direct ERCOT API data source. Not implemented in v1."""

    def get_rtm_prices(self, year: int, hub: str) -> pl.DataFrame:
        # TODO(v2): implement against the ERCOT public reports API.
        #   - RTM Settlement Point Prices report (15-minute intervals)
        #   - parse CSV/XML, normalize to RTM_PRICE_SCHEMA
        #   - respect ERCOT rate limits; cache aggressively
        raise NotImplementedError(
            "direct ERCOT client is not implemented in v1; use GridstatusERCOTSource"
        )
