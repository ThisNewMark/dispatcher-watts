"""Polars schemas for ERCOT price data.

One schema definition, used by every data source, means gridstatus.io today
and a direct ERCOT client tomorrow produce an identical frame shape -- the
backtest engine never has to know which provider the data came from.
"""

from __future__ import annotations

import polars as pl

# ERCOT trading-hub settlement points used in v1.
ERCOT_HUBS: tuple[str, ...] = ("HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST")

# ERCOT real-time settlement-point prices are published on 15-minute intervals.
# (The 5-minute series is SCED LMPs -- prices before real-time adders, not the
# price a battery actually settles against.)
RTM_INTERVAL_MINUTES: int = 15

# Canonical schema for a real-time price series:
#   interval_start : start of the settlement interval, timezone-aware (UTC)
#   price          : settlement-point price, $/MWh
RTM_PRICE_SCHEMA: pl.Schema = pl.Schema(
    {
        "interval_start": pl.Datetime(time_unit="us", time_zone="UTC"),
        "price": pl.Float64(),
    }
)


def validate_rtm_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Check that `df` matches `RTM_PRICE_SCHEMA`, then return it unchanged.

    Raising at the data boundary means a malformed frame is caught here rather
    than deep inside the backtest.
    """
    if df.schema != RTM_PRICE_SCHEMA:
        raise ValueError(
            "price frame does not match RTM_PRICE_SCHEMA\n"
            f"  expected: {dict(RTM_PRICE_SCHEMA)}\n"
            f"  actual:   {dict(df.schema)}"
        )
    return df
