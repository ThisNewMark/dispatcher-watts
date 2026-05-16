# dispatcher-watts

Backtest battery dispatch strategies against historical ERCOT wholesale
electricity market data.

> **Status: v1 in progress.** The data layer and CLI are in place. Battery
> model, dispatch strategies, and the results report are being built out over
> the v1 milestones (see `CLAUDE.md`). Headline results will land here when v1
> ships.

## What this is

`dispatcher-watts` simulates how a grid-connected battery would have performed
arbitraging real-time electricity prices in the ERCOT (Texas) market. You pick
a battery spec and a dispatch strategy; it replays a year of historical prices
and reports revenue, capture rate versus a perfect-foresight benchmark, cycle
count, and capacity factor.

It is, in effect, a financial backtester where the asset is a battery and the
market is ERCOT.

## What this is *not* (v1 honesty notes)

- **Single asset, single market, single revenue stream.** v1 models energy
  arbitrage only. Real operators co-optimize arbitrage with ancillary services
  (frequency regulation, reserves) and capacity markets -- often more than half
  of total revenue. v1 ignores those.
- **Backtests overstate real revenue.** Real dispatch has latency, forecast
  error, and operational constraints this does not model. Treat the numbers as
  an upper bound, not a forecast.
- **No cherry-picking.** Results are reported across all hubs and years, good
  and bad.

## Data source

v1 sources ERCOT real-time prices through the third-party
[gridstatus.io](https://www.gridstatus.io) hosted API (`gridstatusio` Python
client). Specifically the `ercot_spp_real_time_15_min` dataset: **15-minute**
settlement-point prices -- the price a battery actually settles against.

All data access sits behind the `MarketDataSource` interface
(`src/dispatcher_watts/data/base.py`). A direct, first-party ERCOT API client
is the planned replacement -- see the stub in `data/ercot_direct.py`.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Add your gridstatus.io API key: copy the template, then paste your key in.
# Get a key (free tier available) at https://www.gridstatus.io/settings/api
cp .env.example .env
# ...then edit .env and set GRIDSTATUS_API_KEY=your_key

# Fetch one year of real-time prices for one hub (cached locally as parquet)
uv run dispatcher-watts data fetch --year 2025 --hub HB_HOUSTON

# Print a statistical summary of cached prices
uv run dispatcher-watts data summary --year 2025 --hub HB_HOUSTON
```

ERCOT trading hubs: `HB_HOUSTON`, `HB_NORTH`, `HB_SOUTH`, `HB_WEST`.

## Development

```bash
uv run pytest                      # tests
uv run ruff check src tests        # lint
uv run ruff format src tests       # format
uv run mypy                        # type-check
```

## License

MIT -- see [LICENSE](LICENSE).
