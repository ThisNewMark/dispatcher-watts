# dispatcher-watts

Backtest battery dispatch strategies against historical ERCOT wholesale
electricity market data.

`dispatcher-watts` simulates how a grid-connected battery would have performed
arbitraging real-time electricity prices in the ERCOT (Texas) market. You pick
a battery spec and a dispatch strategy; it replays a year of historical prices
and reports revenue, capture rate versus a perfect-foresight benchmark, cycle
count, and capacity factor. It is, in effect, a financial backtester whose
asset is a battery and whose market is ERCOT.

## Headline results

A **1 MWh / 0.5 MW battery (2-hour duration, 87% round-trip efficiency)** doing
nothing but real-time energy arbitrage. Revenue is per MWh-year; the percentage
is capture rate against the perfect-foresight ceiling.

### 2024

| Hub | threshold | rolling-average | perfect-foresight |
|---|---|---|---|
| HB_HOUSTON | $11,918 (36%) | $9,907 (29%) | $33,604 |
| HB_NORTH | $14,285 (41%) | $9,861 (28%) | $34,770 |
| HB_SOUTH | $11,819 (34%) | $8,417 (24%) | $34,733 |
| HB_WEST | $15,986 (39%) | $12,402 (31%) | $40,614 |

### 2025

| Hub | threshold | rolling-average | perfect-foresight |
|---|---|---|---|
| HB_HOUSTON | $9,119 (34%) | $8,125 (30%) | $26,808 |
| HB_NORTH | $9,873 (35%) | $8,290 (29%) | $28,605 |
| HB_SOUTH | $9,093 (34%) | $7,412 (28%) | $26,810 |
| HB_WEST | $11,101 (35%) | $9,204 (29%) | $32,062 |

**Three things this shows:**

1. **Arbitrage spreads compressed.** Perfect-foresight revenue — the cleanest
   measure of available arbitrage value — fell **18–23% at every hub** from
   2024 to 2025, consistent with growing battery saturation eroding spreads.
2. **Simple strategies leave most of the value on the table.** The rule-based
   strategies capture only **24–41%** of the perfect-foresight ceiling; real
   operators typically capture 60–80%.
3. **More trading is not better trading.** The rolling-average strategy
   under-performs the cruder threshold strategy at every hub — it over-cycles
   (~600 vs ~180 equivalent cycles per year), bleeding round-trip efficiency
   losses chasing small spreads.

Full analysis, including all charts, is in
[`notebooks/results_v1.ipynb`](notebooks/results_v1.ipynb).

## How it works

- **Data** — ERCOT real-time settlement-point prices (15-minute intervals; the
  price a battery actually settles against), for four trading hubs.
- **Battery model** — state of charge, power and capacity limits, round-trip
  efficiency modelled as a per-leg `√RTE` loss on both charge and discharge,
  and equivalent-cycle tracking.
- **Strategies**
  - `threshold` — charge below a fixed low price, discharge above a fixed high
    price.
  - `rolling-average` — charge below the trailing N-hour average price,
    discharge above it.
  - `perfect-foresight` — a linear program (via `pulp`) solving the
    revenue-maximizing dispatch given complete knowledge of all future prices.
    Not deployable; it is the theoretical ceiling.
- **Capture rate** — strategy revenue ÷ perfect-foresight revenue. The single
  most honest measure of how good a strategy is.

## What this is *not* (v1 honesty notes)

- **Single asset, single market, single revenue stream.** v1 models energy
  arbitrage only. Real operators co-optimize arbitrage with ancillary services
  (frequency regulation, reserves) and capacity markets — often more than half
  of total revenue. v1 ignores those.
- **Backtests overstate real revenue.** Real dispatch has latency, forecast
  error, and outages this does not model. Treat the numbers as an upper bound.
- **No cherry-picking.** Results are reported for all four hubs and both years,
  good and bad.

## Data source

v1 sources ERCOT prices through the third-party
[gridstatus.io](https://www.gridstatus.io) hosted API — specifically the
`ercot_spp_real_time_15_min` dataset. All data access sits behind the
`MarketDataSource` interface (`src/dispatcher_watts/data/base.py`); a direct,
first-party ERCOT API client is the planned replacement (see the stub in
`data/ercot_direct.py`).

Price data is **not committed** to the repo (it is gitignored). Fetch it with
the CLI — see Quickstart.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Add your gridstatus.io API key (free tier available):
# https://www.gridstatus.io/settings/api
cp .env.example .env
# ...then edit .env and set GRIDSTATUS_API_KEY=your_key

# Fetch one year of real-time prices for one hub (cached locally as parquet)
uv run dispatcher-watts data fetch --year 2024 --hub HB_HOUSTON

# Backtest a single strategy
uv run dispatcher-watts backtest --strategy threshold --year 2024 --hub HB_HOUSTON

# Compare all strategies on one hub-year (revenue + capture rate)
uv run dispatcher-watts compare --year 2024 --hub HB_HOUSTON

# Regenerate every v1 result from scratch (fetches any missing data)
uv run dispatcher-watts reproduce-v1
```

ERCOT trading hubs: `HB_HOUSTON`, `HB_NORTH`, `HB_SOUTH`, `HB_WEST`.

## Project layout

```
src/dispatcher_watts/
  data/        ERCOT data access (MarketDataSource interface, gridstatus, cache)
  battery/     battery model and dispatch constraints
  strategies/  threshold, rolling-average, perfect-foresight
  backtest/    backtest engine and metrics
  reporting/   plotly charts
  cli.py       command-line interface
notebooks/     results_v1.ipynb — the published analysis
tests/         pytest suite
```

## Development

```bash
uv run pytest                      # tests
uv run ruff check src tests        # lint
uv run ruff format src tests       # format
uv run mypy                        # type-check
```

## License

MIT — see [LICENSE](LICENSE).
