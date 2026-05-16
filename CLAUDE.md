# CLAUDE.md — Battery Dispatch Optimizer

## Project overview

A Python library and CLI tool that backtests battery dispatch strategies against
historical wholesale electricity market data. Open-source, GitHub-published, designed
to be the credible artifact that demonstrates competence in battery economics and
wholesale market mechanics.

**Long-term vision (v2+):** evolve into an open-source AI data center microgrid
dispatcher — battery dispatch + flexible compute load scheduling + grid harmonics
smoothing. v1 is the foundation. Do not over-engineer for v2 yet.

**Author context:** strong full-stack and Python skills, fluent with LLM-assisted
development via Claude Code. New to wholesale electricity markets — assume the
domain knowledge is being built alongside the code.

## Project name

`dispatcher-watts`. CLI command: `dispatcher-watts`. Python package:
`dispatcher_watts` (underscore version since hyphens are illegal in Python
module names).

## Goals for v1

By the end of v1 (~4 weeks of focused work), the project should produce:

1. A reproducible backtest of battery dispatch against 2+ years of historical
   ERCOT real-time market data.
2. Three implemented dispatch strategies (threshold, rolling-average,
   perfect-foresight benchmark).
3. A results report: revenue per MWh-year, capture rate vs. perfect foresight,
   battery cycle count, capacity factor.
4. Charts: dispatch decisions over time, state of charge profile, daily revenue.
5. A polished README and one published blog post / thread explaining the results.
6. CI tests passing on GitHub Actions.

**v1 explicitly excludes:** live trading, paper-trading on real-time data, ML-based
strategies, multi-market (CAISO/NYISO/PJM), AI workload simulation, microgrid
dispatch, web UI. Those are all v2.

## Tech stack

- **Language:** Python 3.11+
- **Package manager:** `uv` (faster than poetry, cleaner than pip+venv)
- **Data:** `polars` for time-series (faster than pandas for this workload),
  `pyarrow` for parquet I/O
- **CLI:** `typer` (consistent with modern Python CLI conventions)
- **Testing:** `pytest`, `pytest-cov`
- **Linting/formatting:** `ruff` for both
- **Charts:** `plotly` for interactive charts in notebooks, `matplotlib` for static
  charts in reports
- **API client:** `httpx` (async-capable, sane defaults)

No frameworks or heavy dependencies in v1. Resist the urge to add anything not
explicitly justified.

## Project structure

```
dispatcher-watts/
├── README.md
├── CLAUDE.md
├── pyproject.toml
├── .github/workflows/ci.yml
├── data/                       # gitignored, local parquet files
│   └── ercot/
├── src/dispatcher_watts/
│   ├── __init__.py
│   ├── cli.py                  # typer entry point
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ercot.py            # ERCOT data fetching (via gridstatus in v1)
│   │   └── schemas.py          # polars schemas for price data
│   ├── battery/
│   │   ├── __init__.py
│   │   ├── model.py            # Battery class: SoC, capacity, RTE, degradation
│   │   └── constraints.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py             # Strategy ABC
│   │   ├── threshold.py        # charge < X, discharge > Y
│   │   ├── rolling_avg.py      # charge below N-hour avg, discharge above
│   │   └── perfect_foresight.py # LP solver, theoretical max
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py           # main backtest loop
│   │   └── metrics.py          # revenue, capture rate, cycles
│   └── reporting/
│       ├── __init__.py
│       └── charts.py
├── tests/
│   ├── test_battery.py
│   ├── test_strategies.py
│   └── test_backtest.py
├── notebooks/
│   └── results_v1.ipynb        # the published analysis
└── results/
    └── v1/                     # output CSVs, charts
```

## Core domain concepts (so Claude Code stays accurate)

### Wholesale electricity markets

US wholesale markets are run by Independent System Operators (ISOs). v1 uses
ERCOT (Texas) because the API is the most accessible and prices are the most
volatile. Each ISO runs two coupled markets:

- **Day-Ahead Market (DAM):** hourly prices set the day before delivery, based on
  bids and forecasted demand. Settled in $/MWh.
- **Real-Time Market (RTM):** prices that reflect actual conditions during
  delivery, much more volatile than DAM. ERCOT runs SCED (the dispatch engine)
  every 5 minutes, but **real-time settlement point prices — the prices a
  battery actually settles against — are 15-minute** time-weighted values. The
  5-minute series is locational marginal prices (LMPs) before real-time
  adders, not the settlement price. v1 uses the 15-minute RTM SPP series.

A battery operator can participate in both. v1 only considers RTM dispatch
against RTM prices — this is the simplest and most volatile case.

### Battery model

A battery is described by:
- **Capacity (kWh or MWh):** total energy that can be stored
- **Power rating (kW or MW):** max instantaneous charge/discharge rate
- **State of charge (SoC):** current energy stored, 0 to capacity
- **Round-trip efficiency (RTE):** typically 0.85–0.92 for modern lithium. Energy
  lost on each charge-discharge cycle. Model as half-loss on charge, half on
  discharge (i.e. √RTE each way), or as one-sided loss on discharge.
- **Cycle degradation:** each charge-discharge cycle marginally reduces capacity.
  v1 can use a simple linear model (e.g. 0.005% capacity loss per equivalent
  full cycle) and track for reporting; not necessary to apply to active capacity.

v1 default spec: 1 MWh capacity, 500 kW power (so 2-hour duration), 0.87 RTE.

### Dispatch and revenue

At each timestep (15-min interval), the strategy decides: charge, discharge, or
idle. The battery model enforces constraints (cannot discharge more than current
SoC, cannot charge above capacity, cannot exceed power rating).

Revenue per interval = (discharge_MWh × price_$/MWh) − (charge_MWh × price_$/MWh)

Cumulative revenue over a year is the primary metric. For ERCOT 1 MWh batteries,
real operators have reported $50k–$150k/MWh-year of revenue in recent years.

### Capture rate (key metric)

The ratio of your strategy's revenue to the **perfect foresight** revenue — i.e.
the max revenue achievable if you knew all future prices and could solve the
optimal dispatch as a linear program. Real operators capture 60–80% of perfect
foresight. v1 should report this.

Perfect foresight is solved with `scipy.optimize.linprog` or `pulp`. It's not a
strategy you'd ever deploy — it's a benchmark for evaluating real strategies.

## Data sources

### ERCOT (primary, v1)

ERCOT publishes settlement prices via their public API. Key endpoints:

- **RTM Settlement Point Prices:** 15-minute prices at hundreds of nodes. Use
  hub prices (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST) for v1. (Sourced in v1
  via the gridstatus.io `ercot_spp_real_time_15_min` dataset.)
- **DAM Settlement Point Prices:** hourly day-ahead prices.

Their public reports portal: https://www.ercot.com/mp/data-products

For programmatic access:
- ERCOT's public API requires no auth for most reports
- Reports come as CSV or XML; parse and cache as parquet locally
- Be respectful: don't scrape aggressively. Cache everything.

**v1 data source decision: use `gridstatusio` first, migrate to direct ERCOT later.**

For v1, use the `gridstatusio` Python library (third-party, free tier covers
the v1 historical workload). It wraps ERCOT and several other ISOs with a
clean consistent API, and saves ~1 week of data-plumbing work. The README
should clearly disclose the data source.

The eventual goal is to replace gridstatusio with a direct ERCOT API client
— bulletproof, no third-party dependency, no rate limits, and a stronger
"first-principles" credibility story when the project goes public.

Plan for this transition by:
- Putting all data access behind an abstract `MarketDataSource` interface in
  `data/`, so swapping implementations is a one-line change
- Treating the `gridstatusio` implementation as a `GridstatusERCOTSource`
- Leaving a stub file `data/ercot_direct.py` with a TODO comment, so the
  migration target is explicit from day one

### Sample historical range for v1

Pull 2024 and 2025 ERCOT RTM data. Run the backtest on each year separately
and report results side-by-side — this shows the strategy's behavior under
different market conditions, including the compression of arbitrage spreads
as battery saturation has grown. Showing both years honestly (rather than
cherry-picking) is a credibility multiplier.

## v1 milestones (4 weeks)

### Week 1: Scaffolding + data

- [ ] Initialize repo, pyproject.toml, ruff, pytest, GitHub Actions
- [ ] Implement `data/ercot.py` — fetch and cache historical RTM prices
- [ ] Local data store: parquet files in `data/ercot/{year}/{hub}.parquet`
- [ ] CLI command: `dispatcher-watts data fetch --year 2024 --hub HB_HOUSTON`
- [ ] Write tests for data loading
- [ ] First commit, push, CI green

**Definition of done:** can fetch 1 year of 15-min prices for one hub, load
into a polars DataFrame, run a basic statistical summary.

### Week 2: Battery model + threshold strategy

- [ ] Implement `Battery` class with charge/discharge methods, SoC tracking,
      constraint enforcement
- [ ] Implement `Strategy` base class
- [ ] Implement `ThresholdStrategy`: charge below price P_low, discharge above P_high
- [ ] Implement `Backtest` engine: iterate timesteps, call strategy, update
      battery, compute revenue
- [ ] Implement basic metrics: total revenue, total cycles, time spent at each
      SoC level
- [ ] CLI command: `dispatcher-watts backtest --strategy threshold --year 2024`
- [ ] Tests for battery model edge cases (over-charge, over-discharge, exact SoC limits)

**Definition of done:** end-to-end backtest produces a single revenue number for
one strategy on one year of one hub's data.

### Week 3: Better strategies + perfect foresight + analysis

- [ ] Implement `RollingAverageStrategy`: charge below N-hour rolling avg,
      discharge above
- [ ] Implement `PerfectForesightStrategy` using linear programming (pulp or
      scipy.optimize.linprog)
- [ ] Compute capture rate: strategy revenue / perfect foresight revenue
- [ ] Charts: SoC over time, dispatch decisions, cumulative revenue, daily
      revenue distribution
- [ ] Multi-year backtest: run 2024 and 2025 separately, compare results
      side-by-side. Report should make the year-over-year compression of
      revenue (if any) visible — that's part of the story.
- [ ] Multi-hub: run across all four ERCOT hubs

**Definition of done:** a notebook (`notebooks/results_v1.ipynb`) that runs all
strategies across all hubs for 2 years, produces a results table, and renders
the key charts.

### Week 4: Polish + publish

- [ ] README with clear "what this is" and headline results table
- [ ] Reproducibility: `make reproduce` or `dispatcher-watts reproduce-v1` regenerates
      all results from scratch
- [ ] License (MIT)
- [ ] Blog post draft: "What an open-source battery dispatch optimizer earns on
      ERCOT" — include the methodology, the strategies, the results table, the
      limitations
- [ ] Publish repo publicly on GitHub
- [ ] Publish blog post (personal site, LinkedIn, X thread)

**Definition of done:** the repo is public, the blog post is up, and any
energy/storage person could clone the repo and reproduce the results with one
command.

## Coding conventions

- Use polars over pandas. The performance gain matters when iterating on
  multi-year backtests.
- Type hints on all public functions. Run `mypy` in CI.
- Prefer explicit over clever. Energy domain code that someone else will read.
- Tests required for: battery model state transitions, perfect foresight
  correctness (against a known small example), data loading.
- Tests not required for: chart-rendering code, the CLI wrapper, the notebook.
- Commit messages: present tense, imperative ("Add threshold strategy" not
  "Added threshold strategy").
- Branch model: just use `main`. Solo project, no PRs needed unless that pattern
  helps with discipline.

## Common commands

```bash
# Install dependencies
uv sync

# Fetch ERCOT data for a year
uv run dispatcher-watts data fetch --year 2024 --hub HB_HOUSTON

# Run a backtest
uv run dispatcher-watts backtest --strategy threshold --year 2024 --hub HB_HOUSTON

# Run the perfect-foresight benchmark
uv run dispatcher-watts backtest --strategy perfect-foresight --year 2024 --hub HB_HOUSTON

# Reproduce all v1 results
uv run dispatcher-watts reproduce-v1

# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## References (read before / during build)

- **gridstatus.io documentation** — for the data fetching layer
- **Modo Energy GB Battery Bible** — free, very accessible explainer on how
  battery revenue stacks work. UK-focused but concepts transfer.
- **ERCOT Market Education** materials — free, dry but accurate
- **"Battery Storage Asset Optimization" by Habitat Energy** — public blog posts
  on dispatch strategy
- **NREL ReV / SAM Battery Storage Technology** — public reports
- **Sungrow / Wartsila / Tesla Megapack spec sheets** — for realistic battery
  parameters

For the conceptual model:
- Backtest framework architecture should feel similar to a financial trading
  backtester (Backtrader, vectorbt, zipline). The "asset" is a battery; the
  "market" is ERCOT.

## Honesty notes for v1

- This is a single-asset, single-market, single-revenue-stream backtester. Real
  battery operators co-optimize across energy arbitrage, ancillary services
  (frequency regulation, reserves), and capacity markets. v1 ignores ancillary
  services. This is okay — energy arbitrage alone is the largest revenue stream
  for most batteries and the easiest to model.
- Backtest results will overstate real-world revenue. Real dispatch has
  latency, forecast error, and operational constraints v1 doesn't model. Be
  upfront about this in the README and blog post. Honesty is a credibility
  multiplier in this industry.
- Do NOT cherry-pick the best year, the best hub, or the best strategy
  parameters. Show full results, including poor performance.

---

## v2 notes (NOT v1 scope)

Captured here so Claude Code can keep the long arc in mind, but resist
implementing any of this during v1.

### v2 modules

1. **CAISO + NYISO + PJM market support.** Generalize the data layer with an
   abstract market interface.

2. **Live paper-trading.** Connect to gridstatus.io's near-real-time feed. Run
   the strategy on incoming prices, log decisions, publish a daily P&L. This is
   the "receipts" layer — anyone in the industry can watch the strategy work
   in real time.

3. **Day-ahead + real-time co-optimization.** Real operators bid in DAM and
   adjust in RTM. Model this two-stage decision properly.

4. **Ancillary services revenue.** Frequency regulation, spinning reserves,
   non-spinning reserves. These are large revenue streams (sometimes >50%) for
   real operators.

5. **AI workload simulator.** Synthetic generator producing realistic AI training
   and inference load profiles based on PUBLISHED data — not personal workload
   measurements. Sources to use:
   - EPRI "Powering Intelligence" report load shapes
   - NVIDIA published GPU power envelopes (H100, H200, B100)
   - Published training disclosures from major model providers
   - MLPerf benchmark traces
   - Academic papers on training cluster power dynamics
   
   The output is a time-series of MW load profiles that look like a real data
   center under different workload mixes. Inputs to the dispatcher.

6. **Flexible compute scheduler.** Given a queue of compute jobs with deadlines
   and energy requirements, schedule them across time to minimize cost subject
   to grid and battery constraints. Mixed integer programming or RL.

7. **Grid harmonics smoothing.** Battery dispatches at sub-second resolution to
   keep grid import smooth even when compute load is spiky. This is the
   "harmonics" thing Doug emphasized — what makes large data center
   interconnections possible.

8. **Web dashboard.** Public live view of the paper-trading system. React + Vite
   + a simple chart library. Hosted on Vercel or Railway.

9. **Tycoon-style game UI (parked, v3 candidate).** A RollerCoaster Tycoon /
   SimCity-inspired isometric visualization of the microgrid: visible battery
   cells filling and draining, data center racks lighting up as compute runs,
   grid connection showing smooth-vs-spiky power draw, onsite solar/gas
   indicators reflecting real-time generation. The state is driven by the same
   underlying dispatch simulation as the boring chart-based dashboard — game UI
   is rendering, not simulation.

   Why this matters: a dispatcher rendered as a tycoon game is faithful to the
   system (it IS a real-time multi-component optimization), and game-aesthetic
   ops UIs get used more, shared more, and trusted faster than chart-only ones.
   Strong content-marketing potential — "we rendered open-source battery
   dispatch as an isometric tycoon game" is a follow-up blog post that probably
   outperforms the original launch post.

   Sub-modes worth considering:
   - **Live mode** — real-time isometric scene driven by paper-trading state
   - **Learn-by-playing scenario** — user manually dispatches a battery on one
     historical day, sees their revenue vs. perfect foresight vs. the algorithm
   - **Scenario explorer / "build a microgrid"** — drag-and-drop components,
     run dispatch simulation, see annual revenue + IRR. Becomes a sales tool
     for any developer evaluating a project.

   Tech stack candidates: Phaser 3, PixiJS, Three.js with isometric camera,
   kaplay, or DOM-based pixel art with CSS. Already on React + Vite +
   TypeScript so any of these slot in cleanly.

   **Do not start this before v1 ships and the basic chart-based dashboard
   exists.** This is the second-wave attention play, not the first one.

### v2 positioning

When v2 ships, the project re-launches as: "open-source AI data center microgrid
dispatcher." Same core code, different framing, much bigger audience. Target
audience for v2 announcement: Emerald AI, Crusoe, Loadcrest, Crux, every
hyperscaler's energy team, every infrastructure investor watching data centers.

### v2 timing

Don't start v2 until v1 is fully shipped (repo public, blog post out, at least
two weeks of "is anyone reaching out?" observation). The information from those
two weeks shapes v2's priorities. Possibly v2 doesn't happen at all if v1 leads
to consulting or hiring conversations that consume time productively.

---

## Anti-drift reminders

The author of this project has a self-acknowledged pattern: builds deep for
about a month, then drifts to new ideas. This project is structured to
accommodate that:

- v1 is sized to be completable in 4 weeks of focused work
- v1 produces a complete, shippable artifact even if v2 never happens
- v2 is scoped but deliberately deferred — do not start it during v1
- The artifact has standalone value (credibility, learning, portfolio) even if
  no business outcome materializes

Resist the urge to:
- Add ML-based strategies in v1
- Build the web UI early
- Build the tycoon-style game UI (it's a v3 candidate — gorgeous to imagine,
  catastrophic to start during v1)
- Generalize to other markets before ERCOT is solid
- Skip the blog post / publication step (this is the highest-leverage hour of
  the whole project)
- Pivot to a different project before the repo is public

If during v1 you find yourself bored or pulled toward something more
interesting, finish v1 anyway. The discipline of shipping is the project.
