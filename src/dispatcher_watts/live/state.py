"""Persistent state for the live paper-trading simulator.

Two artifacts, both under ``state/`` (gitignored):

* ``state/live_state.json`` -- the resumable scalar state: battery state of
  charge and throughput counters, running revenue by source, the AS leg
  currently held, and the last interval processed. Reloading this is what lets
  an external scheduler invoke ``simulate-live`` every few minutes without
  losing the thread.
* ``state/decisions.parquet`` -- an append-only log, one row per interval, of
  what was observed and decided. This is the audit trail behind the P&L chart
  and the source for the capture-rate comparison.

The decision log shares ``interval_start`` semantics with the price/MCPC frames
so it lines up with captured market data when analyzing a run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from dispatcher_watts.battery.model import Battery, BatterySpec
from dispatcher_watts.data.schemas import MCPC_PRODUCTS
from dispatcher_watts.paths import home_dir
from dispatcher_watts.strategies.live import ASCommitment

# Anchored to the project home (not the process cwd) so an unattended scheduler
# always resumes the same run rather than silently seeding a fresh one.
DEFAULT_STATE_DIR: Path = home_dir() / "state"
_STATE_FILENAME: str = "live_state.json"
_DECISIONS_FILENAME: str = "decisions.parquet"

# Revenue accrues into one bucket per source: energy plus the five AS products.
REVENUE_SOURCES: tuple[str, ...] = ("energy", *MCPC_PRODUCTS)


def _zeroed_revenue() -> dict[str, float]:
    return {source: 0.0 for source in REVENUE_SOURCES}


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass(frozen=True)
class DecisionRecord:
    """One interval's observation + decision + realized revenue.

    ``energy_mwh`` is the *executed* grid-side energy, signed (+ discharge /
    - charge) after the battery clamped the strategy's request; ``energy_mw`` is
    the strategy's pre-clamp intent. ``mcpc`` is the full snapshot of AS prices
    acted on, so the log alone is enough to reconstruct the decision.
    """

    interval_start: dt.datetime
    price: float
    mcpc: dict[str, float]
    energy_mw: float
    energy_mwh: float
    as_product: str | None
    as_mw: float
    energy_revenue: float
    as_revenue: float
    soc_mwh_after: float
    reason: str


# Column order/types for the decision-log parquet.
DECISION_LOG_SCHEMA: pl.Schema = pl.Schema(
    {
        "interval_start": pl.Datetime(time_unit="us", time_zone="UTC"),
        "price": pl.Float64(),
        **{f"mcpc_{product}": pl.Float64() for product in MCPC_PRODUCTS},
        "energy_mw": pl.Float64(),
        "energy_mwh": pl.Float64(),
        "as_product": pl.Utf8(),
        "as_mw": pl.Float64(),
        "energy_revenue": pl.Float64(),
        "as_revenue": pl.Float64(),
        "soc_mwh_after": pl.Float64(),
        "reason": pl.Utf8(),
    }
)


@dataclass
class LiveState:
    """Resumable state of one live paper-trading run."""

    hub: str
    spec: BatterySpec
    strategy_name: str
    strategy_config: dict[str, float]
    interval_minutes: int
    soc_mwh: float = 0.0
    energy_charged_mwh: float = 0.0
    energy_discharged_mwh: float = 0.0
    throughput_internal_mwh: float = 0.0
    revenue_by_source: dict[str, float] = field(default_factory=_zeroed_revenue)
    last_commitment: ASCommitment | None = None
    last_processed_interval: dt.datetime | None = None
    created_at: dt.datetime = field(default_factory=_now_utc)

    @property
    def total_revenue(self) -> float:
        return sum(self.revenue_by_source.values())

    @property
    def equivalent_full_cycles(self) -> float:
        return self.throughput_internal_mwh / self.spec.capacity_mwh

    def to_battery(self) -> Battery:
        """Rebuild a `Battery` carrying this state's SoC and throughput counters.

        Lets the engine reuse the battery model's constraint logic instead of
        re-implementing charge/discharge physics for the live loop.
        """
        battery = Battery(self.spec, initial_soc_mwh=self.soc_mwh)
        battery.energy_charged_mwh = self.energy_charged_mwh
        battery.energy_discharged_mwh = self.energy_discharged_mwh
        battery._throughput_internal_mwh = self.throughput_internal_mwh
        return battery

    def adopt_battery(self, battery: Battery) -> None:
        """Copy a stepped battery's counters back into this state."""
        self.soc_mwh = battery.soc_mwh
        self.energy_charged_mwh = battery.energy_charged_mwh
        self.energy_discharged_mwh = battery.energy_discharged_mwh
        self.throughput_internal_mwh = battery._throughput_internal_mwh


# --- JSON (de)serialization --------------------------------------------------


def _commitment_to_json(c: ASCommitment | None) -> dict[str, object] | None:
    if c is None:
        return None
    return {
        "product": c.product,
        "mw": c.mw,
        "committed_at": c.committed_at.isoformat(),
    }


def _commitment_from_json(raw: dict[str, object] | None) -> ASCommitment | None:
    if raw is None:
        return None
    committed_at = dt.datetime.fromisoformat(str(raw["committed_at"]))
    product = raw["product"]
    return ASCommitment(
        product=str(product) if product is not None else None,
        mw=float(raw["mw"]),  # type: ignore[arg-type]
        committed_at=committed_at,
    )


def state_path(data_dir: Path = DEFAULT_STATE_DIR) -> Path:
    return data_dir / _STATE_FILENAME


def decisions_path(data_dir: Path = DEFAULT_STATE_DIR) -> Path:
    return data_dir / _DECISIONS_FILENAME


def save_state(state: LiveState, data_dir: Path = DEFAULT_STATE_DIR) -> Path:
    """Write `state` to ``state/live_state.json``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "hub": state.hub,
        "spec": dataclasses.asdict(state.spec),
        "strategy_name": state.strategy_name,
        "strategy_config": state.strategy_config,
        "interval_minutes": state.interval_minutes,
        "soc_mwh": state.soc_mwh,
        "energy_charged_mwh": state.energy_charged_mwh,
        "energy_discharged_mwh": state.energy_discharged_mwh,
        "throughput_internal_mwh": state.throughput_internal_mwh,
        "revenue_by_source": state.revenue_by_source,
        "last_commitment": _commitment_to_json(state.last_commitment),
        "last_processed_interval": (
            state.last_processed_interval.isoformat()
            if state.last_processed_interval is not None
            else None
        ),
        "created_at": state.created_at.isoformat(),
    }
    path = state_path(data_dir)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_state(data_dir: Path = DEFAULT_STATE_DIR) -> LiveState:
    """Load state from ``state/live_state.json`` (raises if absent)."""
    path = state_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no live state at {path}; run `dispatcher-watts simulate-live` first"
        )
    raw = json.loads(path.read_text())
    last_processed = raw["last_processed_interval"]
    return LiveState(
        hub=raw["hub"],
        spec=BatterySpec(**raw["spec"]),
        strategy_name=raw["strategy_name"],
        strategy_config=raw["strategy_config"],
        interval_minutes=raw["interval_minutes"],
        soc_mwh=raw["soc_mwh"],
        energy_charged_mwh=raw["energy_charged_mwh"],
        energy_discharged_mwh=raw["energy_discharged_mwh"],
        throughput_internal_mwh=raw["throughput_internal_mwh"],
        revenue_by_source={**_zeroed_revenue(), **raw["revenue_by_source"]},
        last_commitment=_commitment_from_json(raw["last_commitment"]),
        last_processed_interval=(
            dt.datetime.fromisoformat(last_processed) if last_processed is not None else None
        ),
        created_at=dt.datetime.fromisoformat(raw["created_at"]),
    )


def state_exists(data_dir: Path = DEFAULT_STATE_DIR) -> bool:
    return state_path(data_dir).exists()


# --- decision log ------------------------------------------------------------


def records_to_frame(records: list[DecisionRecord]) -> pl.DataFrame:
    """Convert decision records to a frame matching ``DECISION_LOG_SCHEMA``."""
    rows = [
        {
            "interval_start": r.interval_start,
            "price": r.price,
            **{f"mcpc_{p}": r.mcpc.get(p, 0.0) for p in MCPC_PRODUCTS},
            "energy_mw": r.energy_mw,
            "energy_mwh": r.energy_mwh,
            "as_product": r.as_product,
            "as_mw": r.as_mw,
            "energy_revenue": r.energy_revenue,
            "as_revenue": r.as_revenue,
            "soc_mwh_after": r.soc_mwh_after,
            "reason": r.reason,
        }
        for r in records
    ]
    return pl.DataFrame(rows, schema=DECISION_LOG_SCHEMA)


def append_decisions(records: list[DecisionRecord], data_dir: Path = DEFAULT_STATE_DIR) -> None:
    """Append decision records to ``state/decisions.parquet`` (dedup on interval)."""
    if not records:
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = records_to_frame(records)
    path = decisions_path(data_dir)
    if path.exists():
        frame = pl.concat([pl.read_parquet(path), frame])
    frame.unique(subset=["interval_start"], keep="last").sort("interval_start").write_parquet(path)


def load_decisions(data_dir: Path = DEFAULT_STATE_DIR) -> pl.DataFrame:
    """Load the decision log, or an empty frame if none has been written."""
    path = decisions_path(data_dir)
    if not path.exists():
        return pl.DataFrame(schema=DECISION_LOG_SCHEMA)
    return pl.read_parquet(path)


__all__ = [
    "DECISION_LOG_SCHEMA",
    "REVENUE_SOURCES",
    "DecisionRecord",
    "LiveState",
    "append_decisions",
    "decisions_path",
    "load_decisions",
    "load_state",
    "records_to_frame",
    "save_state",
    "state_exists",
    "state_path",
]
