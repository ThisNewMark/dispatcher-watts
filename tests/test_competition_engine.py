"""Tests for the multi-participant competition stepper (competition/engine.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from dispatcher_watts.competition import COMPETITION_SPEC
from dispatcher_watts.competition.engine import AdvanceSummary, advance_market
from dispatcher_watts.competition.store import CompetitionStore, QueuedDecision
from dispatcher_watts.data.schemas import MCPC_PRODUCTS, MCPC_SCHEMA, RTM_PRICE_SCHEMA

NOW = dt.datetime(2026, 5, 22, 13, 0, tzinfo=dt.UTC)


class _FakeSource:
    def __init__(self, prices: pl.DataFrame, mcpc: pl.DataFrame) -> None:
        self._prices = prices
        self._mcpc = mcpc

    def get_rtm_prices_window(self, start: dt.datetime, end: dt.datetime, hub: str) -> pl.DataFrame:
        return self._prices.filter(
            (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
        )

    def get_indicative_mcpc_window(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        return self._mcpc.filter(
            (pl.col("interval_start") >= start) & (pl.col("interval_start") < end)
        )


def _intervals(n: int) -> list[dt.datetime]:
    base = NOW - dt.timedelta(minutes=15 * n)
    return [base + dt.timedelta(minutes=15 * i) for i in range(n)]


def _prices(starts: list[dt.datetime], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"interval_start": starts, "price": values}, schema=RTM_PRICE_SCHEMA)


def _mcpc(starts: list[dt.datetime], rrs: list[float]) -> pl.DataFrame:
    cols = {f"mcpc_{p}": [0.0] * len(starts) for p in MCPC_PRODUCTS}
    cols["mcpc_rrs"] = rrs
    return pl.DataFrame({"interval_start": starts, **cols}, schema=MCPC_SCHEMA)


def _source(n: int = 4, price: float = 80.0, rrs: float = 9.0) -> _FakeSource:
    starts = _intervals(n)
    return _FakeSource(_prices(starts, [price] * n), _mcpc(starts, [rrs] * n))


def _store_with(tmp_path: Path, *names: str) -> tuple[CompetitionStore, dict[str, str]]:
    store = CompetitionStore(tmp_path / "comp.db")
    ids = {name: store.register_participant(name).id for name in names}
    return store, ids


def _advance(
    store: CompetitionStore,
    source: _FakeSource,
    tmp_path: Path,
    now: dt.datetime = NOW,
) -> AdvanceSummary:
    # data_dir points at a temp capture dir so tests never touch the real data/live.
    return advance_market(store, source, now=now, lookback_hours=1.0, data_dir=tmp_path / "live")


def _revenue(store: CompetitionStore, pid: str) -> float:
    row = store.connection.execute(
        "SELECT COALESCE(SUM(energy_revenue + as_revenue), 0) AS r "
        "FROM decision_log WHERE participant_id = ?",
        (pid,),
    ).fetchone()
    return float(row["r"])


# --- core stepping -----------------------------------------------------------


def test_advances_all_participants_over_shared_tape(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "a", "b")
    summary = _advance(store, _source(4), tmp_path)
    assert summary.intervals_processed == 4
    assert summary.participants == 2
    assert store.count_decisions(ids["a"]) == 4
    assert store.count_decisions(ids["b"]) == 4


def test_no_queued_decision_uses_default_fill(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "idle")
    _advance(store, _source(4), tmp_path)
    rows = store.connection.execute(
        "SELECT was_default, energy_mwh, as_revenue FROM decision_log"
    ).fetchall()
    assert all(r["was_default"] == 1 for r in rows)
    assert all(r["energy_mwh"] == 0.0 for r in rows)
    assert _revenue(store, ids["idle"]) == 0.0


def _seed_soc(store: CompetitionStore, participant_id: str, soc_mwh: float) -> None:
    p = store.get_participant(participant_id)
    assert p is not None
    p.soc_mwh = soc_mwh
    store.save_participant_state(p)


def test_valid_queued_decision_is_applied(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "trader")
    _seed_soc(store, ids["trader"], 5.0)  # charge to back the AS reservation
    target = _intervals(4)[1]
    store.queue_decision(
        QueuedDecision(
            ids["trader"],
            target,
            energy_mw=0.0,
            as_product="rrs",
            as_mw=1.25,
            submitted_at=target - dt.timedelta(minutes=10),
        )
    )
    _advance(store, _source(4), tmp_path)
    row = store.connection.execute(
        "SELECT was_default, as_product, as_revenue FROM decision_log "
        "WHERE participant_id = ? AND interval_start = ?",
        (ids["trader"], target.isoformat()),
    ).fetchone()
    assert row["was_default"] == 0
    assert row["as_product"] == "rrs"
    assert row["as_revenue"] == 1.25 * 9.0


def test_decision_submitted_after_cutoff_is_rejected(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "cheater")
    target = _intervals(4)[1]
    # Submitted at the interval start (== cutoff) -> NOT strictly before -> rejected.
    store.queue_decision(
        QueuedDecision(ids["cheater"], target, 0.0, "rrs", 1.25, submitted_at=target)
    )
    _advance(store, _source(4), tmp_path)
    row = store.connection.execute(
        "SELECT was_default FROM decision_log WHERE participant_id = ? AND interval_start = ?",
        (ids["cheater"], target.isoformat()),
    ).fetchone()
    assert row["was_default"] == 1  # fell through to default-fill


def test_default_fill_holds_existing_as_commitment(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "holder")
    _seed_soc(store, ids["holder"], 5.0)  # charge to back the held AS reservation
    starts = _intervals(4)
    store.queue_decision(
        QueuedDecision(
            ids["holder"],
            starts[0],
            0.0,
            "rrs",
            1.25,
            submitted_at=starts[0] - dt.timedelta(minutes=10),
        )
    )
    _advance(store, _source(4), tmp_path)
    rows = store.connection.execute(
        "SELECT as_product, as_revenue, was_default FROM decision_log "
        "WHERE participant_id = ? ORDER BY interval_start",
        (ids["holder"],),
    ).fetchall()
    assert [r["as_product"] for r in rows] == ["rrs", "rrs", "rrs", "rrs"]
    assert all(r["as_revenue"] == 1.25 * 9.0 for r in rows)
    assert [r["was_default"] for r in rows] == [0, 1, 1, 1]  # held by default after the first


def test_battery_state_persists_and_resumes(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "trader")
    starts = _intervals(4)
    p = store.get_participant(ids["trader"])
    assert p is not None
    p.soc_mwh = 5.0
    store.save_participant_state(p)
    for s in starts:
        store.queue_decision(
            QueuedDecision(
                ids["trader"],
                s,
                COMPETITION_SPEC.power_mw,
                None,
                0.0,
                submitted_at=s - dt.timedelta(minutes=10),
            )
        )
    _advance(store, _source(4), tmp_path)
    reloaded = store.get_participant(ids["trader"])
    assert reloaded is not None
    assert reloaded.soc_mwh < 5.0  # discharged down
    assert reloaded.throughput_internal_mwh > 0.0


def test_second_advance_only_processes_new_intervals(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "a")
    _advance(store, _source(4), tmp_path)
    assert store.count_decisions(ids["a"]) == 4

    later = NOW + dt.timedelta(minutes=30)
    new_starts = _intervals(4) + [NOW, NOW + dt.timedelta(minutes=15)]
    src2 = _FakeSource(_prices(new_starts, [80.0] * 6), _mcpc(new_starts, [9.0] * 6))
    summary = _advance(store, src2, tmp_path, now=later)
    assert summary.intervals_processed == 2  # only the two new ones
    assert store.count_decisions(ids["a"]) == 6


def test_no_new_intervals_is_a_noop(tmp_path: Path) -> None:
    store, _ = _store_with(tmp_path, "a")
    _advance(store, _source(4), tmp_path)
    summary = _advance(store, _source(4), tmp_path)
    assert summary.intervals_processed == 0


def test_deliverability_clamp_closes_the_empty_battery_as_exploit(tmp_path: Path) -> None:
    # A bot tries to stack full AS (2.5 MW RegUp) on an empty battery, no energy.
    store, ids = _store_with(tmp_path, "exploiter")
    target = _intervals(4)[1]
    store.queue_decision(
        QueuedDecision(
            ids["exploiter"],
            target,
            0.0,
            "regup",
            2.5,
            submitted_at=target - dt.timedelta(minutes=10),
        )
    )
    _advance(store, _source(4), tmp_path)  # SoC starts at 0 -> AS not deliverable
    row = store.connection.execute(
        "SELECT as_product, as_mw, as_revenue FROM decision_log "
        "WHERE participant_id = ? AND interval_start = ?",
        (ids["exploiter"], target.isoformat()),
    ).fetchone()
    assert row["as_mw"] == 0.0  # clamped: can't be paid to stand ready with no charge
    assert row["as_revenue"] == 0.0


def test_disabling_enforcement_reopens_the_exploit(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "exploiter")
    target = _intervals(4)[1]
    store.queue_decision(
        QueuedDecision(
            ids["exploiter"],
            target,
            0.0,
            "regup",
            2.5,
            submitted_at=target - dt.timedelta(minutes=10),
        )
    )
    advance_market(
        store,
        _source(4),
        now=NOW,
        lookback_hours=1.0,
        data_dir=tmp_path / "live",
        enforce_deliverability=False,
    )
    row = store.connection.execute(
        "SELECT as_mw FROM decision_log WHERE participant_id = ? AND interval_start = ?",
        (ids["exploiter"], target.isoformat()),
    ).fetchone()
    assert row["as_mw"] == 2.5  # unclamped when enforcement is off


def test_as_deployment_drains_a_standby_farmer(tmp_path: Path) -> None:
    # An AS farmer parks charge and commits RegUp every interval with idle energy.
    # Deployment now calls a slice of it each interval, discharging the battery --
    # so its charge falls and it logs nonzero energy throughput (it's no longer
    # free standby income).
    store, ids = _store_with(tmp_path, "farmer")
    _seed_soc(store, ids["farmer"], 8.0)
    for s in _intervals(4):
        store.queue_decision(
            QueuedDecision(
                ids["farmer"], s, 0.0, "regup", 2.0, submitted_at=s - dt.timedelta(minutes=10)
            )
        )
    _advance(store, _source(4), tmp_path)
    farmer = store.get_participant(ids["farmer"])
    assert farmer is not None
    assert farmer.soc_mwh < 8.0  # deployment discharged it
    moved = store.connection.execute(
        "SELECT COALESCE(SUM(ABS(energy_mwh)), 0) AS t FROM decision_log WHERE participant_id = ?",
        (ids["farmer"],),
    ).fetchone()
    assert moved["t"] > 0  # logged throughput -> will incur the wear fee


def test_disabling_deployment_leaves_standby_free(tmp_path: Path) -> None:
    store, ids = _store_with(tmp_path, "farmer")
    _seed_soc(store, ids["farmer"], 8.0)
    for s in _intervals(4):
        store.queue_decision(
            QueuedDecision(
                ids["farmer"], s, 0.0, "regup", 2.0, submitted_at=s - dt.timedelta(minutes=10)
            )
        )
    advance_market(
        store,
        _source(4),
        now=NOW,
        lookback_hours=1.0,
        data_dir=tmp_path / "live",
        model_as_deployment=False,
    )
    farmer = store.get_participant(ids["farmer"])
    assert farmer is not None
    assert farmer.soc_mwh == 8.0  # untouched when deployment is off


def test_advances_clock_with_zero_participants(tmp_path: Path) -> None:
    store = CompetitionStore(tmp_path / "comp.db")
    summary = _advance(store, _source(4), tmp_path)
    assert summary.participants == 0
    assert summary.intervals_processed == 4
    assert store.get_last_processed_interval() is not None
