"""Tests for the SQLite competition store (competition/store.py)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from dispatcher_watts.competition.store import CompetitionStore, QueuedDecision

T0 = dt.datetime(2026, 5, 22, 12, 0, tzinfo=dt.UTC)


def _store() -> CompetitionStore:
    return CompetitionStore(":memory:")


# --- participants ------------------------------------------------------------


def test_register_and_get_participant() -> None:
    store = _store()
    p = store.register_participant("alice-bot")
    fetched = store.get_participant(p.id)
    assert fetched is not None
    assert fetched.display_name == "alice-bot"
    assert fetched.soc_mwh == 0.0
    assert fetched.held_as_product is None


def test_register_gives_distinct_ids() -> None:
    store = _store()
    a = store.register_participant("a")
    b = store.register_participant("b")
    assert a.id != b.id
    assert {p.id for p in store.list_participants()} == {a.id, b.id}


def test_get_unknown_participant_returns_none() -> None:
    assert _store().get_participant("nope") is None


def test_register_issues_token_and_stores_email() -> None:
    store = _store()
    p = store.register_participant("alice-bot", email="alice@example.com")
    assert p.token  # a non-empty secret was issued
    assert p.email == "alice@example.com"


def test_distinct_participants_get_distinct_tokens() -> None:
    store = _store()
    assert store.register_participant("a").token != store.register_participant("b").token


def test_authenticate_by_token() -> None:
    store = _store()
    p = store.register_participant("bot", email="b@x.com")
    found = store.get_participant_by_token(p.token)
    assert found is not None
    assert found.id == p.id
    assert store.get_participant_by_token("wrong-token") is None


def test_save_and_reload_participant_state() -> None:
    store = _store()
    p = store.register_participant("bot")
    p.soc_mwh = 4.0
    p.throughput_internal_mwh = 12.5
    p.held_as_product = "rrs"
    p.held_as_mw = 1.25
    p.held_committed_at = T0
    store.save_participant_state(p)

    reloaded = store.get_participant(p.id)
    assert reloaded is not None
    assert reloaded.soc_mwh == 4.0
    assert reloaded.throughput_internal_mwh == 12.5
    assert reloaded.held_as_product == "rrs"
    assert reloaded.held_committed_at == T0


# --- decision queue ----------------------------------------------------------


def test_queue_and_get_decision() -> None:
    store = _store()
    p = store.register_participant("bot")
    store.queue_decision(
        QueuedDecision(p.id, T0, energy_mw=2.5, as_product="regup", as_mw=1.25, submitted_at=T0)
    )
    got = store.get_queued_decision(p.id, T0)
    assert got is not None
    assert got.energy_mw == 2.5
    assert got.as_product == "regup"


def test_queue_decision_replaces_on_resubmit() -> None:
    store = _store()
    p = store.register_participant("bot")
    store.queue_decision(QueuedDecision(p.id, T0, 2.5, "regup", 1.25, T0))
    store.queue_decision(QueuedDecision(p.id, T0, -2.5, "rrs", 0.5, T0 + dt.timedelta(minutes=1)))
    got = store.get_queued_decision(p.id, T0)
    assert got is not None
    assert got.energy_mw == -2.5  # latest submission wins
    assert got.as_product == "rrs"


def test_get_missing_queued_decision_returns_none() -> None:
    store = _store()
    p = store.register_participant("bot")
    assert store.get_queued_decision(p.id, T0) is None


# --- decision log ------------------------------------------------------------


def test_record_and_count_decisions() -> None:
    store = _store()
    p = store.register_participant("bot")
    for i in range(3):
        store.record_decision(
            participant_id=p.id,
            interval_start=T0 + dt.timedelta(minutes=15 * i),
            price=80.0,
            energy_mw=2.5,
            energy_mwh=0.625,
            as_product="rrs",
            as_mw=1.25,
            energy_revenue=50.0,
            as_revenue=11.25,
            soc_mwh_after=5.0,
            was_default=False,
            reason="test",
        )
    assert store.count_decisions(p.id) == 3


def test_decision_log_is_queryable_for_aggregation() -> None:
    store = _store()
    p = store.register_participant("bot")
    store.record_decision(
        participant_id=p.id,
        interval_start=T0,
        price=80.0,
        energy_mw=2.5,
        energy_mwh=0.625,
        as_product="rrs",
        as_mw=1.25,
        energy_revenue=50.0,
        as_revenue=11.25,
        soc_mwh_after=5.0,
        was_default=True,
        reason="default-fill",
    )
    row = store.connection.execute(
        "SELECT energy_revenue + as_revenue AS total, was_default FROM decision_log"
    ).fetchone()
    assert row["total"] == 61.25
    assert row["was_default"] == 1


# --- market clock ------------------------------------------------------------


def test_market_meta_round_trip() -> None:
    store = _store()
    assert store.get_market_meta("foo") is None
    store.set_market_meta("foo", "bar")
    assert store.get_market_meta("foo") == "bar"


def test_last_processed_interval_round_trip() -> None:
    store = _store()
    assert store.get_last_processed_interval() is None
    store.set_last_processed_interval(T0)
    assert store.get_last_processed_interval() == T0


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "comp.db"
    store = CompetitionStore(db)
    p = store.register_participant("durable-bot")
    store.set_last_processed_interval(T0)
    store.close()

    reopened = CompetitionStore(db)
    assert reopened.get_participant(p.id) is not None
    assert reopened.get_last_processed_interval() == T0
