"""SQLite-backed store for the live dispatch competition.

The multi-tenant counterpart to the single-player file store
(``live/state.py``): many participants, one shared market clock, queryable for
time-windowed leaderboards. SQLite keeps it zero-ops while handling concurrent
reads (from the MCP server) and serialized writes for a single-server
competition.

Tables:
  ``participants``    identity + battery physical state + held AS commitment
  ``decision_queue``  pending decisions for upcoming intervals (+ submission time)
  ``decision_log``    settled per-interval outcomes (revenue, was_default)
  ``market_meta``     key/value: last processed interval, etc.

Cumulative revenue is deliberately *not* cached on ``participants``: leaderboards
are time-windowed, so they aggregate ``decision_log`` directly -- a running total
would only duplicate it and risk drifting out of sync.
"""

from __future__ import annotations

import datetime as dt
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

_LAST_PROCESSED_KEY = "last_processed_interval"


def _iso(value: dt.datetime) -> str:
    return value.isoformat()


def _parse(value: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(value) if value else None


@dataclass
class Participant:
    """A competitor's identity plus the physical state of their battery."""

    id: str
    display_name: str
    created_at: dt.datetime
    email: str | None = None
    token: str = ""
    soc_mwh: float = 0.0
    energy_charged_mwh: float = 0.0
    energy_discharged_mwh: float = 0.0
    throughput_internal_mwh: float = 0.0
    held_as_product: str | None = None
    held_as_mw: float = 0.0
    held_committed_at: dt.datetime | None = None


@dataclass
class QueuedDecision:
    """A decision a participant has queued for an upcoming interval."""

    participant_id: str
    interval_start: dt.datetime
    energy_mw: float
    as_product: str | None
    as_mw: float
    submitted_at: dt.datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    id                      TEXT PRIMARY KEY,
    display_name            TEXT NOT NULL,
    created_at              TEXT NOT NULL,
    email                   TEXT,
    token                   TEXT NOT NULL UNIQUE,
    soc_mwh                 REAL NOT NULL DEFAULT 0,
    energy_charged_mwh      REAL NOT NULL DEFAULT 0,
    energy_discharged_mwh   REAL NOT NULL DEFAULT 0,
    throughput_internal_mwh REAL NOT NULL DEFAULT 0,
    held_as_product         TEXT,
    held_as_mw              REAL NOT NULL DEFAULT 0,
    held_committed_at       TEXT
);

CREATE TABLE IF NOT EXISTS decision_queue (
    participant_id  TEXT NOT NULL,
    interval_start  TEXT NOT NULL,
    energy_mw       REAL NOT NULL,
    as_product      TEXT,
    as_mw           REAL NOT NULL DEFAULT 0,
    submitted_at    TEXT NOT NULL,
    PRIMARY KEY (participant_id, interval_start)
);

CREATE TABLE IF NOT EXISTS decision_log (
    participant_id  TEXT NOT NULL,
    interval_start  TEXT NOT NULL,
    price           REAL NOT NULL,
    energy_mw       REAL NOT NULL,
    energy_mwh      REAL NOT NULL,
    as_product      TEXT,
    as_mw           REAL NOT NULL DEFAULT 0,
    energy_revenue  REAL NOT NULL,
    as_revenue      REAL NOT NULL,
    soc_mwh_after   REAL NOT NULL,
    was_default     INTEGER NOT NULL DEFAULT 0,
    reason          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (participant_id, interval_start)
);

CREATE INDEX IF NOT EXISTS idx_decision_log_interval ON decision_log (interval_start);

CREATE TABLE IF NOT EXISTS market_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class CompetitionStore:
    """SQLite store for participants, their decisions, and the market clock."""

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """The underlying connection, for leaderboard aggregation queries."""
        return self._conn

    def close(self) -> None:
        self._conn.close()

    # --- participants ------------------------------------------------------

    def register_participant(self, display_name: str, email: str | None = None) -> Participant:
        """Create a participant with a fresh id, secret token, and empty battery.

        The returned ``token`` is the participant's API credential -- they pass
        it on every subsequent call. It is shown only here, at registration.
        """
        participant = Participant(
            id=uuid.uuid4().hex[:12],
            display_name=display_name,
            created_at=dt.datetime.now(dt.UTC),
            email=email,
            token=secrets.token_urlsafe(24),
        )
        self._conn.execute(
            "INSERT INTO participants (id, display_name, created_at, email, token) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                participant.id,
                participant.display_name,
                _iso(participant.created_at),
                participant.email,
                participant.token,
            ),
        )
        self._conn.commit()
        return participant

    def get_participant(self, participant_id: str) -> Participant | None:
        row = self._conn.execute(
            "SELECT * FROM participants WHERE id = ?", (participant_id,)
        ).fetchone()
        return _participant_from_row(row) if row is not None else None

    def get_participant_by_token(self, token: str) -> Participant | None:
        """Look up a participant by their secret token (for authentication)."""
        row = self._conn.execute("SELECT * FROM participants WHERE token = ?", (token,)).fetchone()
        return _participant_from_row(row) if row is not None else None

    def list_participants(self) -> list[Participant]:
        rows = self._conn.execute("SELECT * FROM participants ORDER BY created_at").fetchall()
        return [_participant_from_row(row) for row in rows]

    def save_participant_state(self, participant: Participant) -> None:
        """Persist a participant's battery state after stepping them."""
        self._conn.execute(
            """
            UPDATE participants SET
                soc_mwh = ?, energy_charged_mwh = ?, energy_discharged_mwh = ?,
                throughput_internal_mwh = ?, held_as_product = ?, held_as_mw = ?,
                held_committed_at = ?
            WHERE id = ?
            """,
            (
                participant.soc_mwh,
                participant.energy_charged_mwh,
                participant.energy_discharged_mwh,
                participant.throughput_internal_mwh,
                participant.held_as_product,
                participant.held_as_mw,
                _iso(participant.held_committed_at) if participant.held_committed_at else None,
                participant.id,
            ),
        )
        self._conn.commit()

    # --- decision queue ----------------------------------------------------

    def queue_decision(self, decision: QueuedDecision) -> None:
        """Insert or replace a participant's queued decision for an interval."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO decision_queue
                (participant_id, interval_start, energy_mw, as_product, as_mw, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                decision.participant_id,
                _iso(decision.interval_start),
                decision.energy_mw,
                decision.as_product,
                decision.as_mw,
                _iso(decision.submitted_at),
            ),
        )
        self._conn.commit()

    def get_queued_decision(
        self, participant_id: str, interval_start: dt.datetime
    ) -> QueuedDecision | None:
        row = self._conn.execute(
            "SELECT * FROM decision_queue WHERE participant_id = ? AND interval_start = ?",
            (participant_id, _iso(interval_start)),
        ).fetchone()
        return _queued_from_row(row) if row is not None else None

    # --- decision log ------------------------------------------------------

    def record_decision(
        self,
        *,
        participant_id: str,
        interval_start: dt.datetime,
        price: float,
        energy_mw: float,
        energy_mwh: float,
        as_product: str | None,
        as_mw: float,
        energy_revenue: float,
        as_revenue: float,
        soc_mwh_after: float,
        was_default: bool,
        reason: str,
    ) -> None:
        """Append a settled interval outcome to the immutable decision log."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO decision_log
                (participant_id, interval_start, price, energy_mw, energy_mwh,
                 as_product, as_mw, energy_revenue, as_revenue, soc_mwh_after,
                 was_default, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                participant_id,
                _iso(interval_start),
                price,
                energy_mw,
                energy_mwh,
                as_product,
                as_mw,
                energy_revenue,
                as_revenue,
                soc_mwh_after,
                int(was_default),
                reason,
            ),
        )
        self._conn.commit()

    def count_decisions(self, participant_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM decision_log WHERE participant_id = ?",
            (participant_id,),
        ).fetchone()
        return int(row["n"])

    # --- market clock ------------------------------------------------------

    def get_market_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM market_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None

    def set_market_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO market_meta (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def get_last_processed_interval(self) -> dt.datetime | None:
        return _parse(self.get_market_meta(_LAST_PROCESSED_KEY))

    def set_last_processed_interval(self, interval_start: dt.datetime) -> None:
        self.set_market_meta(_LAST_PROCESSED_KEY, _iso(interval_start))


def _participant_from_row(row: sqlite3.Row) -> Participant:
    return Participant(
        id=row["id"],
        display_name=row["display_name"],
        created_at=dt.datetime.fromisoformat(row["created_at"]),
        email=row["email"],
        token=row["token"],
        soc_mwh=row["soc_mwh"],
        energy_charged_mwh=row["energy_charged_mwh"],
        energy_discharged_mwh=row["energy_discharged_mwh"],
        throughput_internal_mwh=row["throughput_internal_mwh"],
        held_as_product=row["held_as_product"],
        held_as_mw=row["held_as_mw"],
        held_committed_at=_parse(row["held_committed_at"]),
    )


def _queued_from_row(row: sqlite3.Row) -> QueuedDecision:
    return QueuedDecision(
        participant_id=row["participant_id"],
        interval_start=dt.datetime.fromisoformat(row["interval_start"]),
        energy_mw=row["energy_mw"],
        as_product=row["as_product"],
        as_mw=row["as_mw"],
        submitted_at=dt.datetime.fromisoformat(row["submitted_at"]),
    )


__all__ = ["CompetitionStore", "Participant", "QueuedDecision"]
