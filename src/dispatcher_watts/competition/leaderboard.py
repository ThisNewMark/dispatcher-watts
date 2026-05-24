"""Time-windowed leaderboards over the competition decision log.

Rankings for a window (this hour / day / week / month / all-time) are pure
aggregations over ``decision_log`` plus a single perfect-foresight LP solve for
the window (the ceiling is the same for everyone, since all participants share
one battery spec and one price tape). Capture rate -- revenue as a fraction of
that ceiling -- is the longevity-neutral metric: a great strategy ranks high
whether it has played a day or a year.

The score is *net* of a flat per-MWh wear fee on energy throughput, so
thin-spread over-cycling -- profitable on paper but a money-loser on a real
battery -- doesn't win. The ceiling is solved degradation-aware and netted the
same way, keeping capture rate apples-to-apples. ``gross_revenue`` and
``degradation_cost`` are reported alongside for transparency.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import cast

import polars as pl

from dispatcher_watts.backtest.metrics import capture_rate
from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.competition import COMPETITION_DEGRADATION_COST_PER_MWH, COMPETITION_SPEC
from dispatcher_watts.competition.store import CompetitionStore
from dispatcher_watts.cooptimization.solver import solve_co_optimization
from dispatcher_watts.data.schemas import RTM_INTERVAL_MINUTES

# Supported leaderboard windows and the sortable metrics.
WINDOWS: tuple[str, ...] = ("hour", "day", "week", "month", "all-time")
SORT_KEYS: tuple[str, ...] = ("revenue", "capture_rate", "cycles")

_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


@dataclass
class LeaderboardEntry:
    """One participant's standing within a window.

    ``revenue`` is the score: gross trading + standby revenue minus the wear fee
    on energy throughput. ``gross_revenue`` and ``degradation_cost`` are shown
    for transparency.
    """

    participant_id: str
    display_name: str
    revenue: float
    gross_revenue: float
    energy_revenue: float
    as_revenue: float
    degradation_cost: float
    capture_rate: float
    equivalent_cycles: float
    intervals: int


@dataclass
class Leaderboard:
    """A ranked board for one window."""

    window: str
    start: dt.datetime
    end: dt.datetime
    ceiling_revenue: float
    entries: list[LeaderboardEntry]


def window_bounds(window: str, now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    """Return ``[start, end)`` in UTC for a named window ending at ``now``.

    Windows are calendar-aligned (this hour / today / this ISO week / this
    month), so every participant is scored on the same intervals within a
    window regardless of when they joined. ``all-time`` runs from the epoch.
    """
    now = now.astimezone(dt.UTC)
    if window == "hour":
        start = now.replace(minute=0, second=0, microsecond=0)
    elif window == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif window == "week":
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = midnight - dt.timedelta(days=now.weekday())  # back to Monday
    elif window == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif window == "all-time":
        start = _EPOCH
    else:
        raise ValueError(f"unknown window {window!r}; expected one of {WINDOWS}")
    return start, now


def compute_leaderboard(
    store: CompetitionStore,
    prices: pl.DataFrame,
    mcpc: pl.DataFrame,
    *,
    window: str,
    now: dt.datetime,
    spec: BatterySpec = COMPETITION_SPEC,
    interval_minutes: int = RTM_INTERVAL_MINUTES,
    sort_by: str = "revenue",
    degradation_cost_per_mwh: float = COMPETITION_DEGRADATION_COST_PER_MWH,
) -> Leaderboard:
    """Rank all participants over ``window``.

    ``prices`` / ``mcpc`` are the captured market tape (any range covering the
    window); they are sliced to the window here and fed to the foresight LP for
    the ceiling. Both the ceiling and each participant's score are net of the
    per-MWh wear fee, so the comparison stays apples-to-apples. ``sort_by`` is
    one of ``SORT_KEYS``.
    """
    if sort_by not in SORT_KEYS:
        raise ValueError(f"unknown sort_by {sort_by!r}; expected one of {SORT_KEYS}")
    start, end = window_bounds(window, now)
    ceiling = _ceiling_net_revenue(
        prices, mcpc, start, end, spec, interval_minutes, degradation_cost_per_mwh
    )
    rows = _aggregate_rows(store, start, end)

    entries = []
    for row in rows:
        energy_revenue = cast(float, row["e"])
        as_revenue = cast(float, row["a"])
        throughput = cast(float, row["throughput"])
        gross = energy_revenue + as_revenue
        degradation_cost = throughput * degradation_cost_per_mwh
        net = gross - degradation_cost
        entries.append(
            LeaderboardEntry(
                participant_id=cast(str, row["id"]),
                display_name=cast(str, row["display_name"]),
                revenue=net,
                gross_revenue=gross,
                energy_revenue=energy_revenue,
                as_revenue=as_revenue,
                degradation_cost=degradation_cost,
                capture_rate=capture_rate(net, ceiling),
                equivalent_cycles=cast(float, row["discharge"]) / spec.capacity_mwh,
                intervals=cast(int, row["n"]),
            )
        )
    _sort_entries(entries, sort_by)
    return Leaderboard(
        window=window, start=start, end=end, ceiling_revenue=ceiling, entries=entries
    )


def _aggregate_rows(
    store: CompetitionStore, start: dt.datetime, end: dt.datetime
) -> list[dict[str, object]]:
    """Per-participant revenue/cycles/intervals over ``[start, end)``."""
    cursor = store.connection.execute(
        """
        SELECT p.id AS id, p.display_name AS display_name,
               COALESCE(SUM(d.energy_revenue), 0) AS e,
               COALESCE(SUM(d.as_revenue), 0) AS a,
               COUNT(d.interval_start) AS n,
               COALESCE(
                   SUM(CASE WHEN d.energy_mwh > 0 THEN d.energy_mwh ELSE 0 END), 0
               ) AS discharge,
               COALESCE(SUM(ABS(d.energy_mwh)), 0) AS throughput
        FROM participants p
        LEFT JOIN decision_log d
            ON d.participant_id = p.id
            AND d.interval_start >= ? AND d.interval_start < ?
        GROUP BY p.id, p.display_name
        """,
        (start.isoformat(), end.isoformat()),
    )
    return [dict(row) for row in cursor.fetchall()]


def _ceiling_net_revenue(
    prices: pl.DataFrame,
    mcpc: pl.DataFrame,
    start: dt.datetime,
    end: dt.datetime,
    spec: BatterySpec,
    interval_minutes: int,
    degradation_cost_per_mwh: float,
) -> float:
    """Perfect-foresight revenue over the window, net of the wear fee.

    The LP is solved degradation-aware (its optimal dispatch already trades
    cycling against revenue), then the same per-MWh wear fee is subtracted from
    its gross -- matching how each participant is scored. 0 if the window is
    empty.
    """
    prices_w = prices.filter((pl.col("interval_start") >= start) & (pl.col("interval_start") < end))
    mcpc_w = mcpc.filter((pl.col("interval_start") >= start) & (pl.col("interval_start") < end))
    try:
        result = solve_co_optimization(
            prices_w,
            mcpc_w,
            spec,
            interval_minutes,
            degradation_cost_per_mwh=degradation_cost_per_mwh,
        )
    except ValueError:
        # No overlapping intervals in the window -> nothing to capture.
        return 0.0
    throughput = float(
        result.frame.select(pl.col("charge_mwh").sum() + pl.col("discharge_mwh").sum()).item()
    )
    return result.total_revenue - throughput * degradation_cost_per_mwh


def _sort_entries(entries: list[LeaderboardEntry], sort_by: str) -> None:
    key = {
        "revenue": lambda e: e.revenue,
        "capture_rate": lambda e: e.capture_rate,
        "cycles": lambda e: e.equivalent_cycles,
    }[sort_by]
    entries.sort(key=key, reverse=True)


__all__ = [
    "SORT_KEYS",
    "WINDOWS",
    "Leaderboard",
    "LeaderboardEntry",
    "compute_leaderboard",
    "window_bounds",
]
