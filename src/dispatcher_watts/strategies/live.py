"""Deployable live strategies for paper trading.

Unlike the foresight LP (``cooptimization/solver.py``), the strategies here
decide using only information available *now* -- the current real-time price,
the latest *indicative* AS clearing prices, and the battery's own state. They
never peek at the future, so they are deployable: the same code that runs a
historical replay could run against ERCOT's live feed unchanged.

The interface is intentionally separate from the energy-only ``Strategy`` ABC
(``strategies/base.py``). A live decision allocates the battery across two legs
at once -- real-time energy *and* one ancillary-services product -- so it needs
a richer return type than the single ``[-1, 1]`` scalar the backtest engine
consumes.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass

from dispatcher_watts.data.schemas import MCPC_PRODUCTS


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything a live strategy may look at for a single interval.

    Strictly present/past information. ``mcpc`` maps each AS product (the
    suffixes in ``MCPC_PRODUCTS``) to its latest indicative clearing price in
    $/MW for the interval; a product absent from the dict is treated as having
    no quote and is not eligible to be the leader.
    """

    timestamp: dt.datetime
    price: float
    mcpc: dict[str, float]


@dataclass(frozen=True)
class ASCommitment:
    """The ancillary-services leg currently held, carried between ticks.

    ``product`` is ``None`` when nothing is committed. ``committed_at`` is the
    timestamp at which this leg was last (re)selected -- the allocation-interval
    logic compares against it to decide whether to re-pick the leader.
    """

    product: str | None
    mw: float
    committed_at: dt.datetime


@dataclass(frozen=True)
class LiveDecision:
    """A deployable dispatch decision for one interval.

    ``energy_mw`` is signed power for the energy leg: negative charges, positive
    discharges, zero idles. It is the average power held over the interval; the
    caller multiplies by the interval length and lets the battery model clamp to
    what is physically feasible. ``as_product`` / ``as_mw`` is the AS capacity
    reservation, and ``as_committed_at`` records when that leg was last
    (re)selected so the loop can persist and honor the allocation interval.
    """

    energy_mw: float
    as_product: str | None
    as_mw: float
    as_committed_at: dt.datetime
    reason: str = ""

    def commitment(self) -> ASCommitment:
        """The AS leg of this decision, to carry into the next tick."""
        return ASCommitment(self.as_product, self.as_mw, self.as_committed_at)


class LiveStrategy(ABC):
    """Base class for deployable, foresight-free dispatch strategies."""

    name: str = "live-strategy"

    @abstractmethod
    def decide(
        self,
        snapshot: MarketSnapshot,
        soc_fraction: float,
        power_mw: float,
        held: ASCommitment | None,
    ) -> LiveDecision:
        """Decide energy + AS dispatch for the interval at ``snapshot``.

        Args:
            snapshot: present-time market state (price + latest indicative MCPCs).
            soc_fraction: battery state of charge as a fraction of capacity.
            power_mw: the battery's power rating (MW), the envelope shared by the
                energy and AS legs.
            held: the AS leg carried over from the previous tick, or ``None`` on
                the first tick.
        """


def _select_leader(mcpc: dict[str, float]) -> str | None:
    """Return the product with the highest positive MCPC, or ``None``.

    Ties break toward the earlier product in ``MCPC_PRODUCTS`` order, so the
    choice is deterministic regardless of dict iteration order.
    """
    leader: str | None = None
    best = 0.0
    for product in MCPC_PRODUCTS:
        quote = mcpc.get(product)
        if quote is None or quote <= best:
            continue
        best = quote
        leader = product
    return leader


class FollowTheLeaderStrategy(LiveStrategy):
    """Follow the leader for AS, threshold for energy.

    Each interval the strategy reserves a fixed fraction of the battery's power
    for the single AS product with the highest indicative MCPC ("the leader"),
    and trades the remaining power on a simple price threshold for energy. It is
    a heuristic, not an optimizer: defensible, transparent, and -- per the
    project's design notes -- expected to capture roughly 60-80% of what the
    foresight co-optimization LP would earn.

    The AS leg is only re-selected once ``allocation_interval_minutes`` have
    elapsed since the last selection. With the default of one RTD cadence
    (5 min) the leader is re-picked every tick, faithful to RTC+B's 5-minute
    real-time co-optimization. Set it to 60 to lock the leg for an hour, closer
    to how an operator commits AS day-ahead. Anything in between is a config
    change, not a code change.

    Note (named honestly for the eventual write-up): the AS reservation here is
    a capacity commitment that does *not* guarantee the state of charge needed
    to deliver a discharge-direction product if fully called -- the foresight LP
    enforces that SoC reservation, this heuristic does not.
    """

    name = "follow-the-leader"

    def __init__(
        self,
        charge_below: float,
        discharge_above: float,
        as_capacity_fraction: float = 0.5,
        allocation_interval_minutes: float = 5.0,
    ) -> None:
        if charge_below >= discharge_above:
            raise ValueError(
                f"charge_below ({charge_below}) must be less than "
                f"discharge_above ({discharge_above})"
            )
        if not 0.0 <= as_capacity_fraction <= 1.0:
            raise ValueError(f"as_capacity_fraction ({as_capacity_fraction}) must be in [0, 1]")
        if allocation_interval_minutes <= 0:
            raise ValueError(
                f"allocation_interval_minutes ({allocation_interval_minutes}) must be positive"
            )
        self.charge_below = charge_below
        self.discharge_above = discharge_above
        self.as_capacity_fraction = as_capacity_fraction
        self.allocation_interval_minutes = allocation_interval_minutes

    def decide(
        self,
        snapshot: MarketSnapshot,
        soc_fraction: float,
        power_mw: float,
        held: ASCommitment | None,
    ) -> LiveDecision:
        as_product, as_mw, committed_at = self._resolve_as_leg(snapshot, power_mw, held)

        # Energy trades on whatever power the AS leg did not reserve. When no
        # product is worth committing to, the full rating is available for energy.
        energy_power_mw = power_mw - as_mw
        price = snapshot.price
        if price <= self.charge_below:
            energy_mw = -energy_power_mw
            energy_note = f"charge {energy_power_mw:.3f}MW @ {price:g}"
        elif price >= self.discharge_above:
            energy_mw = energy_power_mw
            energy_note = f"discharge {energy_power_mw:.3f}MW @ {price:g}"
        else:
            energy_mw = 0.0
            energy_note = f"idle @ {price:g}"

        if as_product is None:
            as_note = "AS none (no positive MCPC)"
        else:
            as_note = f"AS {as_product} {as_mw:.3f}MW @ {snapshot.mcpc.get(as_product, 0.0):g}/MW"

        return LiveDecision(
            energy_mw=energy_mw,
            as_product=as_product,
            as_mw=as_mw,
            as_committed_at=committed_at,
            reason=f"{energy_note}; {as_note}",
        )

    def _resolve_as_leg(
        self,
        snapshot: MarketSnapshot,
        power_mw: float,
        held: ASCommitment | None,
    ) -> tuple[str | None, float, dt.datetime]:
        """Pick the AS product/MW/commit-time, honoring the allocation interval.

        Holds the existing leg while the interval has not elapsed; otherwise
        (or when nothing is held) re-selects the current highest-MCPC product.
        """
        if self._should_hold(snapshot.timestamp, held):
            assert held is not None  # _should_hold is False when held is None
            return held.product, held.mw, held.committed_at

        leader = _select_leader(snapshot.mcpc)
        as_mw = self.as_capacity_fraction * power_mw if leader is not None else 0.0
        return leader, as_mw, snapshot.timestamp

    def _should_hold(self, now: dt.datetime, held: ASCommitment | None) -> bool:
        if held is None or held.product is None:
            return False
        elapsed = now - held.committed_at
        return elapsed < dt.timedelta(minutes=self.allocation_interval_minutes)


__all__ = [
    "ASCommitment",
    "FollowTheLeaderStrategy",
    "LiveDecision",
    "LiveStrategy",
    "MarketSnapshot",
]
