"""Strategy interface for battery dispatch.

A strategy decides, at each interval, how hard to charge or discharge. The
decision is a single float in [-1, 1]:

    -1   charge at full power
     0   idle
    +1   discharge at full power

Fractional values are allowed -- the perfect-foresight benchmark uses them --
while the simple rule-based strategies return only -1, 0, or +1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Strategy(ABC):
    """Base class for dispatch strategies."""

    name: str = "strategy"

    def prepare(self, prices: list[float]) -> None:
        """Optional pre-computation hook, called once before the backtest.

        Strategies that need the whole price path up front (such as a solver)
        override this. The default does nothing.
        """
        return

    @abstractmethod
    def decide(self, step: int, prices: list[float], soc_fraction: float) -> float:
        """Return the dispatch decision for interval `step`, in [-1, 1].

        Args:
            step: index of the current interval.
            prices: the full price series ($/MWh), one entry per interval.
            soc_fraction: battery state of charge as a fraction of capacity.
        """
