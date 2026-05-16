"""Threshold dispatch strategy: charge when cheap, discharge when expensive."""

from __future__ import annotations

from dispatcher_watts.strategies.base import Strategy


class ThresholdStrategy(Strategy):
    """Charge below a low price, discharge above a high price, otherwise idle.

    The simplest possible arbitrage rule, and a useful baseline: it ignores the
    state of charge and the time of day, acting purely on the current price.
    """

    name = "threshold"

    def __init__(self, charge_below: float, discharge_above: float) -> None:
        if charge_below >= discharge_above:
            raise ValueError(
                f"charge_below ({charge_below}) must be less than "
                f"discharge_above ({discharge_above})"
            )
        self.charge_below = charge_below
        self.discharge_above = discharge_above

    def decide(self, step: int, prices: list[float], soc_fraction: float) -> float:
        price = prices[step]
        if price <= self.charge_below:
            return -1.0
        if price >= self.discharge_above:
            return 1.0
        return 0.0
