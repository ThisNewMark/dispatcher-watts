"""Rolling-average dispatch strategy.

Charge when the current price is below the trailing N-hour average, discharge
when it is above. Unlike the threshold strategy, the reference point adapts to
recent market conditions instead of being a fixed number.
"""

from __future__ import annotations

from dispatcher_watts.data.schemas import RTM_INTERVAL_MINUTES
from dispatcher_watts.strategies.base import Strategy


class RollingAverageStrategy(Strategy):
    """Charge below the trailing rolling-average price, discharge above it.

    `band` widens a no-trade zone around the average: the battery acts only
    when the price is at least `band` (as a fraction of the average) away from
    it. A `band` of 0 reproduces the plain "below average / above average" rule.
    """

    name = "rolling-average"

    def __init__(
        self,
        window_hours: float = 24.0,
        interval_minutes: int = RTM_INTERVAL_MINUTES,
        band: float = 0.0,
    ) -> None:
        if window_hours <= 0:
            raise ValueError("window_hours must be positive")
        if not 0 <= band < 1:
            raise ValueError("band must be in [0, 1)")
        self.window_hours = window_hours
        self.band = band
        self._window = max(1, round(window_hours * 60 / interval_minutes))
        self._averages: list[float | None] = []

    def prepare(self, prices: list[float]) -> None:
        # Precompute the trailing average for every interval via a prefix sum.
        prefix = [0.0]
        for price in prices:
            prefix.append(prefix[-1] + price)
        self._averages = []
        for i in range(len(prices)):
            low = max(0, i - self._window)
            count = i - low
            # No history yet for the first interval -> no reference average.
            self._averages.append(None if count == 0 else (prefix[i] - prefix[low]) / count)

    def decide(self, step: int, prices: list[float], soc_fraction: float) -> float:
        average = self._averages[step]
        if average is None:
            return 0.0
        spread = abs(average) * self.band
        price = prices[step]
        if price < average - spread:
            return -1.0
        if price > average + spread:
            return 1.0
        return 0.0
