"""Perfect-foresight benchmark strategy.

This is NOT a deployable strategy -- it assumes complete knowledge of every
future price. It solves a linear program for the dispatch that maximizes
revenue subject to the battery's constraints, giving the theoretical ceiling
against which real strategies are measured (see `metrics.capture_rate`).

LP formulation, per interval t (all energies are grid-side MWh):

    variables   charge[t]    >= 0
                discharge[t] >= 0
                soc[t]                       (state of charge, MWh)

    maximize    sum( (discharge[t] - charge[t]) * price[t] )
                  - epsilon * sum( charge[t] + discharge[t] )

    subject to  charge[t]    <= power_mw * hours
                discharge[t] <= power_mw * hours
                soc[t] = soc[t-1] + charge[t] * eff - discharge[t] / eff
                0 <= soc[t] <= capacity_mwh

`eff` is the one-way efficiency (sqrt of round-trip). Revenue depends only on
the net dispatch `discharge[t] - charge[t]`, so no integer variables are
needed to forbid simultaneous charge and discharge -- it is already dominated.
The tiny `epsilon` throughput penalty makes that explicit and also breaks
revenue ties toward fewer cycles; at 1e-7 it shifts annual revenue by far less
than a cent.
"""

from __future__ import annotations

import pulp

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.data.schemas import RTM_INTERVAL_MINUTES
from dispatcher_watts.strategies.base import Strategy

# Throughput penalty -- see the module docstring.
_THROUGHPUT_PENALTY: float = 1e-7


def solve_perfect_foresight_dispatch(
    prices: list[float],
    spec: BatterySpec,
    hours: float,
    initial_soc_mwh: float = 0.0,
) -> list[float]:
    """Return the revenue-maximizing net grid dispatch per interval.

    Each entry is grid-side MWh: positive = discharge, negative = charge.
    Raises `RuntimeError` if the solver does not reach an optimal solution.
    """
    n = len(prices)
    if n == 0:
        return []

    efficiency = spec.one_way_efficiency
    max_grid_mwh = spec.power_mw * hours
    capacity = spec.capacity_mwh

    problem = pulp.LpProblem("perfect_foresight", pulp.LpMaximize)
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_grid_mwh) for t in range(n)]
    discharge = [
        pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_grid_mwh) for t in range(n)
    ]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=capacity) for t in range(n)]

    problem += pulp.lpSum(
        (discharge[t] - charge[t]) * prices[t] - _THROUGHPUT_PENALTY * (charge[t] + discharge[t])
        for t in range(n)
    )
    for t in range(n):
        previous = soc[t - 1] if t > 0 else initial_soc_mwh
        problem += soc[t] == previous + charge[t] * efficiency - discharge[t] / efficiency

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"perfect-foresight LP did not solve: {pulp.LpStatus[status]}")

    return [(discharge[t].value() or 0.0) - (charge[t].value() or 0.0) for t in range(n)]


class PerfectForesightStrategy(Strategy):
    """Replays the revenue-maximizing dispatch found by the LP above.

    Needs the battery spec up front because the LP encodes the battery's
    power, capacity, and efficiency. The whole dispatch is solved once in
    `prepare`; `decide` just looks up the precomputed action.
    """

    name = "perfect-foresight"

    def __init__(
        self,
        spec: BatterySpec,
        interval_minutes: int = RTM_INTERVAL_MINUTES,
        initial_soc_mwh: float = 0.0,
    ) -> None:
        self._spec = spec
        self._hours = interval_minutes / 60.0
        self._initial_soc_mwh = initial_soc_mwh
        self._actions: list[float] = []

    def prepare(self, prices: list[float]) -> None:
        net = solve_perfect_foresight_dispatch(
            prices, self._spec, self._hours, self._initial_soc_mwh
        )
        max_grid_mwh = self._spec.power_mw * self._hours
        self._actions = [value / max_grid_mwh for value in net]

    def decide(self, step: int, prices: list[float], soc_fraction: float) -> float:
        return self._actions[step]
