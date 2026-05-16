"""The backtest engine: replay a price series, dispatch the battery, tally revenue.

Conceptually this is a financial backtester whose asset is a battery and whose
market is ERCOT. At each interval the strategy returns a dispatch decision, the
battery executes as much of it as its constraints allow, and revenue accrues:

    revenue = grid_energy_mwh * price        ($/MWh)

where `grid_energy_mwh` is positive when discharging (selling) and negative
when charging (buying).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from dispatcher_watts.battery.model import Battery, BatterySpec
from dispatcher_watts.data.schemas import RTM_INTERVAL_MINUTES, validate_rtm_frame
from dispatcher_watts.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Outcome of a backtest run.

    `frame` has one row per interval, with columns: interval_start, price,
    action ([-1, 1]), grid_energy_mwh (+discharge / -charge), soc_mwh,
    revenue, cumulative_revenue.
    """

    strategy_name: str
    spec: BatterySpec
    interval_minutes: int
    frame: pl.DataFrame
    final_battery: Battery


def run_backtest(
    prices: pl.DataFrame,
    battery: Battery,
    strategy: Strategy,
    interval_minutes: int = RTM_INTERVAL_MINUTES,
) -> BacktestResult:
    """Run `strategy` on `battery` over the `prices` series.

    `prices` must match `RTM_PRICE_SCHEMA`. The battery is mutated in place, so
    pass a fresh one per run.
    """
    validate_rtm_frame(prices)
    hours = interval_minutes / 60.0
    price_list: list[float] = prices["price"].to_list()
    timestamps = prices["interval_start"]

    strategy.prepare(price_list)

    actions: list[float] = []
    grid_energy: list[float] = []
    soc: list[float] = []
    revenue: list[float] = []

    for step, price in enumerate(price_list):
        action = max(-1.0, min(1.0, strategy.decide(step, price_list, battery.soc_fraction)))
        if action < 0:
            energy = -battery.charge(-action * battery.spec.power_mw * hours, hours)
        elif action > 0:
            energy = battery.discharge(action * battery.spec.power_mw * hours, hours)
        else:
            energy = 0.0
        actions.append(action)
        grid_energy.append(energy)
        soc.append(battery.soc_mwh)
        revenue.append(energy * price)

    frame = pl.DataFrame(
        {
            "interval_start": timestamps,
            "price": price_list,
            "action": actions,
            "grid_energy_mwh": grid_energy,
            "soc_mwh": soc,
            "revenue": revenue,
        }
    ).with_columns(cumulative_revenue=pl.col("revenue").cum_sum())

    return BacktestResult(
        strategy_name=strategy.name,
        spec=battery.spec,
        interval_minutes=interval_minutes,
        frame=frame,
        final_battery=battery,
    )
