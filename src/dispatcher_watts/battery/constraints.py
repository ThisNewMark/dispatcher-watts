"""Battery dispatch constraints.

Pure functions that compute the feasible grid-side energy for a charge or a
discharge over one interval, given the battery's state and rating. Keeping the
limit logic separate from `Battery` makes it independently testable.

Energy bookkeeping: `one_way_efficiency` is applied on each leg (charge and
discharge), so a full charge-then-discharge round trip retains
``one_way_efficiency ** 2 == round_trip_efficiency`` of the energy. All
energies here are *grid-side* (measured at the interconnection).
"""

from __future__ import annotations


def max_charge_grid_mwh(
    soc_mwh: float,
    capacity_mwh: float,
    power_mw: float,
    one_way_efficiency: float,
    hours: float,
) -> float:
    """Largest grid-side energy (MWh) the battery can absorb this interval.

    Limited by the power rating and by remaining headroom. Grid energy ``e``
    raises the state of charge by ``e * one_way_efficiency``.
    """
    by_power = power_mw * hours
    by_headroom = (capacity_mwh - soc_mwh) / one_way_efficiency
    return max(0.0, min(by_power, by_headroom))


def max_discharge_grid_mwh(
    soc_mwh: float,
    power_mw: float,
    one_way_efficiency: float,
    hours: float,
) -> float:
    """Largest grid-side energy (MWh) the battery can deliver this interval.

    Limited by the power rating and by available charge. Delivering grid
    energy ``e`` lowers the state of charge by ``e / one_way_efficiency``.
    """
    by_power = power_mw * hours
    by_soc = soc_mwh * one_way_efficiency
    return max(0.0, min(by_power, by_soc))
