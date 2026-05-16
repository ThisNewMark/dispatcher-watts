"""Battery model: state of charge, charge/discharge, constraint enforcement."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dispatcher_watts.battery.constraints import (
    max_charge_grid_mwh,
    max_discharge_grid_mwh,
)


@dataclass(frozen=True)
class BatterySpec:
    """Physical specification of a battery.

    Defaults are the v1 reference asset: a 1 MWh / 0.5 MW (2-hour duration)
    lithium battery at 87% round-trip efficiency.
    """

    capacity_mwh: float = 1.0
    power_mw: float = 0.5
    round_trip_efficiency: float = 0.87
    # Capacity lost per equivalent full cycle (0.005%). Tracked for reporting
    # only -- v1 does not shrink the active capacity (see CLAUDE.md).
    degradation_per_cycle: float = 0.00005

    def __post_init__(self) -> None:
        if self.capacity_mwh <= 0:
            raise ValueError("capacity_mwh must be positive")
        if self.power_mw <= 0:
            raise ValueError("power_mw must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")

    @property
    def one_way_efficiency(self) -> float:
        """Per-leg efficiency: sqrt(RTE), applied on both charge and discharge."""
        return math.sqrt(self.round_trip_efficiency)


class Battery:
    """A battery being dispatched: tracks state of charge and energy throughput.

    All charge/discharge amounts are *grid-side* energy (measured at the
    interconnection). Internally the state of charge moves by the grid energy
    scaled by the one-way efficiency -- see `constraints.py`.

    Charge and discharge requests are clamped to the feasible range rather than
    rejected, so a strategy can safely ask for more than is possible.
    """

    def __init__(self, spec: BatterySpec, initial_soc_mwh: float = 0.0) -> None:
        if not 0 <= initial_soc_mwh <= spec.capacity_mwh:
            raise ValueError("initial_soc_mwh must be within [0, capacity_mwh]")
        self.spec = spec
        self.soc_mwh = initial_soc_mwh
        self.energy_charged_mwh = 0.0  # cumulative grid-side energy in
        self.energy_discharged_mwh = 0.0  # cumulative grid-side energy out
        self._throughput_internal_mwh = 0.0  # cumulative state-of-charge drawn

    @property
    def soc_fraction(self) -> float:
        """State of charge as a fraction of capacity, in [0, 1]."""
        return self.soc_mwh / self.spec.capacity_mwh

    @property
    def equivalent_full_cycles(self) -> float:
        """Total discharge throughput expressed in full-capacity cycles."""
        return self._throughput_internal_mwh / self.spec.capacity_mwh

    @property
    def capacity_lost_mwh(self) -> float:
        """Cumulative degradation so far (reported, not applied to capacity)."""
        return (
            self.equivalent_full_cycles * self.spec.degradation_per_cycle * self.spec.capacity_mwh
        )

    def max_charge_mwh(self, hours: float) -> float:
        """Feasible grid-side charge energy for an interval of `hours` hours."""
        return max_charge_grid_mwh(
            self.soc_mwh,
            self.spec.capacity_mwh,
            self.spec.power_mw,
            self.spec.one_way_efficiency,
            hours,
        )

    def max_discharge_mwh(self, hours: float) -> float:
        """Feasible grid-side discharge energy for an interval of `hours` hours."""
        return max_discharge_grid_mwh(
            self.soc_mwh,
            self.spec.power_mw,
            self.spec.one_way_efficiency,
            hours,
        )

    def charge(self, grid_mwh: float, hours: float) -> float:
        """Charge by up to `grid_mwh` of grid energy; return the amount accepted.

        The request is clamped to the feasible range, so over-charging is safe.
        """
        if grid_mwh < 0:
            raise ValueError("charge grid_mwh must be non-negative")
        accepted = min(grid_mwh, self.max_charge_mwh(hours))
        stored = accepted * self.spec.one_way_efficiency
        self.soc_mwh = min(self.spec.capacity_mwh, self.soc_mwh + stored)
        self.energy_charged_mwh += accepted
        return accepted

    def discharge(self, grid_mwh: float, hours: float) -> float:
        """Discharge up to `grid_mwh` of grid energy; return the amount delivered.

        The request is clamped to the feasible range, so over-discharging is safe.
        """
        if grid_mwh < 0:
            raise ValueError("discharge grid_mwh must be non-negative")
        delivered = min(grid_mwh, self.max_discharge_mwh(hours))
        drawn = delivered / self.spec.one_way_efficiency
        self.soc_mwh = max(0.0, self.soc_mwh - drawn)
        self.energy_discharged_mwh += delivered
        self._throughput_internal_mwh += drawn
        return delivered
