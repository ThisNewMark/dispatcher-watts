"""Multi-participant live dispatch competition.

The platform layer on top of the single-player simulator: many participants
trade independent batteries (identical spec) against one shared live ERCOT tape,
submit decisions through a queue, and are ranked on time-windowed leaderboards.
Strategy logic lives entirely on the participant's side; this package owns the
canonical market, the physics, and the scoring.
"""

from __future__ import annotations

from dispatcher_watts.battery.model import BatterySpec

# Every participant trades this exact battery, so revenue is comparable. The
# competition's canonical asset (4-hour, the realistic v2 default at HB_HOUSTON).
COMPETITION_SPEC: BatterySpec = BatterySpec(
    capacity_mwh=10.0, power_mw=2.5, round_trip_efficiency=0.87
)
COMPETITION_HUB: str = "HB_HOUSTON"

# Flat battery-wear fee charged on every MWh of grid energy moved (charge +
# discharge). Subtracted from gross revenue to get the score, so thin-spread
# over-cycling -- profitable on paper but a money-loser on a real battery that
# wears out -- doesn't win. Standby (AS) moves no energy, so it pays no wear fee.
# Tunable; a modest per-MWh-throughput default, not a precise real-world figure.
COMPETITION_DEGRADATION_COST_PER_MWH: float = 5.0

# Expected real-time deployment fraction per AS product: the share of a
# committed MW that is actually *called* each interval, moving the battery's
# charge. Regulation (RegUp/RegDn) is deployed almost continuously; reserves
# (RRS/ECRS/NSpin) rarely. Tunable approximations -- they turn standby from free
# money into a commitment that spends charge (and so incurs the wear fee on the
# recharge), which is the real-world tension.
COMPETITION_AS_DEPLOYMENT_FRACTION: dict[str, float] = {
    "regup": 0.15,
    "regdn": 0.15,
    "rrs": 0.03,
    "ecrs": 0.03,
    "nspin": 0.01,
}

__all__ = [
    "COMPETITION_AS_DEPLOYMENT_FRACTION",
    "COMPETITION_DEGRADATION_COST_PER_MWH",
    "COMPETITION_HUB",
    "COMPETITION_SPEC",
]
