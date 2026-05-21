"""Joint energy + ancillary-services revenue-maximizing LP (post-RTC+B).

This is the post-RTC+B equivalent of the v1 perfect-foresight benchmark: it
assumes complete knowledge of every future energy and AS clearing price, and
solves a single LP that allocates the battery's power and state of charge
across real-time energy *and* the five AS products jointly. Not deployable;
it is the theoretical ceiling for a battery under RTC+B's co-optimized
real-time market.

LP formulation, per 15-min interval t (E = energy MWh, AS = capacity MW):

    variables   charge[t], discharge[t]                in [0, p_max * hours]
                regup[t], regdn[t], rrs[t],
                ecrs[t], nspin[t]                      in [0, p_max]
                soc[t]                                 in [0, capacity_mwh]

    maximize    sum_t  (discharge[t] - charge[t]) * energy_price[t]
                       + sum_p  as_vars[p][t] * mcpc[p][t]
                       - epsilon * (charge[t] + discharge[t])

    s.t.        discharge[t] / hours
                  + regup[t] + rrs[t] + ecrs[t] + nspin[t]  <=  p_max
                charge[t] / hours + regdn[t]                <=  p_max
                soc[t] = soc[t-1] + charge[t] * eff - discharge[t] / eff
                soc[t]              >=  sum_{discharge-AS} commit * duration / eff
                cap - soc[t]        >=  regdn[t] * duration_regdn * eff

The SoC-reservation constraints guarantee the battery could actually deliver
every AS commitment for its required duration if fully called -- the joint
constraint that makes co-optimization non-trivial.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pulp

from dispatcher_watts.battery.model import BatterySpec
from dispatcher_watts.data.schemas import (
    MCPC_PRODUCTS,
    RTM_INTERVAL_MINUTES,
    validate_mcpc_frame,
    validate_rtm_frame,
)

# Industry-typical delivery durations (hours) -- how long a resource must be
# able to sustain its AS commitment if fully called. Used as the SoC-reservation
# coefficient for each product. Conservative defaults; refine per product spec.
AS_DURATION_HOURS: dict[str, float] = {
    "regup": 1.0,
    "regdn": 1.0,
    "rrs": 0.25,
    "ecrs": 1.0,
    "nspin": 0.5,
}

# AS products that, if called, deliver energy OUT of the battery (require SoC).
_DISCHARGE_AS: tuple[str, ...] = ("regup", "rrs", "ecrs", "nspin")

# Tiny penalty per MWh of throughput to break revenue ties toward fewer cycles
# and to make simultaneous charge/discharge strictly dominated.
_THROUGHPUT_PENALTY: float = 1e-7


@dataclass
class CoOptimizationResult:
    """Outcome of a co-optimized run.

    `frame` has one row per interval with the original prices, each
    decision variable, the resulting SoC, and per-source revenue.
    `revenue_by_source` keys are: ``energy`` and the five AS products.
    """

    frame: pl.DataFrame
    revenue_by_source: dict[str, float]
    total_revenue: float
    spec: BatterySpec
    interval_minutes: int


def solve_co_optimization(
    prices: pl.DataFrame,
    mcpc: pl.DataFrame,
    spec: BatterySpec,
    interval_minutes: int = RTM_INTERVAL_MINUTES,
    initial_soc_mwh: float = 0.0,
    degradation_cost_per_mwh: float = 0.0,
) -> CoOptimizationResult:
    """Solve the joint energy + AS LP for the overlap of `prices` and `mcpc`.

    `prices` must match ``RTM_PRICE_SCHEMA``; `mcpc` must match ``MCPC_SCHEMA``.
    The two are inner-joined on ``interval_start``, so the result covers
    exactly the intervals present in both.

    `degradation_cost_per_mwh` adds a per-MWh-throughput penalty to the
    objective so the optimizer trades cycles against revenue. With the default
    of 0 the LP maximizes pure gross revenue -- useful as a ceiling but, on a
    real battery, that strategy can cycle the asset to death.
    """
    validate_rtm_frame(prices)
    validate_mcpc_frame(mcpc)
    aligned = prices.join(mcpc, on="interval_start", how="inner").sort("interval_start")
    if aligned.is_empty():
        raise ValueError(
            "no overlapping intervals between prices and mcpc; "
            "did you fetch both for the same period?"
        )

    n = aligned.height
    hours = interval_minutes / 60.0
    eff = spec.one_way_efficiency
    p_max = spec.power_mw
    cap = spec.capacity_mwh
    max_grid_mwh = p_max * hours

    price_energy = aligned["price"].to_list()
    mcpc_by_product: dict[str, list[float]] = {
        p: aligned[f"mcpc_{p}"].to_list() for p in MCPC_PRODUCTS
    }

    problem = pulp.LpProblem("co_optimization", pulp.LpMaximize)
    charge = [pulp.LpVariable(f"chg_{t}", lowBound=0, upBound=max_grid_mwh) for t in range(n)]
    discharge = [pulp.LpVariable(f"dis_{t}", lowBound=0, upBound=max_grid_mwh) for t in range(n)]
    as_vars: dict[str, list[pulp.LpVariable]] = {
        product: [pulp.LpVariable(f"as_{product}_{t}", lowBound=0, upBound=p_max) for t in range(n)]
        for product in MCPC_PRODUCTS
    }
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=0, upBound=cap) for t in range(n)]

    # Penalty per MWh of throughput: a tiny tie-breaker plus any real
    # degradation cost the caller passed in. The tie-breaker keeps simultaneous
    # charge/discharge strictly dominated; the degradation term turns the LP
    # from gross-revenue-maximizing into net-of-degradation-maximizing.
    throughput_penalty = _THROUGHPUT_PENALTY + degradation_cost_per_mwh
    problem += pulp.lpSum(
        (discharge[t] - charge[t]) * price_energy[t]
        + pulp.lpSum(as_vars[p][t] * mcpc_by_product[p][t] for p in MCPC_PRODUCTS)
        - throughput_penalty * (charge[t] + discharge[t])
        for t in range(n)
    )

    for t in range(n):
        previous = soc[t - 1] if t > 0 else initial_soc_mwh
        # State of charge dynamics.
        problem += soc[t] == previous + charge[t] * eff - discharge[t] / eff
        # Power-capacity sharing -- discharge direction.
        problem += discharge[t] / hours + pulp.lpSum(as_vars[p][t] for p in _DISCHARGE_AS) <= p_max
        # Power-capacity sharing -- charge direction (only regdn opposes charging).
        problem += charge[t] / hours + as_vars["regdn"][t] <= p_max
        # Must hold enough SoC to deliver every discharge-direction AS if fully called.
        problem += soc[t] >= (
            pulp.lpSum(as_vars[p][t] * AS_DURATION_HOURS[p] for p in _DISCHARGE_AS) / eff
        )
        # Must hold enough headroom to absorb a fully-called Reg-Down commitment.
        problem += cap - soc[t] >= as_vars["regdn"][t] * AS_DURATION_HOURS["regdn"] * eff

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"co-optimization LP did not solve: {pulp.LpStatus[status]}")

    # Collect decisions and revenue breakdown.
    charge_vals = [charge[t].value() or 0.0 for t in range(n)]
    discharge_vals = [discharge[t].value() or 0.0 for t in range(n)]
    soc_vals = [soc[t].value() or 0.0 for t in range(n)]
    as_vals: dict[str, list[float]] = {
        p: [as_vars[p][t].value() or 0.0 for t in range(n)] for p in MCPC_PRODUCTS
    }

    energy_revenue = [(discharge_vals[t] - charge_vals[t]) * price_energy[t] for t in range(n)]
    as_revenue: dict[str, list[float]] = {
        p: [as_vals[p][t] * mcpc_by_product[p][t] for t in range(n)] for p in MCPC_PRODUCTS
    }

    new_cols: dict[str, list[float]] = {
        "charge_mwh": charge_vals,
        "discharge_mwh": discharge_vals,
        "soc_mwh": soc_vals,
        "energy_revenue": energy_revenue,
    }
    for p in MCPC_PRODUCTS:
        new_cols[f"{p}_mw"] = as_vals[p]
        new_cols[f"{p}_revenue"] = as_revenue[p]

    frame = aligned.with_columns(
        **{name: pl.Series(name, values) for name, values in new_cols.items()}
    )

    revenue_by_source = {
        "energy": float(sum(energy_revenue)),
        **{p: float(sum(as_revenue[p])) for p in MCPC_PRODUCTS},
    }
    total_revenue = sum(revenue_by_source.values())

    return CoOptimizationResult(
        frame=frame,
        revenue_by_source=revenue_by_source,
        total_revenue=total_revenue,
        spec=spec,
        interval_minutes=interval_minutes,
    )
