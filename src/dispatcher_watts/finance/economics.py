"""Turn a backtest's gross revenue into the numbers an investor cares about.

The dispatch model gives an idealized gross revenue. Real cash flow is
smaller: there are operational haircuts (availability, QSE fees), a real cost
per cycle (degradation), and fixed annual costs. And the question that
actually matters -- *does it pencil?* -- needs the capex and the federal
Investment Tax Credit too.

All assumptions are configurable; defaults are industry-typical-but-illustrative
and should be treated as a starting point, not a quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryEconomics:
    """Cost and capital assumptions for the project-finance layer.

    Defaults are industry-typical for a utility-scale battery circa 2025/2026.
    They are illustrative -- real projects vary substantially.
    """

    # Capital.
    capex_per_kwh: float = 350.0  # all-in install cost ($/kWh of energy capacity)
    itc_rate: float = 0.30  # federal Investment Tax Credit on the capex

    # Operating haircuts.
    availability: float = 0.95  # fraction of intervals the battery is actually online
    qse_fee_rate: float = 0.02  # Qualified Scheduling Entity service fee (fraction of revenue)
    degradation_cost_per_mwh: float = 50.0  # cycle wear cost per MWh of energy throughput

    # Annual fixed costs (insurance, lease, maintenance contracts, monitoring, ...).
    fixed_om_per_kw_year: float = 30.0  # $/kW of power rating, per year


@dataclass(frozen=True)
class FinanceSummary:
    """Reported numbers after applying costs and the project-finance wrapper."""

    capacity_mwh: float
    power_mw: float
    days_in_window: int

    # Gross-to-net waterfall (period totals, in $).
    gross_revenue: float
    availability_haircut: float
    degradation_cost: float
    qse_fee: float
    net_operating_revenue: float  # in the window

    # Annualized cash flow ($/year).
    net_operating_revenue_annualized: float
    fixed_annual_costs: float
    net_free_cash_flow_annualized: float

    # Capital.
    capex_gross: float
    capex_after_itc: float

    # Headline investment metrics.
    simple_payback_years: float  # capex / annual FCF, infinity if FCF <= 0
    cash_on_cash_return_pct: float  # annual FCF / capex-after-ITC, %


def summarize_finance(
    *,
    gross_revenue: float,
    throughput_mwh: float,
    capacity_mwh: float,
    power_mw: float,
    days_in_window: int,
    econ: BatteryEconomics | None = None,
) -> FinanceSummary:
    """Apply operating costs, annualize, and compute investment metrics.

    `throughput_mwh` is total grid-side energy moved through the battery in
    the window (sum of charge + discharge MWh, or equivalently 2 x discharge
    for a battery that finishes near its starting state of charge).
    """
    econ = econ or BatteryEconomics()
    if days_in_window <= 0:
        raise ValueError("days_in_window must be positive")

    # Operating waterfall on the period gross revenue.
    after_availability = gross_revenue * econ.availability
    availability_haircut = gross_revenue - after_availability
    degradation_cost = throughput_mwh * econ.degradation_cost_per_mwh
    after_degradation = after_availability - degradation_cost
    qse_fee = max(after_degradation, 0.0) * econ.qse_fee_rate
    net_operating_revenue = after_degradation - qse_fee

    # Annualize the period number, then deduct fixed annual costs.
    annualization = 365.0 / days_in_window
    net_operating_annualized = net_operating_revenue * annualization
    fixed_annual = econ.fixed_om_per_kw_year * power_mw * 1000.0  # $/kW * MW * 1000
    net_free_cash_flow_annualized = net_operating_annualized - fixed_annual

    # Capital, with the ITC applied.
    capex_gross = econ.capex_per_kwh * capacity_mwh * 1000.0  # $/kWh * MWh * 1000
    capex_after_itc = capex_gross * (1.0 - econ.itc_rate)

    # Simple investment metrics.
    simple_payback = (
        capex_after_itc / net_free_cash_flow_annualized
        if net_free_cash_flow_annualized > 0
        else math.inf
    )
    cash_on_cash = (
        net_free_cash_flow_annualized / capex_after_itc * 100.0 if capex_after_itc > 0 else 0.0
    )

    return FinanceSummary(
        capacity_mwh=capacity_mwh,
        power_mw=power_mw,
        days_in_window=days_in_window,
        gross_revenue=gross_revenue,
        availability_haircut=availability_haircut,
        degradation_cost=degradation_cost,
        qse_fee=qse_fee,
        net_operating_revenue=net_operating_revenue,
        net_operating_revenue_annualized=net_operating_annualized,
        fixed_annual_costs=fixed_annual,
        net_free_cash_flow_annualized=net_free_cash_flow_annualized,
        capex_gross=capex_gross,
        capex_after_itc=capex_after_itc,
        simple_payback_years=simple_payback,
        cash_on_cash_return_pct=cash_on_cash,
    )
