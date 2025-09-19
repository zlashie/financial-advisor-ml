########################################
#### Dependencies
########################################
from __future__ import annotations
from typing import Dict, List
import math

from src.simulation.paydown_sim import Scenario, Debt

########################################
#### Definitions
########################################
def _debt_features(debts: List[Debt]) -> Dict[str, float]:
    total_debt = sum(d.amount for d in debts)
    if total_debt <= 0:
        return dict(
            total_debt=0.0, max_apr=0.0, weighted_apr=0.0, top_apr_share=0.0, num_debts=0.0
        )
    max_apr = max((d.apr for d in debts), default=0.0)
    weighted_apr = sum(d.amount * d.apr for d in debts) / total_debt

    top_apr = max_apr
    top_amt = sum(d.amount for d in debts if d.apr == top_apr)
    top_apr_share = top_amt / total_debt
    return dict(
        total_debt=total_debt,
        max_apr=max_apr,
        weighted_apr=weighted_apr,
        top_apr_share=top_apr_share,
        num_debts=float(len(debts)),
    )

def build_features(scenario: Scenario) -> Dict[str, float]:
    """
    Produces a flat numeric feature dict suitable for LR/RF/NN.
    """
    f = {}
    f.update(_debt_features(scenario.debts))
    inv = scenario.investments
    f.update(dict(
        monthly_extra=scenario.monthly_extra,
        horizon_years=float(scenario.horizon_years),
        equity_value=inv.equity_value,
        property_value=inv.property_value,
        equity_return_rate=inv.equity_return_rate,
        property_growth_rate=inv.property_growth_rate,

        debt_to_assets = (f["total_debt"] / max(1e-9, (inv.equity_value + inv.property_value))) if (inv.equity_value + inv.property_value) > 0 else float("inf"),
        surplus_to_debt = (scenario.monthly_extra * 12.0) / max(1e-9, f["total_debt"]) if f["total_debt"] > 0 else math.inf,
    ))
    return f
