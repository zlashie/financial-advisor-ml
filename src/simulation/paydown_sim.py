########################################
#### Dependencies
########################################
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from copy import deepcopy
from datetime import date

########################################
#### Data Classes
########################################
@dataclass
class Debt:
    amount: float   # current principal
    apr: float      # yearly nominal rate as decimal (e.g., 0.199)

@dataclass
class Investments:
    equity_value: float            # starting equity/index value
    property_value: float          # starting property market value
    equity_return_rate: float      # yearly expected return (e.g., 0.08)
    property_growth_rate: float    # yearly expected growth (e.g., 0.02)

@dataclass
class Scenario:
    debts: List[Debt]
    investments: Investments
    horizon_years: int
    monthly_extra: float           # surplus funds per month (split by model)

@dataclass
class Projection:
    years: List[int]        # calendar years, starting from current year
    assets_pos: List[float] # total assets each year (positive)
    debt_pos: List[float]   # total debt each year (positive)
    debt_neg: List[float]   # total debt as negative (for plotting)
    net_worth: List[float]  # assets - debt

########################################
#### Definitions
########################################
def _apply_yearly_debt_interest_avalanche(debts: List[Debt], annual_extra_payment: float) -> None:
    """
    Apply yearly interest to each debt, then pay extra using avalanche.
    """
    #### 1. Apply interest first (yearly compounding) ####
    for d in debts:
        if d.amount > 0:
            d.amount *= (1.0 + max(d.apr, -1.0))

    #### 2. Skip minimum payments for pure optimization testing ####
    
    #### 3. Avalanche: pay extra to highest APR first ####
    debts.sort(key=lambda d: d.apr, reverse=True)
    
    remaining_extra = annual_extra_payment
    for d in debts:
        if remaining_extra <= 0:
            break
        if d.amount <= 0:
            continue
        extra_payment = min(d.amount, remaining_extra)
        d.amount -= extra_payment
        remaining_extra -= extra_payment

def simulate_projection(
    scenario: Scenario,
    invest_fraction: float
) -> Projection:
    """
    Simulate YEARLY steps for given scenario and investing fraction s ∈ [0,1].
    - Yearly compounding for debts (APR once per year).
    - Avalanche payoff: all extra goes to the highest APR bucket until cleared.
    - Extra split applies ONLY to the monthly surplus (not minimums).
    - Equity contributions: invest_share*monthly_extra * 12 per year, then grow by equity_return_rate.
    - Property grows yearly by property_growth_rate (no contributions).
    Outputs per-year snapshots INCLUDING year 0.
    """
    s = max(0.0, min(1.0, invest_fraction))
    
    base_to_investing_m = scenario.monthly_extra * s
    base_to_debt_m = scenario.monthly_extra * (1.0 - s)
    base_to_investing_y = base_to_investing_m * 12.0
    base_to_debt_y = base_to_debt_m * 12.0

    debts = deepcopy(scenario.debts)
    debts.sort(key=lambda d: d.apr, reverse=True)

    equity = scenario.investments.equity_value
    prop = scenario.investments.property_value
    r_e = scenario.investments.equity_return_rate
    r_p = scenario.investments.property_growth_rate

    current_year = date.today().year
    years, assets_pos, debt_pos, debt_neg, net_worth = [], [], [], [], []

    def snapshot():
        total_debt = sum(max(0.0, d.amount) for d in debts)
        total_assets = max(0.0, equity) + max(0.0, prop)
        years.append(current_year + len(years))
        assets_pos.append(total_assets)
        debt_pos.append(total_debt)
        debt_neg.append(-total_debt)
        net_worth.append(total_assets - total_debt)

    snapshot()

    #### simulate each full year ####
    for _ in range(scenario.horizon_years):
        total_debt_before = sum(max(0.0, d.amount) for d in debts) if debts else 0.0

        if total_debt_before > 0:
            to_investing_y = base_to_investing_y
            to_debt_y = base_to_debt_y
        else:
            to_investing_y = base_to_investing_y  # Only invest the investing portion
            to_debt_y = 0.0

        #### INVESTMENTS: add contributions for the year, then apply YEARLY growth ####
        equity = (equity + to_investing_y) * (1.0 + r_e)
        prop = prop * (1.0 + r_p)

        #### DEBT: Only process if there are debts with positive amounts ####
        if debts and total_debt_before > 0:
            _apply_yearly_debt_interest_avalanche(debts, to_debt_y)

        snapshot()

    return Projection(
        years=years, assets_pos=assets_pos, debt_pos=debt_pos, debt_neg=debt_neg, net_worth=net_worth
    )

def final_net_worth(scenario: Scenario, invest_fraction: float) -> float:
    proj = simulate_projection(scenario, invest_fraction)
    return proj.net_worth[-1]
