########################################
#### Dependencies
########################################
import math
from datetime import date
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.simulation.paydown_sim import (
    Debt, Investments, Scenario, simulate_projection, final_net_worth
)

########################################
#### Tests 
########################################
def make_scenario(
    debts=None,
    equity_value=0.0,
    property_value=0.0,
    equity_return_rate=0.08,
    property_growth_rate=0.02,
    horizon_years=5,
    monthly_extra=1000.0,
):
    debts = debts or [Debt(amount=1000.0, apr=0.10)]
    inv = Investments(
        equity_value=equity_value,
        property_value=property_value,
        equity_return_rate=equity_return_rate,
        property_growth_rate=property_growth_rate,
    )
    return Scenario(
        debts=debts,
        investments=inv,
        horizon_years=horizon_years,
        monthly_extra=monthly_extra,
    )

def test_projection_shapes_and_years():
    sc = make_scenario(horizon_years=3)
    proj = simulate_projection(sc, invest_fraction=0.5)

    assert len(proj.years) == 4
    assert len(proj.assets_pos) == 4
    assert len(proj.debt_pos) == 4
    assert len(proj.debt_neg) == 4
    assert len(proj.net_worth) == 4

    assert proj.years[0] == date.today().year
    assert proj.years[-1] == date.today().year + 3

def test_debt_neg_is_negative_of_debt_pos():
    sc = make_scenario(horizon_years=2)
    proj = simulate_projection(sc, invest_fraction=0.3)
    for dp, dn in zip(proj.debt_pos, proj.debt_neg):
        assert pytest.approx(-dp, rel=1e-12, abs=1e-12) == dn

def test_yearly_compounding_and_contributions_basic():

    sc = make_scenario(
        debts=[Debt(amount=1000.0, apr=0.10)],
        equity_value=0.0,
        property_value=0.0,
        equity_return_rate=0.08,
        property_growth_rate=0.00,
        horizon_years=1,
        monthly_extra=120.0,  
    )

    proj_debt = simulate_projection(sc, invest_fraction=0.0)
    assert proj_debt.debt_pos[-1] == pytest.approx(0.0, abs=1e-9)
    assert proj_debt.assets_pos[-1] == pytest.approx(0.0, abs=1e-9)
    assert proj_debt.net_worth[-1] == pytest.approx(0.0, abs=1e-9)

    proj_inv = simulate_projection(sc, invest_fraction=1.0)
    assert proj_inv.debt_pos[-1] == pytest.approx(1100.0, abs=1e-9)
    assert proj_inv.assets_pos[-1] == pytest.approx(1440.0 * 1.08, abs=1e-9)
    assert proj_inv.net_worth[-1] == pytest.approx(1440.0 * 1.08 - 1100.0, abs=1e-9)

def test_avalanche_priority_higher_apr_first():

    sc = make_scenario(
        debts=[Debt(1000.0, 0.20), Debt(1000.0, 0.05)],
        horizon_years=1,
        monthly_extra=100.0,
        equity_value=0.0,
        property_value=0.0,
        equity_return_rate=0.00,
        property_growth_rate=0.00,
    )
    proj = simulate_projection(sc, invest_fraction=0.0)

    assert proj.debt_pos[-1] == pytest.approx(1050.0, abs=1e-9)

def test_property_growth_applied_yearly_no_contrib():
    sc = make_scenario(
        debts=[Debt(0.0, 0.0)],
        equity_value=0.0,
        property_value=100_000.0,
        property_growth_rate=0.02,
        equity_return_rate=0.0,
        horizon_years=2,
        monthly_extra=0.0
    )
    proj = simulate_projection(sc, invest_fraction=0.0)

    assert proj.assets_pos[0] == pytest.approx(100_000.0, abs=1e-6)
    assert proj.assets_pos[1] == pytest.approx(102_000.0, abs=1e-6)
    assert proj.assets_pos[2] == pytest.approx(104_040.0, abs=1e-6)
    assert all(d == 0.0 for d in proj.debt_pos)

def test_final_net_worth_matches_projection_last_point():
    sc = make_scenario(horizon_years=4, monthly_extra=500.0)
    s = 0.42
    proj = simulate_projection(sc, invest_fraction=s)
    assert final_net_worth(sc, s) == pytest.approx(proj.net_worth[-1], rel=1e-12)
