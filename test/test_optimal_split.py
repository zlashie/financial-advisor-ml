########################################
#### Dependencies
########################################
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.simulation.paydown_sim import Debt, Investments, Scenario, final_net_worth
from src.labeling.optimal_split import argmax_optimal_split

########################################
#### Tests 
########################################

def test_debug_optimal_split():
    """Debug test to see what's actually happening"""
    import numpy as np
    from src.simulation.paydown_sim import Debt, Investments, Scenario
    from src.labeling.optimal_split import argmax_optimal_split
    
    inv = Investments(
        equity_value=0.0, property_value=0.0,
        equity_return_rate=0.00, property_growth_rate=0.00
    )
    debts = [Debt(amount=5000.0, apr=0.40)]  
    sc = Scenario(debts=debts, investments=inv, horizon_years=5, monthly_extra=500.0)
    
    splits = np.linspace(0, 1, 11) 
    s_opt, v_opt = argmax_optimal_split(sc, splits=splits)
    
    print(f"Optimal split: {s_opt}")
    print(f"Optimal value: {v_opt}")
    
    for s in splits:
        from src.simulation.paydown_sim import final_net_worth
        nw = final_net_worth(sc, s)
        print(f"Split {s:.1f}: Net Worth = {nw:.2f}")

def scenario_no_debt_positive_returns():
    inv = Investments(
        equity_value=0.0, property_value=0.0,
        equity_return_rate=0.08, property_growth_rate=0.02
    )
    return Scenario(debts=[], investments=inv, horizon_years=5, monthly_extra=1000.0)

def scenario_high_apr_low_returns():
    inv = Investments(
        equity_value=0.0, property_value=0.0,
        equity_return_rate=0.00, property_growth_rate=0.00
    )
    debts = [Debt(amount=50000.0, apr=0.40)]  
    return Scenario(debts=debts, investments=inv, horizon_years=5, monthly_extra=500.0)

def scenario_negative_equity_return():
    inv = Investments(
        equity_value=0.0, property_value=0.0,
        equity_return_rate=-0.05, property_growth_rate=0.00
    )
    debts = [Debt(amount=30000.0, apr=0.05)]  
    return Scenario(debts=debts, investments=inv, horizon_years=3, monthly_extra=300.0)

def test_optimal_split_no_debt_prefers_investing():
    sc = scenario_no_debt_positive_returns()
    s_opt, v_opt = argmax_optimal_split(sc, splits=np.linspace(0,1,51))
    assert s_opt == pytest.approx(1.0, abs=1e-6)
    assert v_opt >= final_net_worth(sc, 0.0)

def test_optimal_split_high_apr_prefers_debt():
    sc = scenario_high_apr_low_returns()
    s_opt, _ = argmax_optimal_split(sc, splits=np.linspace(0,1,51))
    assert s_opt <= 0.1  

def test_optimal_split_negative_equity_return_zero_investing():
    sc = scenario_negative_equity_return()
    s_opt, _ = argmax_optimal_split(sc, splits=np.linspace(0,1,51))
    assert s_opt == pytest.approx(0.0, abs=1e-6)

def test_optimal_split_monotonic_tendency_with_horizon():
    inv = Investments(equity_value=0.0, property_value=0.0, equity_return_rate=0.06, property_growth_rate=0.00)
    debts = [Debt(amount=4000.0, apr=0.06)]
    sc_short = Scenario(debts=debts, investments=inv, horizon_years=1, monthly_extra=500.0)
    sc_long  = Scenario(debts=debts, investments=inv, horizon_years=10, monthly_extra=500.0)

    s_short, _ = argmax_optimal_split(sc_short, splits=np.linspace(0,1,51))
    s_long, _  = argmax_optimal_split(sc_long,  splits=np.linspace(0,1,51))

    assert s_long + 1e-6 >= s_short
