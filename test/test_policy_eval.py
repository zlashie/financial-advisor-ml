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
#### Helper Functions
########################################
def simulate_policy_score(scenarios, predict_ratio_fn):
    """
    Returns mean policy efficiency over scenarios:
    efficiency = NW_pred / NW_opt (clipped to 1.0 if tiny FP drift).
    """
    effs = []
    for sc in scenarios:
        s_opt, _ = argmax_optimal_split(sc, splits=np.linspace(0,1,101))
        nw_opt = final_net_worth(sc, s_opt)
        s_hat = float(np.clip(predict_ratio_fn(sc), 0.0, 1.0))
        nw_hat = final_net_worth(sc, s_hat)

        eff = nw_hat / nw_opt if nw_opt != 0 else 1.0
        effs.append(min(max(eff, 0.0), 1.0 + 1e-9)) 
    return float(np.mean(effs))

def make_scenarios():
    out = []
    out.append(Scenario(
        debts=[], 
        investments=Investments(0.0, 0.0, 0.08, 0.02),
        horizon_years=5, monthly_extra=1000.0
    ))
    out.append(Scenario(
        debts=[Debt(6000.0, 0.30), Debt(2000.0, 0.15)],
        investments=Investments(0.0, 0.0, 0.00, 0.00),
        horizon_years=5, monthly_extra=500.0
    ))
    out.append(Scenario(
        debts=[Debt(4000.0, 0.08)],
        investments=Investments(5000.0, 100000.0, 0.07, 0.02),
        horizon_years=10, monthly_extra=1000.0
    ))
    out.append(Scenario(
        debts=[Debt(3000.0, 0.04)],
        investments=Investments(0.0, 0.0, 0.09, 0.02),
        horizon_years=20, monthly_extra=800.0
    ))
    return out

def oracle_predict(sc: Scenario):
    s_opt, _ = argmax_optimal_split(sc, splits=np.linspace(0,1,101))
    return s_opt

def heuristic_predict(sc: Scenario):
    max_apr = max([d.apr for d in sc.debts], default=0.0)
    return 0.2 if max_apr >= 0.10 else 0.6

########################################
#### Tests
########################################
def test_policy_eval_oracle_hits_100_percent():
    scenarios = make_scenarios()
    score = simulate_policy_score(scenarios, oracle_predict)
    assert score == pytest.approx(1.0, abs=1e-9)

def test_policy_eval_heuristic_reasonable():
    scenarios = make_scenarios()
    score = simulate_policy_score(scenarios, heuristic_predict)
    assert 0.55 <= score <= 1.0  

def test_policy_eval_improved_heuristic():
    """Test a more sophisticated heuristic"""
    scenarios = make_scenarios()
    
    def improved_heuristic(sc: Scenario):
        if not sc.debts:
            return 1.0  
        
        max_apr = max(d.apr for d in sc.debts)
        expected_return = sc.investments.equity_return_rate
        
        margin = 0.02  
        if max_apr > expected_return + margin:
            return 0.1  
        elif max_apr < expected_return - margin:
            return 0.9 
        else:
            return 0.5  
    
    score = simulate_policy_score(scenarios, improved_heuristic)
    print(f"Improved heuristic score: {score:.3f}")
    assert score >= 0.70  

def test_policy_eval_realistic_scenarios():
    """Test with more realistic scenarios"""
    scenarios = [

        Scenario(
            debts=[Debt(5000.0, 0.25)], 
            investments=Investments(0.0, 0.0, 0.07, 0.02),  
            horizon_years=5, monthly_extra=500.0
        ),
 
        Scenario(
            debts=[Debt(10000.0, 0.03)], 
            investments=Investments(0.0, 0.0, 0.08, 0.02), 
            horizon_years=10, monthly_extra=1000.0
        ),

        Scenario(
            debts=[Debt(8000.0, 0.06)],  
            investments=Investments(0.0, 0.0, 0.065, 0.02),  
            horizon_years=15, monthly_extra=800.0
        )
    ]
    
    def smart_heuristic(sc: Scenario):
        if not sc.debts:
            return 1.0
        
        max_apr = max(d.apr for d in sc.debts)
        expected_return = sc.investments.equity_return_rate
        

        diff = expected_return - max_apr
        if diff > 0.03: 
            return 0.8
        elif diff < -0.03: 
            return 0.2
        else: 
            long_term_bonus = min(sc.horizon_years / 20.0, 0.3)
            base_split = 0.5 + diff * 5  
            return np.clip(base_split + long_term_bonus, 0.0, 1.0)
    
    score = simulate_policy_score(scenarios, smart_heuristic)
    print(f"Smart heuristic score: {score:.3f}")
    assert score >= 0.65  
