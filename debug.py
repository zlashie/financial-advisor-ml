# debug.py
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.simulation.paydown_sim import Debt, Investments, Scenario, final_net_worth, simulate_projection
from src.labeling.optimal_split import argmax_optimal_split

def debug_no_debt_scenario():
    print("="*70)
    print("NO DEBT SCENARIO DEBUG")
    print("="*70)
    
    # Recreate the exact no debt scenario
    inv = Investments(
        equity_value=0.0, property_value=0.0,
        equity_return_rate=0.08, property_growth_rate=0.02
    )
    sc = Scenario(debts=[], investments=inv, horizon_years=5, monthly_extra=1000.0)
    
    print(f"Scenario details:")
    print(f"  Debts: {sc.debts} (length: {len(sc.debts)})")
    print(f"  Equity return: {sc.investments.equity_return_rate:.1%}")
    print(f"  Property return: {sc.investments.property_growth_rate:.1%}")
    print(f"  Monthly extra: ${sc.monthly_extra}")
    print(f"  Horizon: {sc.horizon_years} years")
    
    # Test different splits with detailed output
    test_splits = [0.0, 0.5, 1.0]
    print(f"\nTesting splits:")
    
    for split in test_splits:
        print(f"\n--- SPLIT {split} ---")
        proj = simulate_projection(sc, split)
        
        print(f"Year-by-year breakdown:")
        for i, (debt, assets, nw) in enumerate(zip(proj.debt_pos, proj.assets_pos, proj.net_worth)):
            print(f"  Year {i}: Debt=${debt:8.2f}, Assets=${assets:8.2f}, NW=${nw:8.2f}")
        
        final_nw = final_net_worth(sc, split)
        print(f"Final Net Worth: ${final_nw:,.2f}")
    
    # Find optimal split
    print(f"\n--- OPTIMIZATION ---")
    splits = np.linspace(0, 1, 11)
    all_results = []
    
    for split in splits:
        nw = final_net_worth(sc, split)
        all_results.append((split, nw))
        print(f"Split {split:.1f}: ${nw:8.2f}")
    
    s_opt, v_opt = argmax_optimal_split(sc, splits=splits)
    print(f"\nOptimal split: {s_opt}")
    print(f"Optimal value: ${v_opt:,.2f}")
    
    # Analysis
    print(f"\n--- ANALYSIS ---")
    print(f"Expected: Split 1.0 should be optimal (all to investing)")
    print(f"Actual: Split {s_opt} is optimal")
    
    if s_opt != 1.0:
        print(f"❌ BUG: With no debt and positive returns, split should be 1.0")
        
        # Check if all splits give the same result
        nw_values = [result[1] for result in all_results]
        if len(set(nw_values)) == 1:
            print(f"🔍 All splits give same result: ${nw_values[0]:,.2f}")
            print(f"🔍 This suggests the split parameter isn't affecting the simulation")
        else:
            print(f"🔍 Split values do differ, optimization may have other issues")
    else:
        print(f"✅ Correct: Split 1.0 is optimal")

def debug_simulation_internals():
    print("\n" + "="*70)
    print("SIMULATION INTERNALS DEBUG")
    print("="*70)
    
    # Test with a simple no-debt scenario
    inv = Investments(equity_value=0.0, property_value=0.0, 
                     equity_return_rate=0.10, property_growth_rate=0.0)
    sc = Scenario(debts=[], investments=inv, horizon_years=2, monthly_extra=1000.0)
    
    print("Testing split calculation logic:")
    
    for split in [0.0, 1.0]:
        print(f"\n--- Split {split} Internal Logic ---")
        
        # Manually trace the simulation logic
        base_to_investing_y = sc.monthly_extra * split * 12.0
        base_to_debt_y = sc.monthly_extra * (1.0 - split) * 12.0
        
        print(f"  base_to_investing_y: ${base_to_investing_y}")
        print(f"  base_to_debt_y: ${base_to_debt_y}")
        
        # Check debt logic
        total_debt_before = sum(max(0.0, d.amount) for d in sc.debts) if sc.debts else 0.0
        print(f"  total_debt_before: ${total_debt_before}")
        
        if total_debt_before > 0:
            to_investing_y = base_to_investing_y
            to_debt_y = base_to_debt_y
            print(f"  Using base amounts (debt exists)")
        else:
            to_investing_y = sc.monthly_extra * 12.0
            to_debt_y = 0.0
            print(f"  Using all-to-investing (no debt)")
        
        print(f"  Final to_investing_y: ${to_investing_y}")
        print(f"  Final to_debt_y: ${to_debt_y}")
        
        # Run actual simulation
        proj = simulate_projection(sc, split)
        print(f"  Actual final net worth: ${proj.net_worth[-1]:,.2f}")

if __name__ == "__main__":
    debug_no_debt_scenario()
    debug_simulation_internals()
