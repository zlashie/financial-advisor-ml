########################################
#### Dependencies
########################################
from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
import numpy as np

from src.simulation.paydown_sim import Scenario, Debt, Investments, final_net_worth

########################################
#### Definitions
########################################
def argmax_optimal_split(
    scenario: Scenario,
    splits: Iterable[float] = np.linspace(0.0, 1.0, 101)
) -> Tuple[float, float]:
    """
    Returns (s_opt, networth_at_opt) where s_opt maximizes final net worth.
    """
    best_s, best_v = 0.0, -1e30
    for s in splits:
        v = final_net_worth(scenario, float(s))
        if v > best_v:
            best_v, best_s = v, float(s)
    return best_s, best_v

def build_training_example(scenario: Scenario, feature_vector: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Given a scenario + its computed feature vector, compute the label y* = s_opt.
    """
    s_opt, _ = argmax_optimal_split(scenario)
    return feature_vector, s_opt
