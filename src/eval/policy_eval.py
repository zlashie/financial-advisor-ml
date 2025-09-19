########################################
#### Dependencies
########################################
from __future__ import annotations
from typing import Iterable, Callable
import numpy as np
from src.labeling.optimal_split import argmax_optimal_split
from src.simulation.paydown_sim import final_net_worth, Scenario

########################################
#### Definitions
########################################
def policy_efficiency(
    scenarios: Iterable[Scenario],
    predict_ratio_fn: Callable[[Scenario], float],
    n_grid: int = 101
) -> float:
    effs = []
    grid = np.linspace(0.0, 1.0, n_grid)
    for sc in scenarios:
        s_opt, _ = argmax_optimal_split(sc, splits=grid)
        nw_opt = final_net_worth(sc, s_opt)
        s_hat = float(np.clip(predict_ratio_fn(sc), 0.0, 1.0))
        nw_hat = final_net_worth(sc, s_hat)
        eff = (nw_hat / nw_opt) if nw_opt != 0 else 1.0
        effs.append(min(max(eff, 0.0), 1.0 + 1e-9))
    return float(np.mean(effs))
