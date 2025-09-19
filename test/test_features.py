########################################
#### Dependencies
########################################

import math
import pytest

from src.simulation.paydown_sim import Debt, Investments, Scenario
from src.features.build_features import build_features

########################################
#### Tests
########################################
def make_scenario(
    debts=None,
    equity_value=10_000.0,
    property_value=300_000.0,
    equity_return_rate=0.08,
    property_growth_rate=0.02,
    horizon_years=10,
    monthly_extra=1000.0
):
    debts = debts or [Debt(12_000.0, 0.199), Debt(8_000.0, 0.06)]
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

def test_features_basic_presence_and_types():
    sc = make_scenario()
    f = build_features(sc)
    required = {
        "monthly_extra",
        "horizon_years",
        "equity_value",
        "property_value",
        "equity_return_rate",
        "property_growth_rate",
        "total_debt",
        "max_apr",
        "weighted_apr",
        "top_apr_share",
        "num_debts",
        "debt_to_assets",
        "surplus_to_debt",
    }
    assert required.issubset(f.keys())
    assert all(isinstance(f[k], float) for k in required)

def test_debt_stats_are_correct():
    sc = make_scenario(debts=[Debt(1000.0, 0.20), Debt(3000.0, 0.10), Debt(6000.0, 0.10)])
    f = build_features(sc)
    assert f["total_debt"] == pytest.approx(10000.0, abs=1e-9)
    assert f["max_apr"] == pytest.approx(0.20, abs=1e-12)
    assert f["weighted_apr"] == pytest.approx(0.11, abs=1e-12)
    assert f["top_apr_share"] == pytest.approx(0.1, abs=1e-12)
    assert f["num_debts"] == pytest.approx(3.0, abs=1e-12)

def test_ratios_handle_edges():
    sc1 = make_scenario(equity_value=0.0, property_value=0.0)
    f1 = build_features(sc1)
    assert math.isinf(f1["debt_to_assets"]) and f1["debt_to_assets"] > 0

    sc2 = make_scenario(debts=[Debt(0.0, 0.0)])
    f2 = build_features(sc2)
    assert math.isinf(f2["surplus_to_debt"]) and f2["surplus_to_debt"] > 0

def test_monotonic_intuition_checks():
    sc_low = make_scenario(monthly_extra=200.0)
    sc_high = make_scenario(monthly_extra=2000.0)
    f_low = build_features(sc_low)
    f_high = build_features(sc_high)
    assert f_high["surplus_to_debt"] > f_low["surplus_to_debt"]

    sc_h1 = make_scenario(horizon_years=5)
    sc_h2 = make_scenario(horizon_years=20)
    assert build_features(sc_h2)["horizon_years"] > build_features(sc_h1)["horizon_years"]
