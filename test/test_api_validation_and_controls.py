########################################
#### Dependencies
########################################
import pytest
from fastapi.testclient import TestClient
from src.services.inference_api import app

########################################
#### Tests
########################################
client = TestClient(app)

def base_payload():
    return {
        "debts": [{"amount": 12000.0, "apr": 0.199}],
        "investments": {
            "equity_value": 15000.0,
            "property_value": 300000.0,
            "property_growth_rate": 0.02,
            "equity_return_rate": 0.08
        },
        "horizon_years": 10,
        "monthly_extra": 1000.0
    }

def test_validation_rejects_oob_rates_and_horizon():
    bad = base_payload()
    bad["debts"][0]["apr"] = 1.2 
    assert client.post("/inference/recommend", json=bad).status_code == 422

    bad = base_payload()
    bad["horizon_years"] = 0 
    assert client.post("/inference/recommend", json=bad).status_code == 422

    bad = base_payload()
    bad["investments"]["equity_return_rate"] = -0.9  # 
    assert client.post("/inference/recommend", json=bad).status_code == 422

def test_ratio_is_clipped_and_allocations_sum_to_monthly_extra():
    payload = base_payload()
    payload["debug_force_ratio"] = 1.5 
    r = client.post("/inference/recommend", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["allocation"]["ratio"] == pytest.approx(1.0, abs=1e-12)
    total = data["allocation"]["to_debt"] + data["allocation"]["to_investing"]
    assert total == pytest.approx(payload["monthly_extra"], abs=1e-9)

def test_zero_debts_is_ok():
    payload = base_payload()
    payload["debts"] = []
    r = client.post("/inference/recommend", json=payload)
    assert r.status_code == 200
    proj = r.json()["projections"]
    assert all(abs(x) < 1e-9 for x in proj["debt_neg"])
