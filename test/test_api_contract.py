########################################
#### Dependencies
########################################
import pytest
from datetime import date
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

def test_happy_path_contract_and_shapes():
    r = client.post("/inference/recommend", json=base_payload())
    assert r.status_code == 200
    data = r.json()

    assert set(data.keys()) == {"allocation", "projections", "explain"}

    alloc = data["allocation"]
    assert set(alloc.keys()) == {"ratio", "to_debt", "to_investing"}
    assert 0.0 <= alloc["ratio"] <= 1.0
    assert pytest.approx(alloc["to_debt"] + alloc["to_investing"], abs=1e-6) == base_payload()["monthly_extra"]

    proj = data["projections"]
    assert set(proj.keys()) == {"years", "assets_pos", "debt_neg", "net_worth"}

    n = base_payload()["horizon_years"] + 1 
    for k in ("years", "assets_pos", "debt_neg", "net_worth"):
        assert len(proj[k]) == n

    assert proj["years"][0] == date.today().year
    assert proj["years"][-1] == date.today().year + base_payload()["horizon_years"]

def test_sign_and_networth_coherence():
    r = client.post("/inference/recommend", json=base_payload())
    data = r.json()
    assets = data["projections"]["assets_pos"]
    debt_neg = data["projections"]["debt_neg"]
    net = data["projections"]["net_worth"]
    for a, d, n in zip(assets, debt_neg, net):
        assert a >= 0.0
        assert d <= 0.0
        assert pytest.approx(n, rel=1e-9, abs=1e-9) == a + d  
