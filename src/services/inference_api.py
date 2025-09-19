########################################
#### Dependencies
########################################
from __future__ import annotations
from typing import List, Optional
from datetime import date
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat

from src.simulation.paydown_sim import (
    Debt as SimDebt, Investments as SimInv, Scenario, simulate_projection
)
from src.features.build_features import build_features

########################################
#### API
########################################
app = FastAPI(title="Financial Advisor ML - Inference API")

class Debt(BaseModel):
    amount: confloat(ge=0) = Field(..., description="Principal >= 0")
    apr: confloat(ge=-0.5, le=1.0) = Field(..., description="Yearly APR in [-0.5, 1.0]")

class Investments(BaseModel):
    equity_value: confloat(ge=0) = 0.0
    property_value: confloat(ge=0) = 0.0
    property_growth_rate: confloat(ge=-0.5, le=1.0) = 0.02
    equity_return_rate: confloat(ge=-0.5, le=1.0) = 0.08

class RecommendationRequest(BaseModel):
    debts: List[Debt] = Field(default_factory=list)
    investments: Investments
    horizon_years: conint(ge=1, le=60) = 10
    monthly_extra: confloat(ge=0) = 0.0
    # test hook to force a ratio (used by tests)
    debug_force_ratio: Optional[float] = None

class Allocation(BaseModel):
    ratio: confloat(ge=0, le=1)
    to_debt: float
    to_investing: float

class Projections(BaseModel):
    years: List[int]
    assets_pos: List[float]
    debt_neg: List[float]
    net_worth: List[float]

class RecommendationResponse(BaseModel):
    allocation: Allocation
    projections: Projections
    explain: List[str]

# ---------- Prediction plumbing ----------

def heuristic_predict_ratio(features: dict, req: RecommendationRequest) -> float:
    """
    Simple, deterministic baseline while training is pending:
    - If max APR >= 10% -> invest 20%
    - Else invest 60%
    """
    if req.debug_force_ratio is not None:
        return float(req.debug_force_ratio)
    max_apr = float(features.get("max_apr", 0.0))
    return 0.2 if max_apr >= 0.10 else 0.6

PREDICTOR_FN = heuristic_predict_ratio  # swap with RF later

# ---------- Endpoint ----------

@app.post("/inference/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    
    #### Scenario object for simulator ####
    sim_debts = [SimDebt(amount=d.amount, apr=d.apr) for d in req.debts]
    sim_inv = SimInv(
        equity_value=req.investments.equity_value,
        property_value=req.investments.property_value,
        equity_return_rate=req.investments.equity_return_rate,
        property_growth_rate=req.investments.property_growth_rate,
    )
    sc = Scenario(
        debts=sim_debts,
        investments=sim_inv,
        horizon_years=int(req.horizon_years),
        monthly_extra=float(req.monthly_extra),
    )

    #### Build features and get ratio ####
    features = build_features(sc)
    ratio_raw = PREDICTOR_FN(features, req)
    ratio = float(np.clip(ratio_raw, 0.0, 1.0))

    #### Allocation (extra-only) ####
    to_investing = round(req.monthly_extra * ratio, 10)
    to_debt = round(req.monthly_extra - to_investing, 10)

    #### Simulate yearly projection with predicted split ####
    proj = simulate_projection(sc, invest_fraction=ratio)

    #### Compose response ####
    return RecommendationResponse(
        allocation=Allocation(ratio=ratio, to_debt=to_debt, to_investing=to_investing),
        projections=Projections(
            years=proj.years,
            assets_pos=proj.assets_pos,
            debt_neg=proj.debt_neg,      
            net_worth=proj.net_worth,
        ),
        explain=[
            "Yearly compounding; avalanche payoff.",
            f"Equity {req.investments.equity_return_rate:.2%}/yr; Property {req.investments.property_growth_rate:.2%}/yr.",
            "Split applies only to monthly surplus.",
        ],
    )
