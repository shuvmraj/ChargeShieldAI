"""FastAPI REST API Server for ChargeShield AI Risk Manager."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from chargeshield.api.schemas import (
    BatchRiskScoreResponse,
    BatchTransactionInput,
    DisputeEvidenceResponse,
    DisputeRequest,
    ModelInfoResponse,
    RiskFactor,
    RiskScoreComponents,
    RiskScoreResponse,
    TransactionInput,
)
from chargeshield.dispute.evidence_generator import DisputeEvidenceGenerator
from chargeshield.explainability.explainer import RiskExplainer
from chargeshield.models.model_trainer import ChargeShieldModelTrainer

ARTIFACTS_DIR = Path(os.getenv("CHARGESHIELD_ARTIFACTS_DIR", "models/artifacts"))

# Global application state
state: Dict[str, Any] = {
    "trainer": None,
    "explainer": None,
    "dispute_generator": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models and explainers into memory on application startup."""
    print(f"[*] Initializing ChargeShield AI backend...")
    state["dispute_generator"] = DisputeEvidenceGenerator()

    if (ARTIFACTS_DIR / "xgboost_model.joblib").exists():
        print(f"[*] Loading model artifacts from {ARTIFACTS_DIR}...")
        try:
            state["trainer"] = ChargeShieldModelTrainer.load_artifacts(ARTIFACTS_DIR)
            state["explainer"] = RiskExplainer(state["trainer"])
            print(f"[+] Successfully loaded ChargeShield AI ({len(state['trainer'].feature_names)} features, threshold={state['trainer'].threshold_optimizer.optimal_threshold:.4f})")
        except Exception as e:
            print(f"[!] Error loading artifacts: {e}. Running in lightweight fallback mode.")
    else:
        print(f"[!] Artifacts directory {ARTIFACTS_DIR} not found. Please run 'python scripts/train.py' first.")

    yield
    print("[*] Shutting down ChargeShield AI server...")


app = FastAPI(
    title="ChargeShield AI - Risk Management API",
    description="Defense-only AI Risk Manager for Razorpay: Pre-settlement chargeback detection & dispute evidence generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ChargeShield AI Risk Engine",
        "model_loaded": state["trainer"] is not None,
        "explainer_loaded": state["explainer"] is not None,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info() -> ModelInfoResponse:
    """Returns model metadata, threshold settings, and feature count."""
    if state["trainer"] is None:
        raise HTTPException(status_code=503, detail="Model is not currently loaded. Please run train.py.")

    opt = state["trainer"].threshold_optimizer
    return ModelInfoResponse(
        model_name="ChargeShield XGBoost + Isolation Forest Hybrid",
        version="1.0.0",
        num_features=len(state["trainer"].feature_names),
        optimal_threshold=opt.optimal_threshold,
        tier_cutoffs=opt.tier_cutoffs,
        status="ACTIVE_PRODUCTION",
    )


@app.post("/predict", response_model=RiskScoreResponse, tags=["Inference"])
async def predict_transaction(txn: TransactionInput) -> RiskScoreResponse:
    """Evaluates a single transaction in real-time, returning score, decision, and Top 5 risk factors."""
    if state["trainer"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model first.")

    txn_dict = txn.model_dump()

    # 1. Base prediction
    pred_res = state["trainer"].predict_single(txn_dict)

    # 2. SHAP Explainability Top 5 factors
    if state["explainer"] is not None:
        explanation = state["explainer"].explain_transaction(txn_dict, top_k=5)
        top_factors_raw = explanation["top_risk_factors"]
    else:
        top_factors_raw = []

    top_factors = [RiskFactor(**f) for f in top_factors_raw]

    return RiskScoreResponse(
        transaction_id=pred_res["transaction_id"],
        risk_score=pred_res["risk_score"],
        risk_tier=pred_res["risk_tier"],
        risk_label=pred_res["risk_label"],
        recommended_action=pred_res["recommended_action"],
        settlement_hold=pred_res["settlement_hold"],
        badge_color=pred_res["badge_color"],
        action_description=pred_res["action_description"],
        confidence=pred_res["confidence"],
        components=RiskScoreComponents(**pred_res["components"]),
        top_risk_factors=top_factors,
    )


@app.post("/predict/batch", response_model=BatchRiskScoreResponse, tags=["Inference"])
async def predict_batch_transactions(batch: BatchTransactionInput) -> BatchRiskScoreResponse:
    """Evaluates a batch of transactions and computes aggregate settlement hold metrics."""
    if state["trainer"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    results: List[RiskScoreResponse] = []
    hold_count = 0
    instant_count = 0
    hold_volume = 0.0

    for txn in batch.transactions:
        single_res = await predict_transaction(txn)
        results.append(single_res)

        if single_res.settlement_hold:
            hold_count += 1
            hold_volume += txn.amount_inr
        else:
            instant_count += 1

    return BatchRiskScoreResponse(
        total_evaluated=len(batch.transactions),
        flagged_hold_count=hold_count,
        instant_settlement_count=instant_count,
        flagged_volume_inr=round(hold_volume, 2),
        results=results,
    )


@app.post("/dispute/generate-evidence", response_model=DisputeEvidenceResponse, tags=["Dispute Defense"])
async def generate_dispute_evidence(req: DisputeRequest) -> DisputeEvidenceResponse:
    """Generates card network & NPCI-compliant arbitration evidence package."""
    if state["dispute_generator"] is None:
        state["dispute_generator"] = DisputeEvidenceGenerator()

    txn_dict = req.transaction.model_dump()
    txn_dict["dispute_reason"] = req.dispute_reason
    txn_dict["delivery_status"] = req.delivery_status
    txn_dict["delivery_awb"] = req.delivery_awb
    txn_dict["courier_partner"] = req.courier_partner
    if req.terms_accepted_timestamp:
        txn_dict["terms_accepted_timestamp"] = req.terms_accepted_timestamp

    meta = {"dispute_id": req.dispute_id} if req.dispute_id else None
    packet = state["dispute_generator"].generate_packet(txn_dict, dispute_metadata=meta)
    html_rendered = state["dispute_generator"].format_html_packet(packet)

    return DisputeEvidenceResponse(
        dispute_id=packet["dispute_id"],
        packet_generated_at=packet["packet_generated_at"],
        case_readiness_score=packet["case_readiness_score"],
        case_readiness_tier=packet["case_readiness_tier"],
        transaction_summary=packet["transaction_summary"],
        dispute_claim_details=packet["dispute_claim_details"],
        authentication_forensics=packet["authentication_forensics"],
        telemetry_evidence=packet["telemetry_evidence"],
        fulfillment_and_pod=packet["fulfillment_and_pod"],
        merchant_policy_alignment=packet["merchant_policy_alignment"],
        recommended_dispute_stance=packet["recommended_dispute_stance"],
        readiness_score_breakdown=packet["readiness_score_breakdown"],
        html_packet=html_rendered,
    )


@app.post("/dispute/render-html", response_class=HTMLResponse, tags=["Dispute Defense"])
async def render_dispute_html(req: DisputeRequest) -> HTMLResponse:
    """Directly returns printable HTML evidence packet."""
    resp = await generate_dispute_evidence(req)
    return HTMLResponse(content=resp.html_packet or "<h1>No Packet Content</h1>")
