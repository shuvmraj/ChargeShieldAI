"""Unit tests for FastAPI endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from chargeshield.api.main import app, state
from chargeshield.models.model_trainer import ChargeShieldModelTrainer
from chargeshield.explainability.explainer import RiskExplainer
from chargeshield.dispute.evidence_generator import DisputeEvidenceGenerator


@pytest.fixture(scope="module")
def client():
    # Preload state for fast test client
    state["trainer"] = ChargeShieldModelTrainer.load_artifacts("models/artifacts")
    state["explainer"] = RiskExplainer(state["trainer"])
    state["dispute_generator"] = DisputeEvidenceGenerator()
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["num_features"] >= 80
    assert "optimal_threshold" in data


def test_predict_single_endpoint(client):
    payload = {
        "transaction_id": "pay_test_api_001",
        "amount_inr": 2500.0,
        "merchant_category": "fashion_apparel",
        "payment_method": "upi",
        "shipping_city": "Bengaluru",
        "ip_city": "Bengaluru",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert len(data["top_risk_factors"]) <= 5


def test_batch_predict_endpoint(client):
    payload = {
        "transactions": [
            {"transaction_id": "pay_b1", "amount_inr": 1500.0, "payment_method": "upi"},
            {"transaction_id": "pay_b2", "amount_inr": 85000.0, "payment_method": "credit_card", "is_vpn_proxy": 1},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_evaluated"] == 2
    assert len(data["results"]) == 2


def test_generate_dispute_endpoint(client):
    payload = {
        "transaction": {
            "transaction_id": "pay_dispute_test",
            "amount_inr": 18500.0,
            "payment_method": "credit_card",
        },
        "dispute_reason": "10.4 - Other Fraud (Cardholder Disputes Transaction)",
    }
    response = client.post("/dispute/generate-evidence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "dispute_id" in data
    assert "case_readiness_score" in data
    assert "recommended_dispute_stance" in data
