"""Pydantic schemas for ChargeShield AI FastAPI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Input payload representing a payment transaction to be evaluated."""
    transaction_id: str = Field(default="pay_test_001", description="Unique transaction ID")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of transaction")
    amount_inr: float = Field(..., gt=0, description="Transaction amount in INR")
    user_id: str = Field(default="usr_0001", description="User identifier")
    merchant_id: str = Field(default="mid_0001", description="Merchant identifier")
    merchant_category: str = Field(default="electronics_gadgets", description="Merchant industry category")
    payment_method: str = Field(default="upi", description="Payment method: upi, credit_card, debit_card, netbanking, wallet_emi")
    card_network: Optional[str] = Field(default="", description="VISA, MASTERCARD, RUPAY, AMEX")
    upi_vpa: Optional[str] = Field(default="", description="Customer UPI Virtual Payment Address")
    ip_address: Optional[str] = Field(default="103.21.244.18", description="Client IP address")
    isp_name: Optional[str] = Field(default="Reliance Jio Infocomm", description="ISP provider")
    asn_code: Optional[str] = Field(default="AS55836", description="ASN code")
    ip_city: Optional[str] = Field(default="Bengaluru", description="IP Geo City")
    ip_state: Optional[str] = Field(default="Karnataka", description="IP Geo State")
    shipping_city: Optional[str] = Field(default="Bengaluru", description="Destination shipping city")
    shipping_state: Optional[str] = Field(default="Karnataka", description="Destination shipping state")
    shipping_pincode: Optional[str] = Field(default="560001", description="Shipping pin code")
    billing_city: Optional[str] = Field(default="Bengaluru", description="Billing city")
    billing_state: Optional[str] = Field(default="Karnataka", description="Billing state")
    device_id: Optional[str] = Field(default="dev_mobile_001", description="Client device hardware fingerprint")
    is_vpn_proxy: Optional[int] = Field(default=0, description="1 if datacenter VPN/proxy detected")
    is_emulator: Optional[int] = Field(default=0, description="1 if Android emulator detected")
    is_rooted: Optional[int] = Field(default=0, description="1 if rooted/jailbroken device")
    fingerprint_entropy: Optional[float] = Field(default=3.2, description="Browser canvas/webgl entropy score")
    session_duration_sec: Optional[int] = Field(default=65, description="Total session duration in seconds")
    time_to_checkout_sec: Optional[int] = Field(default=22, description="Seconds from landing to checkout")
    page_views: Optional[int] = Field(default=4, description="Page views count in session")
    typing_speed_wpm: Optional[int] = Field(default=58, description="Typing cadence in words per minute")
    mouse_entropy: Optional[float] = Field(default=0.82, description="Mouse cursor movement entropy")
    failed_attempts_1h: Optional[int] = Field(default=0, description="Failed payment/auth attempts in last hour")
    failed_attempts_24h: Optional[int] = Field(default=0, description="Failed payment attempts in last 24h")
    cvv_retries: Optional[int] = Field(default=0, description="CVV retry count")
    otp_delay_sec: Optional[int] = Field(default=11, description="OTP entry latency in seconds")
    is_international_card: Optional[int] = Field(default=0, description="1 if card is issued outside India")
    is_international_ip: Optional[int] = Field(default=0, description="1 if IP is foreign")
    user_account_age_days: Optional[int] = Field(default=90, description="Customer account age in days")
    user_order_index: Optional[int] = Field(default=3, description="Order index for user")


class RiskFactor(BaseModel):
    """Plain-English top risk contributor."""
    feature_name: str
    shap_contribution: float
    feature_value: Any
    factor_title: str
    description: str
    severity: str
    category: str


class RiskScoreComponents(BaseModel):
    """Decomposition of the hybrid score."""
    xgboost_prob: float
    isolation_anomaly_score: float
    rule_penalty_score: float


class RiskScoreResponse(BaseModel):
    """ChargeShield Risk Evaluation Response."""
    transaction_id: str
    risk_score: float = Field(..., description="0-100 ChargeShield Risk Score")
    risk_tier: str = Field(..., description="LOW, MODERATE, HIGH, CRITICAL")
    risk_label: str
    recommended_action: str
    settlement_hold: bool
    badge_color: str
    action_description: str
    confidence: float
    components: RiskScoreComponents
    top_risk_factors: List[RiskFactor]


class BatchTransactionInput(BaseModel):
    """Batch list of transactions."""
    transactions: List[TransactionInput]


class BatchRiskScoreResponse(BaseModel):
    """Batch evaluation summary."""
    total_evaluated: int
    flagged_hold_count: int
    instant_settlement_count: int
    flagged_volume_inr: float
    results: List[RiskScoreResponse]


class DisputeRequest(BaseModel):
    """Request to generate dispute evidence package."""
    transaction: TransactionInput
    dispute_id: Optional[str] = None
    dispute_reason: Optional[str] = "10.4 - Other Fraud / Card Absent (Cardholder Disputes Transaction)"
    delivery_status: Optional[str] = "DELIVERED_POD_CONFIRMED"
    delivery_awb: Optional[str] = "AWB_9841204918"
    courier_partner: Optional[str] = "BlueDart Express"
    terms_accepted_timestamp: Optional[str] = None


class DisputeEvidenceResponse(BaseModel):
    """Structured Dispute Evidence Package."""
    dispute_id: str
    packet_generated_at: str
    case_readiness_score: int
    case_readiness_tier: str
    transaction_summary: Dict[str, Any]
    dispute_claim_details: Dict[str, Any]
    authentication_forensics: Dict[str, Any]
    telemetry_evidence: Dict[str, Any]
    fulfillment_and_pod: Dict[str, Any]
    merchant_policy_alignment: Dict[str, Any]
    recommended_dispute_stance: Dict[str, Any]
    readiness_score_breakdown: Dict[str, int]
    html_packet: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Model operational metadata."""
    model_name: str
    version: str
    num_features: int
    optimal_threshold: float
    tier_cutoffs: Dict[str, float]
    status: str
