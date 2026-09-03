"""Unit tests for Dispute Evidence Auto-Generator."""

import pytest
from chargeshield.dispute.evidence_generator import DisputeEvidenceGenerator


def test_dispute_packet_generation():
    generator = DisputeEvidenceGenerator()

    sample_txn = {
        "transaction_id": "pay_test_dispute_123",
        "rrn_utr": "RRN998877665544",
        "amount_inr": 24500.0,
        "payment_method": "credit_card",
        "card_network": "VISA",
        "merchant_name": "Tech Galaxy Electronics",
        "merchant_category": "electronics_gadgets",
        "user_id": "usr_9988",
        "ip_address": "103.21.244.18",
        "isp_name": "Bharti Airtel Ltd",
        "asn_code": "AS45609",
        "ip_city": "Bengaluru",
        "shipping_city": "Bengaluru",
        "shipping_pincode": "560001",
        "device_id": "dev_mobile_ios_88",
        "session_duration_sec": 140,
        "delivery_awb": "AWB_9876543210",
        "delivery_status": "DELIVERED_POD_CONFIRMED",
        "dispute_reason": "10.4 - Other Fraud / Card Absent",
    }

    packet = generator.generate_packet(sample_txn)

    assert "dispute_id" in packet
    assert "case_readiness_score" in packet
    assert 0 <= packet["case_readiness_score"] <= 100
    assert packet["case_readiness_tier"] in ["STRONG", "MODERATE", "WEAK"]
    assert "transaction_summary" in packet
    assert "authentication_forensics" in packet
    assert "telemetry_evidence" in packet
    assert "fulfillment_and_pod" in packet
    assert "merchant_policy_alignment" in packet
    assert "recommended_dispute_stance" in packet

    # HTML rendering
    html_output = generator.format_html_packet(packet)
    assert "<!DOCTYPE html>" in html_output
    assert "ChargeShield AI" in html_output
    assert str(sample_txn["transaction_id"]) in html_output
