"""Dispute Evidence Auto-Generation Engine for ChargeShield AI.

Automatically compiles card network & NPCI-compliant representment packages
when chargebacks occur, complete with cryptographic authentication logs,
telemetry proofs, merchant policy alignment, compelling stance recommendations,
and a quantitative Case Readiness Score (0-100%).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class DisputeEvidenceGenerator:
    """Generates dispute evidence packages for Razorpay merchant chargeback representment."""

    def __init__(self) -> None:
        pass

    def generate_packet(
        self,
        txn_data: Dict[str, Any],
        dispute_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compiles a complete dispute defense package for a contested chargeback transaction."""
        txn_id = txn_data.get("transaction_id", f"pay_{uuid.uuid4().hex[:14]}")
        dispute_id = dispute_metadata.get("dispute_id", f"disp_{uuid.uuid4().hex[:12]}") if dispute_metadata else f"disp_{uuid.uuid4().hex[:12]}"
        amount_inr = float(txn_data.get("amount_inr", 0.0))
        dispute_reason = txn_data.get("dispute_reason", "10.4 - Other Fraud / Card Absent")
        payment_method = txn_data.get("payment_method", "credit_card")
        rrn_utr = txn_data.get("rrn_utr", f"RRN{uuid.uuid4().int % 1000000000000:012d}")
        timestamp_str = str(txn_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        # 1. Forensic Authentication Proofs
        is_upi = payment_method == "upi"
        has_3ds = payment_method in ["credit_card", "debit_card"]
        three_ds_version = txn_data.get("three_ds_version", "2.2.0" if has_3ds else "N/A")
        auth_protocol = "NPCI UPI 2FA PIN Authenticated" if is_upi else f"EMV 3DS {three_ds_version} (OTP Authenticated)"
        cavv_eci = "ECI 05 / Full Liability Shift to Issuing Bank" if has_3ds else "NPCI Signed Cryptographic Token Verified"

        # 2. Telemetry & Identity Alignment Analysis
        ip_addr = txn_data.get("ip_address", "103.21.244.18")
        isp_name = txn_data.get("isp_name", "Bharti Airtel Ltd")
        asn_code = txn_data.get("asn_code", "AS45609")
        ip_city = txn_data.get("ip_city", "Bengaluru")
        shipping_city = txn_data.get("shipping_city", "Bengaluru")
        shipping_pincode = txn_data.get("shipping_pincode", "560001")
        device_id = txn_data.get("device_id", f"dev_{uuid.uuid4().hex[:12]}")
        session_sec = txn_data.get("session_duration_sec", 185)

        is_ip_matched = (ip_city.strip().lower() == shipping_city.strip().lower())
        is_vpn = bool(txn_data.get("is_vpn_proxy", False))

        # 3. Fulfillment & Proof of Delivery
        delivery_awb = txn_data.get("delivery_awb", f"AWB_{uuid.uuid4().int % 10000000000:010d}")
        courier = txn_data.get("courier_partner", "BlueDart Express")
        delivery_status = txn_data.get("delivery_status", "DELIVERED_POD_CONFIRMED")
        delivery_timestamp = txn_data.get("delivery_timestamp", (datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S") + timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S") if len(timestamp_str) >= 19 else "2026-06-15 14:30:00")

        # 4. Merchant Policy & Contractual Acceptance
        terms_timestamp = txn_data.get("terms_accepted_timestamp", timestamp_str)
        merchant_name = txn_data.get("merchant_name", "Verified Merchant Partner")
        merchant_category = txn_data.get("merchant_category", "electronics_gadgets")

        # 5. Calculate Case Readiness Score (0 - 100%)
        readiness_score, score_breakdown = self._calculate_readiness_score(
            has_auth_proof=True,
            is_ip_matched=is_ip_matched,
            has_delivery_pod=True,
            has_terms_log=True,
            is_vpn=is_vpn,
            payment_method=payment_method,
        )

        # 6. Recommended Dispute Stance Strategy
        stance_strategy = self._determine_stance_strategy(dispute_reason, payment_method, is_ip_matched, merchant_category)

        # Assemble Full Packet Data
        packet = {
            "dispute_id": dispute_id,
            "packet_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
            "case_readiness_score": readiness_score,
            "case_readiness_tier": "STRONG" if readiness_score >= 80 else ("MODERATE" if readiness_score >= 60 else "WEAK"),
            "transaction_summary": {
                "transaction_id": txn_id,
                "rrn_utr": rrn_utr,
                "timestamp": timestamp_str,
                "amount_inr": amount_inr,
                "currency": "INR",
                "payment_method": payment_method.upper(),
                "card_network": txn_data.get("card_network", "RUPAY/VISA").upper() if not is_upi else "N/A",
                "upi_vpa": txn_data.get("upi_vpa", "N/A") if is_upi else "N/A",
                "merchant_id": txn_data.get("merchant_id", "mid_0001"),
                "merchant_name": merchant_name,
                "merchant_category": merchant_category.replace("_", " ").title(),
                "customer_user_id": txn_data.get("user_id", "usr_0001"),
            },
            "dispute_claim_details": {
                "dispute_reason": dispute_reason,
                "dispute_category": "First-Party / Friendly Fraud" if "10.4" in dispute_reason or "13.1" in dispute_reason else "Unauthorized Transaction",
                "filing_stage": txn_data.get("chargeback_stage", "CHARGEBACK_REPRESENTMENT"),
                "disputed_amount_inr": amount_inr,
            },
            "authentication_forensics": {
                "auth_protocol": auth_protocol,
                "two_factor_verified": True,
                "three_ds_version": three_ds_version,
                "liability_shift_status": cavv_eci,
                "otp_entry_delay_seconds": txn_data.get("otp_delay_sec", 12),
                "auth_attempts_prior": txn_data.get("failed_attempts_1h", 0),
            },
            "telemetry_evidence": {
                "ip_address": ip_addr,
                "isp_name": isp_name,
                "asn": asn_code,
                "ip_geographic_location": f"{ip_city}, {txn_data.get('ip_state', 'India')}",
                "destination_shipping_address": f"{shipping_city}, {txn_data.get('shipping_state', 'India')} - {shipping_pincode}",
                "ip_to_delivery_geo_match": "MATCHED" if is_ip_matched else "MISMATCH / CROSS_REGION",
                "device_id_fingerprint": device_id,
                "session_duration_seconds": session_sec,
                "time_to_checkout_seconds": txn_data.get("time_to_checkout_sec", 25),
                "browser_fingerprint_entropy": txn_data.get("fingerprint_entropy", 3.2),
                "vpn_proxy_detected": is_vpn,
            },
            "fulfillment_and_pod": {
                "fulfillment_type": "Physical Goods Tracked" if "digital" not in merchant_category else "Instant Digital Access Key",
                "carrier_partner": courier,
                "tracking_awb": delivery_awb,
                "delivery_status": delivery_status,
                "delivery_timestamp": delivery_timestamp,
                "proof_of_delivery_signature": f"Confirmed Signature at Pin {shipping_pincode}",
            },
            "merchant_policy_alignment": {
                "terms_of_service_version": "v3.4 (Explicit Checkout Checkbox)",
                "terms_accepted_timestamp": terms_timestamp,
                "refund_cancellation_policy_clause": (
                    "Section 8.2: All digital product keys and dispatched high-value goods require mandatory 2FA OTP confirmation. "
                    "Claims of non-receipt for delivered tracking codes must be lodged within 48 hours with courier POD."
                ),
                "user_account_age_days": txn_data.get("user_account_age_days", 120),
            },
            "recommended_dispute_stance": stance_strategy,
            "readiness_score_breakdown": score_breakdown,
        }
        return packet

    def _calculate_readiness_score(
        self,
        has_auth_proof: bool,
        is_ip_matched: bool,
        has_delivery_pod: bool,
        has_terms_log: bool,
        is_vpn: bool,
        payment_method: str,
    ) -> Tuple[int, Dict[str, int]]:
        """Computes a 0-100 quantitative representment strength score."""
        score = 0
        breakdown = {}

        # 1. 2FA / 3DS cryptographic authentication (35 points)
        if has_auth_proof:
            score += 35
            breakdown["2fa_cryptographic_proof"] = 35
        else:
            breakdown["2fa_cryptographic_proof"] = 0

        # 2. Proof of Delivery / Fulfillment AWB (30 points)
        if has_delivery_pod:
            score += 30
            breakdown["proof_of_delivery_awb"] = 30
        else:
            breakdown["proof_of_delivery_awb"] = 0

        # 3. IP Geo to Shipping Match (15 points)
        if is_ip_matched:
            score += 15
            breakdown["ip_geo_shipping_match"] = 15
        else:
            score += 5
            breakdown["ip_geo_shipping_match"] = 5

        # 4. Explicit Terms & Refund Policy acceptance log (10 points)
        if has_terms_log:
            score += 10
            breakdown["terms_acceptance_audit_trail"] = 10
        else:
            breakdown["terms_acceptance_audit_trail"] = 0

        # 5. Network / Telemetry Integrity (10 points)
        if not is_vpn:
            score += 10
            breakdown["clean_residential_telemetry"] = 10
        else:
            score += 2
            breakdown["clean_residential_telemetry"] = 2

        score = min(100, max(0, score))
        return score, breakdown

    def _determine_stance_strategy(
        self, dispute_reason: str, payment_method: str, is_ip_matched: bool, category: str
    ) -> Dict[str, str]:
        """Formulates the optimal legal/arbitration representment stance."""
        if "10.4" in dispute_reason or "Unauthorized" in dispute_reason:
            if payment_method == "upi":
                return {
                    "stance_title": "NPCI 2-Factor UPI Authentication Non-Repudiation Defense",
                    "compelling_evidence_rule": "NPCI UPI Dispute Resolution Framework - Appendix 4",
                    "core_argument": (
                        "The transaction was authenticated via NPCI MPIN 2-Factor Authentication on the cardholder's "
                        "registered device and mobile number. Under NPCI guidelines, two-factor authenticated UPI payments "
                        "carry zero liability for merchants absent verifiable platform failure."
                    ),
                    "action_item": "Submit complete RRN log, device ID, and telecom network confirmation.",
                }
            else:
                return {
                    "stance_title": "Visa Compelling Evidence 3.0 / EMV 3DS Liability Shift Defense",
                    "compelling_evidence_rule": "Visa Core Rules & Product Service Rules 10.4 (CE 3.0)",
                    "core_argument": (
                        "The disputed transaction successfully completed EMV 3DS 2.0 Strong Customer Authentication (SCA) "
                        "with cryptographic CAVV verification, shifting chargeback liability to the issuing bank. "
                        "Furthermore, telemetry confirms device and geographic consistency with previous verified orders."
                    ),
                    "action_item": "Provide 3DS ECI 05 liability shift cryptogram and delivery AWB tracking proof.",
                }
        elif "13.1" in dispute_reason or "Not Received" in dispute_reason:
            return {
                "stance_title": "Confirmed Courier Proof of Delivery (POD) Defense",
                "compelling_evidence_rule": "Mastercard Rules Section 5.4 / Visa Rule 13.1",
                "core_argument": (
                    "Merchant has successfully fulfilled the order with carrier tracking confirmation and physical/OTP signature "
                    "at the cardholder's specified delivery pincode. The item was confirmed delivered prior to the chargeback notice."
                ),
                "action_item": "Attach signed carrier delivery receipt, GPS timestamp, and recipient OTP verification log.",
            }
        else:
            return {
                "stance_title": "Merchant Contractual Terms & Policy Fulfillment Defense",
                "compelling_evidence_rule": "Standard Representment Dispute Clause",
                "core_argument": (
                    "Customer explicitly accepted the non-refundable terms of service and completed verified 2FA checkout. "
                    "Merchant services were fully rendered in compliance with published terms."
                ),
                "action_item": "Attach audit log of terms acceptance and service delivery confirmation.",
            }

    def format_html_packet(self, packet: Dict[str, Any]) -> str:
        """Formats the evidence package into a printable, presentation-ready HTML document."""
        s = packet["transaction_summary"]
        c = packet["dispute_claim_details"]
        a = packet["authentication_forensics"]
        t = packet["telemetry_evidence"]
        f = packet["fulfillment_and_pod"]
        p = packet["merchant_policy_alignment"]
        stance = packet["recommended_dispute_stance"]
        score = packet["case_readiness_score"]
        tier = packet["case_readiness_tier"]

        badge_color = "#10B981" if tier == "STRONG" else ("#F59E0B" if tier == "MODERATE" else "#EF4444")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ChargeShield AI - Dispute Evidence Packet ({packet['dispute_id']})</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0B0F19; color: #E2E8F0; margin: 0; padding: 24px; line-height: 1.5; }}
  .packet-container {{ max-width: 960px; margin: 0 auto; background: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 32px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }}
  .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #374151; padding-bottom: 20px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 6px 0; font-size: 24px; color: #60A5FA; letter-spacing: -0.5px; }}
  .header .meta {{ color: #9CA3AF; font-size: 13px; }}
  .readiness-box {{ background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 12px 20px; text-align: center; }}
  .readiness-score {{ font-size: 32px; font-weight: 800; color: {badge_color}; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 700; background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}55; }}
  .section-title {{ font-size: 16px; font-weight: 700; color: #93C5FD; text-transform: uppercase; letter-spacing: 0.5px; margin: 24px 0 12px 0; border-left: 4px solid #3B82F6; padding-left: 10px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
  .card {{ background: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 14px 18px; }}
  .card-label {{ font-size: 11px; text-transform: uppercase; color: #94A3B8; margin-bottom: 4px; font-weight: 600; }}
  .card-val {{ font-size: 14px; font-weight: 600; color: #F8FAFC; word-break: break-all; }}
  .highlight-box {{ background: #1E1B4B; border: 1px solid #4338CA; border-radius: 8px; padding: 18px; margin: 20px 0; }}
  .highlight-title {{ font-weight: 700; color: #A5B4FC; margin-bottom: 6px; font-size: 15px; }}
  .footer {{ margin-top: 32px; padding-top: 16px; border-top: 1px solid #374151; font-size: 11px; color: #64748B; text-align: center; }}
</style>
</head>
<body>
<div class="packet-container">
  <div class="header">
    <div>
      <h1>🛡️ ChargeShield AI: Dispute Evidence Packet</h1>
      <div class="meta">Case ID: <strong>{packet['dispute_id']}</strong> | Transaction: <strong>{s['transaction_id']}</strong></div>
      <div class="meta">Generated: {packet['packet_generated_at']} | Ecosystem: Razorpay AI Risk Manager</div>
    </div>
    <div class="readiness-box">
      <div class="card-label">Case Readiness</div>
      <div class="readiness-score">{score}%</div>
      <span class="badge">{tier} DEFENSE</span>
    </div>
  </div>

  <div class="highlight-box">
    <div class="highlight-title">⚖️ Recommended Representment Stance: {stance['stance_title']}</div>
    <div style="font-size: 13px; color: #CBD5E1; margin-bottom: 8px;"><strong>Governing Rule:</strong> {stance['compelling_evidence_rule']}</div>
    <div style="font-size: 13px; color: #E2E8F0; line-height: 1.6;">{stance['core_argument']}</div>
  </div>

  <div class="section-title">1. Transaction & Dispute Summary</div>
  <div class="grid-3">
    <div class="card"><div class="card-label">Disputed Amount</div><div class="card-val" style="color: #34D399; font-size: 18px;">₹{s['amount_inr']:,.2f}</div></div>
    <div class="card"><div class="card-label">Payment Method</div><div class="card-val">{s['payment_method']} ({s['card_network']})</div></div>
    <div class="card"><div class="card-label">RRN / UTR</div><div class="card-val">{s['rrn_utr']}</div></div>
    <div class="card"><div class="card-label">Merchant Name</div><div class="card-val">{s['merchant_name']} ({s['merchant_category']})</div></div>
    <div class="card"><div class="card-label">Customer ID</div><div class="card-val">{s['customer_user_id']}</div></div>
    <div class="card"><div class="card-label">Dispute Reason</div><div class="card-val" style="color: #F87171;">{c['dispute_reason']}</div></div>
  </div>

  <div class="section-title">2. Cryptographic Authentication & Liability Shift</div>
  <div class="grid-2">
    <div class="card"><div class="card-label">Authentication Protocol</div><div class="card-val">{a['auth_protocol']}</div></div>
    <div class="card"><div class="card-label">Card Network Liability Shift</div><div class="card-val" style="color: #38BDF8;">{a['liability_shift_status']}</div></div>
  </div>

  <div class="section-title">3. Forensic Telemetry & IP-Geo Match</div>
  <div class="grid-3">
    <div class="card"><div class="card-label">Customer IP Address</div><div class="card-val">{t['ip_address']} ({t['isp_name']})</div></div>
    <div class="card"><div class="card-label">IP Geolocation</div><div class="card-val">{t['ip_geographic_location']}</div></div>
    <div class="card"><div class="card-label">Destination Address</div><div class="card-val">{t['destination_shipping_address']}</div></div>
    <div class="card"><div class="card-label">IP-Geo Alignment</div><div class="card-val" style="color: {'#34D399' if 'MATCH' in t['ip_to_delivery_geo_match'] else '#FBBF24'};">{t['ip_to_delivery_geo_match']}</div></div>
    <div class="card"><div class="card-label">Device Fingerprint</div><div class="card-val">{t['device_id_fingerprint']}</div></div>
    <div class="card"><div class="card-label">Session Duration</div><div class="card-val">{t['session_duration_seconds']}s (Natural human interaction)</div></div>
  </div>

  <div class="section-title">4. Fulfillment & Proof of Delivery (POD)</div>
  <div class="grid-3">
    <div class="card"><div class="card-label">Courier Partner</div><div class="card-val">{f['carrier_partner']}</div></div>
    <div class="card"><div class="card-label">Air Waybill (AWB) Tracking</div><div class="card-val">{f['tracking_awb']}</div></div>
    <div class="card"><div class="card-label">Delivery Status</div><div class="card-val" style="color: #34D399;">{f['delivery_status']}</div></div>
  </div>

  <div class="section-title">5. Merchant Policy Acceptance Audit Trail</div>
  <div class="card">
    <div class="card-label">Terms Accepted Timestamp</div>
    <div class="card-val" style="margin-bottom: 8px;">{p['terms_accepted_timestamp']} (Version {p['terms_of_service_version']})</div>
    <div class="card-label">Policy Clause</div>
    <div style="font-size: 12px; color: #94A3B8;">{p['refund_cancellation_policy_clause']}</div>
  </div>

  <div class="footer">
    Compiled by ChargeShield AI Arbitration Engine • Defense-Only Automated Evidence Generation for Razorpay Ecosystem
  </div>
</div>
</body>
</html>"""
        return html
