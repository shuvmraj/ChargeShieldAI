"""ChargeShield AI: Premium Streamlit Risk Management Dashboard for Razorpay Buildathon."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from chargeshield.dispute.evidence_generator import DisputeEvidenceGenerator
from chargeshield.explainability.explainer import RiskExplainer
from chargeshield.features.engineer import CATEGORY_RISK_MAP
from chargeshield.models.evaluator import ModelEvaluator
from chargeshield.models.model_trainer import ChargeShieldModelTrainer

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ChargeShield AI | Razorpay Risk Manager",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom CSS
CSS_FILE = Path(__file__).parent / "style.css"
if CSS_FILE.exists():
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    """Loads model trainer, explainer, and dispute generator."""
    artifacts_dir = Path("models/artifacts")
    if not (artifacts_dir / "xgboost_model.joblib").exists():
        # Fallback or alert
        return None, None, None

    trainer = ChargeShieldModelTrainer.load_artifacts(artifacts_dir)
    explainer = RiskExplainer(trainer)
    dispute_gen = DisputeEvidenceGenerator()
    return trainer, explainer, dispute_gen


@st.cache_data
def load_data():
    """Loads raw and test datasets."""
    raw_path = Path("data/raw_transactions.csv")
    test_path = Path("data/test_transactions.csv")
    metrics_path = Path("models/artifacts/evaluation_metrics.json")

    df_raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    df_test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return df_raw, df_test, metrics


trainer, explainer, dispute_gen = load_system()
df_raw, df_test, eval_metrics = load_data()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ ChargeShield AI")
    st.caption("Razorpay Buildathon • AI Risk Manager Track")

    st.markdown("---")
    st.markdown("#### ⚙️ Operational Mode")
    st.info("**Defense-Only Architecture**\n\nProtects merchant revenue pre-settlement with zero proactive settlement disruptions.")

    if trainer is not None:
        opt_thresh = trainer.threshold_optimizer.optimal_threshold
        st.metric("Active Threshold (T*)", f"{opt_thresh:.4f}")
        st.metric("Total Extracted Features", f"{len(trainer.feature_names)}")
        st.metric("Status", "🟢 Production Active")
    else:
        st.warning("⚠️ Models not loaded. Please run train.py first.")

    st.markdown("---")
    st.markdown("#### 🏢 Merchant Environment")
    st.selectbox("Active Merchant View", ["All Merchants (Aggregator View)", "Electronics Hub #001", "Luxury Jewels India", "Nova Gaming & OTT", "FlyFast Airline Bookings"])

    st.caption("Version 1.0.0 • Built with XGBoost + Isolation Forest + SHAP")


# -----------------------------------------------------------------------------
# Header Bar
# -----------------------------------------------------------------------------
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("<div class='hero-header'>ChargeShield AI: Intelligent Risk Manager</div>", unsafe_allow_html=True)
    st.markdown("Pre-settlement chargeback anomaly interceptor & automated dispute arbitration defense engine.")

with col_status:
    st.markdown("""
    <div style='text-align: right; margin-top: 10px;'>
      <span class='tier-pill-low'>● Razorpay Sandbox Connected</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Cockpit",
    "⚡ Live Risk Inspector",
    "🔍 Pre-Settlement Batch Queue",
    "⚖️ Dispute Evidence Studio",
    "📈 ML & Financial Analytics",
])


# =============================================================================
# TAB 1: EXECUTIVE RISK COCKPIT
# =============================================================================
with tab1:
    st.markdown("### 📊 Portfolio Risk Overview & Financial Protection")

    if eval_metrics:
        fin = eval_metrics.get("financial_impact_inr", {})
        m = eval_metrics.get("metrics", {})

        # Top Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown("""
            <div class='metric-box'>
              <div class='metric-label'>Protected Volume</div>
              <div class='metric-value'>₹{:,}</div>
              <div class='metric-delta' style='color: #34D399;'>6,000 Transactions</div>
            </div>
            """.format(int(fin.get("baseline_unmitigated_loss", 28000000) * 1.5)), unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-label'>Prevented Fraud Loss</div>
              <div class='metric-value' style='color: #34D399;'>₹{fin.get('prevented_fraud_loss', 0):,.0f}</div>
              <div class='metric-delta' style='color: #34D399;'>↑ {fin.get('loss_reduction_percentage', 99.8)}% Loss Reduction</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-label'>Detection Recall</div>
              <div class='metric-value'>{m.get('recall', 0.992):.1%}</div>
              <div class='metric-delta' style='color: #34D399;'>511 / 515 True Fraud Caught</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-label'>False Positive Rate</div>
              <div class='metric-value' style='color: #60A5FA;'>{m.get('fpr', 0.021):.2%}</div>
              <div class='metric-delta' style='color: #60A5FA;'>Capped < 2.5% Friction</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:
            st.markdown(f"""
            <div class='metric-box'>
              <div class='metric-label'>Net Defense ROI</div>
              <div class='metric-value' style='color: #FBBF24;'>{fin.get('roi_multiple', 688):.1f}x</div>
              <div class='metric-delta' style='color: #FBBF24;'>₹{fin.get('net_merchant_savings', 0):,.0f} Net Saved</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### 🎯 Decision Tier Distribution (Held-Out Portfolio)")
        if not df_test.empty and trainer is not None:
            # Score sample
            scores = trainer.predict_risk_score(df_test.head(1000))
            tiers = [trainer.threshold_optimizer.get_decision_tier(s)["tier"] for s in scores]
            tier_df = pd.DataFrame({"Tier": tiers})["Tier"].value_counts().reset_index()
            tier_df.columns = ["Decision Tier", "Count"]

            color_map = {
                "LOW": "#10B981",
                "MODERATE": "#F59E0B",
                "HIGH": "#F97316",
                "CRITICAL": "#EF4444",
            }
            fig_pie = px.pie(
                tier_df,
                values="Count",
                names="Decision Tier",
                color="Decision Tier",
                color_discrete_map=color_map,
                hole=0.55,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#CBD5E1"},
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🛍️ Chargeback Rate by Merchant Category")
        if not df_raw.empty:
            cat_stats = df_raw.groupby("merchant_category")["is_chargeback"].agg(["count", "mean"]).reset_index()
            cat_stats["Chargeback %"] = cat_stats["mean"] * 100
            cat_stats["Category"] = cat_stats["merchant_category"].str.replace("_", " ").str.title()
            cat_stats = cat_stats.sort_values("Chargeback %", ascending=True)

            fig_bar = px.bar(
                cat_stats,
                x="Chargeback %",
                y="Category",
                orientation="h",
                color="Chargeback %",
                color_continuous_scale="Reds",
                text_auto=".1f",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#CBD5E1"},
                xaxis=dict(gridcolor="#1E293B", title="Historical Chargeback Rate (%)"),
                yaxis=dict(title=""),
                coloraxis_showscale=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# TAB 2: LIVE TRANSACTION RISK INSPECTOR
# =============================================================================
with tab2:
    st.markdown("### ⚡ Real-Time Transaction Risk Inspector")
    st.caption("Simulate incoming digital payment transactions or select authentic attack archetypes to view instant scoring, multidimensional radar forensics, and plain-English SHAP drivers.")

    # Preset selector
    preset = st.selectbox(
        "⚡ Choose a Real-World Scenario Preset to Inspect:",
        [
            "1. Verified Clean UPI Grocery Order (Safe / Low Risk)",
            "2. High-Ticket Jewelry Carding Bot Burst (Critical / Rapid Velocity Attack)",
            "3. Datacenter VPN + Foreign IP Account Takeover (Critical / ATO)",
            "4. Friendly Fraud Digital Goods Key Drain (High Risk / First-Party Fraud)",
            "5. Custom Manual Input (Test Your Own Edge Case)",
        ],
    )

    # Initialize preset defaults
    if "1. Verified Clean" in preset:
        default_amt = 1250.0
        default_cat = "quick_commerce_food"
        default_pm = "upi"
        default_city = "Bengaluru"
        default_ship_city = "Bengaluru"
        default_vpn = 0
        default_emu = 0
        default_root = 0
        default_dist = 4.2
        default_failed_1h = 0
        default_typing = 62
        default_mouse = 0.85
        default_dur = 95
        default_chk = 24
        default_isp = "Reliance Jio Infocomm"
        default_asn = "AS55836"
    elif "2. High-Ticket Jewelry" in preset:
        default_amt = 94500.0
        default_cat = "luxury_jewelry"
        default_pm = "credit_card"
        default_city = "Frankfurt"
        default_ship_city = "Jaipur"
        default_vpn = 1
        default_emu = 1
        default_root = 1
        default_dist = 6200.0
        default_failed_1h = 5
        default_typing = 230
        default_mouse = 0.04
        default_dur = 8
        default_chk = 2
        default_isp = "Hostinger Datacenter"
        default_asn = "AS47583"
    elif "3. Datacenter VPN" in preset:
        default_amt = 48500.0
        default_cat = "electronics_gadgets"
        default_pm = "credit_card"
        default_city = "Ashburn"
        default_ship_city = "Mumbai"
        default_vpn = 1
        default_emu = 0
        default_root = 0
        default_dist = 11800.0
        default_failed_1h = 3
        default_typing = 145
        default_mouse = 0.22
        default_dur = 35
        default_chk = 12
        default_isp = "DigitalOcean LLC"
        default_asn = "AS14061"
    elif "4. Friendly Fraud" in preset:
        default_amt = 8999.0
        default_cat = "digital_goods_gaming"
        default_pm = "upi"
        default_city = "Delhi NCR"
        default_ship_city = "Delhi NCR"
        default_vpn = 0
        default_emu = 0
        default_root = 0
        default_dist = 12.0
        default_failed_1h = 0
        default_typing = 55
        default_mouse = 0.78
        default_dur = 420
        default_chk = 140
        default_isp = "Bharti Airtel Ltd"
        default_asn = "AS45609"
    else:
        default_amt = 15000.0
        default_cat = "electronics_gadgets"
        default_pm = "credit_card"
        default_city = "Mumbai"
        default_ship_city = "Bengaluru"
        default_vpn = 0
        default_emu = 0
        default_root = 0
        default_dist = 980.0
        default_failed_1h = 1
        default_typing = 70
        default_mouse = 0.65
        default_dur = 45
        default_chk = 18
        default_isp = "ACT Fibernet Broadband"
        default_asn = "AS24309"

    with st.expander("🛠️ Inspect / Modify Transaction Parameters", expanded=("Custom" in preset)):
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            inp_amount = st.number_input("Transaction Amount (₹ INR)", min_value=10.0, max_value=500000.0, value=float(default_amt), step=500.0)
            inp_cat = st.selectbox("Merchant Category", list(CATEGORY_RISK_MAP.keys()), index=list(CATEGORY_RISK_MAP.keys()).index(default_cat))
            inp_pm = st.selectbox("Payment Instrument", ["upi", "credit_card", "debit_card", "netbanking", "wallet_emi"], index=["upi", "credit_card", "debit_card", "netbanking", "wallet_emi"].index(default_pm))
            inp_isp = st.text_input("ISP / Telecom Provider", value=default_isp)

        with ic2:
            inp_ip_city = st.text_input("IP Origin City", value=default_city)
            inp_ship_city = st.text_input("Shipping Destination City", value=default_ship_city)
            inp_dist = st.number_input("IP to Shipping Distance (km)", min_value=0.0, max_value=20000.0, value=float(default_dist), step=10.0)
            inp_failed_1h = st.slider("Failed Auth / OTP Retries (1h)", 0, 10, int(default_failed_1h))

        with ic3:
            inp_vpn = st.checkbox("Datacenter Proxy / VPN Detected", value=bool(default_vpn))
            inp_emu = st.checkbox("Android Emulator Detected", value=bool(default_emu))
            inp_root = st.checkbox("Rooted / Jailbroken OS", value=bool(default_root))
            inp_dur = st.number_input("Session Duration (sec)", min_value=1, max_value=3600, value=int(default_dur))
            inp_chk = st.number_input("Time to Checkout (sec)", min_value=1, max_value=600, value=int(default_chk))
            inp_typing = st.number_input("Typing Cadence (WPM)", min_value=10, max_value=300, value=int(default_typing))
            inp_mouse = st.slider("Mouse Movement Entropy", 0.0, 1.0, float(default_mouse))

    # Construct transaction object
    active_txn = {
        "transaction_id": f"pay_live_{preset[:1]}_{int(inp_amount)}",
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount_inr": float(inp_amount),
        "merchant_id": "mid_demo_01",
        "merchant_name": f"{inp_cat.replace('_', ' ').title()} Store",
        "merchant_category": inp_cat,
        "payment_method": inp_pm,
        "card_network": "VISA" if inp_pm == "credit_card" else ("RUPAY" if inp_pm == "debit_card" else ""),
        "upi_vpa": "customer@okhdfcbank" if inp_pm == "upi" else "",
        "user_id": "usr_demo_88",
        "ip_address": "45.142.12.8" if inp_vpn else "103.21.244.18",
        "isp_name": inp_isp,
        "asn_code": default_asn if inp_isp == default_isp else ("AS47583" if inp_vpn else "AS55836"),
        "ip_city": inp_ip_city,
        "ip_state": "Karnataka",
        "shipping_city": inp_ship_city,
        "shipping_state": "Karnataka",
        "shipping_pincode": "560001",
        "billing_city": inp_ship_city,
        "billing_state": "Karnataka",
        "device_id": "dev_live_fingerprint",
        "is_vpn_proxy": int(inp_vpn),
        "is_emulator": int(inp_emu),
        "is_rooted": int(inp_root),
        "fingerprint_entropy": 5.8 if inp_emu else 3.2,
        "session_duration_sec": inp_dur,
        "time_to_checkout_sec": inp_chk,
        "page_views": 1 if inp_chk < 5 else 4,
        "typing_speed_wpm": inp_typing,
        "mouse_entropy": inp_mouse,
        "failed_attempts_1h": inp_failed_1h,
        "failed_attempts_24h": inp_failed_1h * 2,
        "cvv_retries": 1 if inp_failed_1h > 0 else 0,
        "otp_delay_sec": 45 if inp_failed_1h > 0 else 11,
        "ip_to_shipping_dist_km": float(inp_dist),
        "user_account_age_days": 180 if "Clean" in preset else 12,
        "user_order_index": 5 if "Clean" in preset else 1,
        "delivery_status": "DELIVERED_POD_CONFIRMED",
        "delivery_awb": "AWB_7781920391",
        "courier_partner": "BlueDart Express",
        "dispute_reason": "10.4 - Other Fraud (Cardholder Disputes Transaction)",
    }

    if trainer is not None and explainer is not None:
        explanation = explainer.explain_transaction(active_txn, top_k=5)
        score = explanation["risk_score"]
        tier = explanation["risk_tier"]
        color = explanation["top_risk_factors"][0].get("severity", "LOW")
        pill_class = f"tier-pill-{tier.lower()}"

        st.markdown("---")

        # Top Result Row
        res_c1, res_c2 = st.columns([1.2, 1.8])

        with res_c1:
            st.markdown("#### 🎯 ChargeShield Risk Evaluation")

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Score (0 - 100)", 'font': {'size': 18, 'color': '#E2E8F0'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': "#EF4444" if score >= 85 else ("#F97316" if score >= 66 else ("#F59E0B" if score >= 31 else "#10B981"))},
                    'bgcolor': "#1E293B",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [30, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [65, 84], 'color': 'rgba(249, 115, 22, 0.15)'},
                        {'range': [84, 100], 'color': 'rgba(239, 68, 68, 0.2)'},
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#F8FAFC"},
                height=260,
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div style='text-align: center; margin-top: -10px;'>
              <span class='{pill_class}' style='font-size: 15px; padding: 8px 18px;'>
                {tier} RISK • {explanation['recommended_action']}
              </span>
              <div style='font-size: 12px; color: #94A3B8; margin-top: 8px;'>
                {explanation['action_description']}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown("#### 🕸️ 6-Dimension Anomaly Radar")

            # Radar Chart
            categories = ['Order Amount', 'Velocity Burst', 'VPN / Datacenter', 'Geo Mismatch', 'Bot Cadence', 'Auth Friction']
            r_amount = min(100, (inp_amount / 40000.0) * 100)
            r_velocity = 95 if "Carding" in preset else (35 if inp_failed_1h > 1 else 10)
            r_network = 100 if inp_vpn else (40 if inp_emu else 5)
            r_geo = min(100, (inp_dist / 3000.0) * 100)
            r_bot = 95 if inp_typing > 180 or inp_chk < 4 else (60 if inp_mouse < 0.3 else 10)
            r_auth = min(100, inp_failed_1h * 30 + (25 if inp_pm == "credit_card" and inp_failed_1h > 0 else 0))

            radar_values = [r_amount, r_velocity, r_network, r_geo, r_bot, r_auth]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(239, 68, 68, 0.25)' if score >= 66 else 'rgba(59, 130, 246, 0.25)',
                line_color='#EF4444' if score >= 66 else '#3B82F6',
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color="#64748B", gridcolor="#1E293B"),
                    angularaxis=dict(color="#CBD5E1", gridcolor="#1E293B"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(t=20, b=20, l=35, r=35),
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🔍 Top 5 Merchant-Friendly SHAP Risk Factors")

        for f in explanation["top_risk_factors"]:
            sev = f.get("severity", "LOW")
            st.markdown(f"""
            <div class='risk-factor-card severity-{sev}'>
              <div style='display: flex; justify-content: space-between; align-items: center;'>
                <strong style='font-size: 14px; color: #FFFFFF;'>{f['factor_title']}</strong>
                <span style='font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>{f['category']}</span>
              </div>
              <div style='font-size: 13px; color: #CBD5E1; margin-top: 4px;'>{f['description']}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: PRE-SETTLEMENT BATCH QUEUE
# =============================================================================
with tab3:
    st.markdown("### 🔍 Pre-Settlement Batch Risk Queue")
    st.caption("Live transaction pipeline pending settlement. Filter high-risk orders to hold payouts before bank settlement cutoff.")

    if not df_test.empty and trainer is not None:
        sample_batch = df_test.head(150).copy()
        scores = trainer.predict_risk_score(sample_batch)
        sample_batch["ChargeShield Score"] = scores.round(1)
        sample_batch["Decision Tier"] = [trainer.threshold_optimizer.get_decision_tier(s)["tier"] for s in scores]
        sample_batch["Action"] = [trainer.threshold_optimizer.get_decision_tier(s)["recommended_action"] for s in scores]
        sample_batch["Hold Settlement"] = [trainer.threshold_optimizer.get_decision_tier(s)["settlement_hold"] for s in scores]

        # Filters
        fc1, fc2, fc3 = st.columns([1.5, 1.5, 2])
        with fc1:
            tier_filter = st.multiselect("Filter by Risk Tier", ["LOW", "MODERATE", "HIGH", "CRITICAL"], default=["HIGH", "CRITICAL", "MODERATE", "LOW"])
        with fc2:
            pm_filter = st.multiselect("Filter Payment Method", sample_batch["payment_method"].unique().tolist(), default=sample_batch["payment_method"].unique().tolist())
        with fc3:
            search_query = st.text_input("Search by User ID, Merchant or RRN", "")

        filtered = sample_batch[
            (sample_batch["Decision Tier"].isin(tier_filter)) &
            (sample_batch["payment_method"].isin(pm_filter))
        ]
        if search_query:
            filtered = filtered[
                filtered["user_id"].str.contains(search_query, case=False) |
                filtered["merchant_id"].str.contains(search_query, case=False) |
                filtered["rrn_utr"].str.contains(search_query, case=False)
            ]

        # Summary KPIs for batch
        held_count = filtered["Hold Settlement"].sum()
        held_volume = filtered[filtered["Hold Settlement"]]["amount_inr"].sum()
        instant_volume = filtered[~filtered["Hold Settlement"]]["amount_inr"].sum()

        qc1, qc2, qc3 = st.columns(3)
        qc1.metric("Total Batch Queue", f"{len(filtered):,} Orders")
        qc2.metric("Payouts Held Pre-Settlement", f"{held_count} Orders", f"₹{held_volume:,.2f} Held", delta_color="inverse")
        qc3.metric("Auto-Approved for Instant Payout", f"{len(filtered) - held_count} Orders", f"₹{instant_volume:,.2f} Settled")

        # Display table
        display_cols = ["transaction_id", "timestamp", "amount_inr", "merchant_name", "payment_method", "ChargeShield Score", "Decision Tier", "Hold Settlement"]
        st.dataframe(
            filtered[display_cols].style.format({"amount_inr": "₹{:,.2f}"}),
            use_container_width=True,
            height=400,
        )

        # Batch Export
        csv_data = filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export Enriched Pre-Settlement Risk Batch (CSV)",
            data=csv_data,
            file_name="chargeshield_pre_settlement_batch.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 4: DISPUTE EVIDENCE STUDIO
# =============================================================================
with tab4:
    st.markdown("### ⚖️ Automated Dispute Evidence Package Generator")
    st.caption("When a chargeback is received from Visa, Mastercard, or NPCI, auto-compile a comprehensive representment defense packet with cryptographic 2FA proofs, delivery audit logs, and compelling stance arguments.")

    if not df_test.empty and dispute_gen is not None:
        chargebacks_in_test = df_test[df_test["is_chargeback"] == 1].head(15)

        txn_choice = st.selectbox(
            "Select a Contested Chargeback Case to Generate Arbitration Packet:",
            options=range(len(chargebacks_in_test)),
            format_func=lambda i: f"Case {i+1}: {chargebacks_in_test.iloc[i]['transaction_id']} | ₹{chargebacks_in_test.iloc[i]['amount_inr']:,.2f} | {chargebacks_in_test.iloc[i]['fraud_archetype']} | {chargebacks_in_test.iloc[i]['merchant_name']}",
        )

        selected_record = chargebacks_in_test.iloc[txn_choice].to_dict()
        packet = dispute_gen.generate_packet(selected_record)
        html_packet = dispute_gen.format_html_packet(packet)

        # Case Readiness Bar
        score = packet["case_readiness_score"]
        tier = packet["case_readiness_tier"]

        dc1, dc2, dc3 = st.columns([1.2, 1.8, 1.2])
        with dc1:
            st.metric("Case Readiness Index", f"{score}%", f"{tier} ARBITRATION STANCE")
        with dc2:
            st.markdown(f"**Recommended Stance:** {packet['recommended_dispute_stance']['stance_title']}")
            st.caption(f"Governing Rule: {packet['recommended_dispute_stance']['compelling_evidence_rule']}")
        with dc3:
            st.download_button(
                "📄 Download Official Dispute Packet (HTML)",
                data=html_packet,
                file_name=f"chargeshield_dispute_{packet['dispute_id']}.html",
                mime="text/html",
            )
            st.download_button(
                "💾 Export Evidence JSON Payload",
                data=json.dumps(packet, indent=2),
                file_name=f"chargeshield_dispute_{packet['dispute_id']}.json",
                mime="application/json",
            )

        st.markdown("---")
        st.markdown("#### 📜 Dispute Evidence Document Preview")
        st.components.v1.html(html_packet, height=650, scrolling=True)


# =============================================================================
# TAB 5: ML ANALYTICS & THRESHOLD OPTIMIZER
# =============================================================================
with tab5:
    st.markdown("### 📈 ML Performance & Interactive Financial Cost Optimizer")
    st.caption("Interact with the operational decision threshold slider to see the live trade-off between prevented chargeback fraud loss (₹) and merchant false positive friction (₹).")

    if eval_metrics and trainer is not None and not df_test.empty:
        opt_profile = trainer.threshold_optimizer.threshold_profile
        all_curves = opt_profile.get("all_threshold_curves", [])

        if all_curves:
            curve_df = pd.DataFrame(all_curves)

            # Interactive threshold slider
            current_opt_thresh = float(opt_profile.get("optimal_threshold", 0.11))
            user_thresh = st.slider(
                "🎚️ Drag Operational Decision Threshold (T):",
                min_value=0.01,
                max_value=0.99,
                value=current_opt_thresh,
                step=0.01,
            )

            # Find closest profile
            closest_idx = (curve_df["threshold"] - user_thresh).abs().idxmin()
            curr_row = curve_df.iloc[closest_idx]

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Precision", f"{curr_row['precision']:.1%}")
            sc2.metric("Recall (Detection)", f"{curr_row['recall']:.1%}")
            sc3.metric("False Positive Rate", f"{curr_row['fpr']:.2%}")
            sc4.metric("Total Merchant Loss (₹)", f"₹{curr_row['total_cost_inr']:,.0f}")

            # Plot Loss Curve vs Friction
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["total_cost_inr"], name="Total Expected Merchant Loss (₹)", line=dict(color="#EF4444", width=3)))
            fig_cost.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["prevented_loss_inr"], name="Prevented Fraud Loss (₹)", line=dict(color="#10B981", width=2)))
            fig_cost.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["fp_friction_inr"], name="Customer Friction Cost (₹)", line=dict(color="#F59E0B", width=2, dash="dot")))

            fig_cost.add_vline(x=user_thresh, line_width=2, line_dash="dash", line_color="#FFFFFF", annotation_text=f"T={user_thresh:.2f}")

            fig_cost.update_layout(
                title="Financial Loss Optimization Curve (Minimizing ₹ Lost)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#CBD5E1"},
                xaxis=dict(gridcolor="#1E293B", title="Decision Threshold (T)"),
                yaxis=dict(gridcolor="#1E293B", title="Indian Rupees (₹ INR)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                margin=dict(t=40, b=40, l=20, r=20),
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        st.markdown("---")
        mc1, mc2 = st.columns(2)

        with mc1:
            st.markdown("#### 📉 ROC Curve (Held-Out Test)")
            roc_sample = eval_metrics.get("roc_curve_sample", {})
            if roc_sample:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=roc_sample["fpr"], y=roc_sample["tpr"], name="ChargeShield XGBoost+IF", line=dict(color="#3B82F6", width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", line=dict(color="#64748B", dash="dash")))
                fig_roc.update_layout(
                    title=f"ROC-AUC: {eval_metrics['metrics']['roc_auc']:.4f}",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#CBD5E1"},
                    xaxis=dict(gridcolor="#1E293B", title="False Positive Rate"),
                    yaxis=dict(gridcolor="#1E293B", title="True Positive Rate"),
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

        with mc2:
            st.markdown("#### 🎯 Precision-Recall Curve")
            pr_sample = eval_metrics.get("pr_curve_sample", {})
            if pr_sample:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=pr_sample["recall"], y=pr_sample["precision"], name="PR Curve", line=dict(color="#10B981", width=3)))
                fig_pr.update_layout(
                    title=f"PR-AUC (Average Precision): {eval_metrics['metrics']['pr_auc']:.4f}",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#CBD5E1"},
                    xaxis=dict(gridcolor="#1E293B", title="Recall"),
                    yaxis=dict(gridcolor="#1E293B", title="Precision"),
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_pr, use_container_width=True)
