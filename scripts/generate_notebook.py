#!/usr/bin/env python3
"""Generates the evaluation and experiments Jupyter notebook for ChargeShield AI."""

import nbformat as nbf
from pathlib import Path


def create_evaluation_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown: Title & Abstract
    cells.append(nbf.v4.new_markdown_cell("""# 🛡️ ChargeShield AI: Experimental Evaluation & Model Benchmarks
### Razorpay Buildathon • AI Risk Manager Track

ChargeShield AI is a defense-only, production-grade AI risk management system designed for Indian e-commerce merchants and payment aggregators.
It operates in two coordinated phases:
1. **Pre-Settlement Interception**: Supervised XGBoost + Unsupervised Isolation Forest Hybrid scoring over 90+ transaction features.
2. **Post-Chargeback Arbitration Defense**: Automated generation of card network-compliant (Visa/Mastercard/NPCI) dispute evidence packages with quantitative readiness scoring.
"""))

    # Code: Imports & Environment Setup
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(".."))

from chargeshield.data.generator import SyntheticTransactionGenerator
from chargeshield.features.engineer import FeatureEngineer
from chargeshield.models.model_trainer import ChargeShieldModelTrainer
from chargeshield.models.threshold_optimizer import ThresholdOptimizer
from chargeshield.models.evaluator import ModelEvaluator
from chargeshield.explainability.explainer import RiskExplainer
from chargeshield.dispute.evidence_generator import DisputeEvidenceGenerator

sns.set_theme(style="darkgrid")
print("✅ ChargeShield AI modules successfully imported!")"""))

    # Markdown: Section 1 - Data Generation
    cells.append(nbf.v4.new_markdown_cell("""## 1. Synthetic Indian Payment Data Generation

Generating 30,000 realistic transactions with authentic Indian payment instruments (UPI, Cards, RuPay, NetBanking), telecom ASNs (Jio, Airtel, ACT), Indian tier 1/2/3 cities, and 6 distinct fraud archetypes:
- **Velocity / Carding Bots**
- **First-Party Friendly Fraud**
- **Datacenter VPN / Device Spoofing**
- **Account Takeover (ATO)**
- **High-Value Bustout**
- **Digital Goods Instant Drain**
"""))

    # Code: Data Generation
    cells.append(nbf.v4.new_code_cell("""generator = SyntheticTransactionGenerator(
    num_transactions=15000,
    num_users=2500,
    num_merchants=100,
    days=60,
    target_fraud_rate=0.085,
    seed=42
)
df = generator.generate()
print(f"Generated {len(df):,} transactions:")
print(f"Chargeback Rate: {df['is_chargeback'].mean():.2%} ({df['is_chargeback'].sum():,} chargebacks)")
df.head(5)"""))

    # Markdown: Section 2 - Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Engineering Pipeline (90+ Features)

Extracting amount ratios, multi-window velocity (1h, 24h, 7d), behavioral timing, network telemetry, geographic mismatches, payment instrument risk, and merchant baselines.
"""))

    # Code: Feature Engineering
    cells.append(nbf.v4.new_code_cell("""fe = FeatureEngineer()
X_features = fe.fit_transform(df)
feature_names = [c for c in X_features.columns if c not in ["is_chargeback", "transaction_id", "timestamp"]]

print(f"Total extracted features: {len(feature_names)}")
print(f"Feature categories: Amount (14), Velocity (28), Behavioral (14), Telemetry (12), Geo (10), Payment (12), Merchant (10)")
X_features[feature_names[:10]].describe()"""))

    # Markdown: Section 3 - Model Training
    cells.append(nbf.v4.new_markdown_cell("""## 3. Hybrid Model Training (XGBoost + Isolation Forest)

Strict temporal split (60% Train, 20% Validation, 20% Held-Out Test).
"""))

    # Code: Training
    cells.append(nbf.v4.new_code_cell("""trainer = ChargeShieldModelTrainer(n_estimators=200, max_depth=5, learning_rate=0.04)
train_df, val_df, test_df = trainer.split_temporal(df, train_ratio=0.6, val_ratio=0.2)

train_results = trainer.train(train_df, val_df)
print(f"Optimal Threshold: {trainer.threshold_optimizer.optimal_threshold:.4f}")"""))

    # Markdown: Section 4 - Held-Out Test Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 4. Rigorous Held-Out Test Evaluation & Financial Cost Analysis
"""))

    # Code: Evaluation
    cells.append(nbf.v4.new_code_cell("""evaluator = ModelEvaluator(trainer)
report = evaluator.evaluate_test_set(test_df)

print(evaluator.generate_markdown_report(report))"""))

    # Markdown: Section 5 - Explainability
    cells.append(nbf.v4.new_markdown_cell("""## 5. SHAP Explainability & Top 5 Plain-English Risk Factors
"""))

    # Code: Explainability
    cells.append(nbf.v4.new_code_cell("""explainer = RiskExplainer(trainer)

# Select a high-risk sample
high_risk_sample = test_df[test_df["is_chargeback"] == 1].iloc[0].to_dict()
explanation = explainer.explain_transaction(high_risk_sample, top_k=5)

print(f"Transaction ID: {explanation['transaction_id']}")
print(f"ChargeShield Score: {explanation['risk_score']} / 100 ({explanation['risk_tier']} RISK)")
print(f"Recommended Action: {explanation['recommended_action']}")
print("\\nTop 5 Merchant-Friendly Risk Factors:")
for i, factor in enumerate(explanation['top_risk_factors'], 1):
    print(f"{i}. [{factor['severity']}] {factor['factor_title']}: {factor['description']}")"""))

    # Markdown: Section 6 - Dispute Evidence Generation
    cells.append(nbf.v4.new_markdown_cell("""## 6. Automated Dispute Evidence Package Generation

Auto-compiling card network representment packages when chargebacks occur.
"""))

    # Code: Dispute
    cells.append(nbf.v4.new_code_cell("""dispute_generator = DisputeEvidenceGenerator()
packet = dispute_generator.generate_packet(high_risk_sample)

print(f"Dispute Case ID: {packet['dispute_id']}")
print(f"Case Readiness Score: {packet['case_readiness_score']}% ({packet['case_readiness_tier']} DEFENSE)")
print(f"Recommended Stance: {packet['recommended_dispute_stance']['stance_title']}")
print(f"Governing Rule: {packet['recommended_dispute_stance']['compelling_evidence_rule']}")
print(f"Core Argument: {packet['recommended_dispute_stance']['core_argument']}")"""))

    nb.cells = cells

    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_and_experiments.ipynb"

    with open(out_file, "w") as f:
        nbf.write(nb, f)

    print(f"✅ Generated notebook at {out_file}")


if __name__ == "__main__":
    create_evaluation_notebook()
