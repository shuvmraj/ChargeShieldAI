"""Unit tests for RiskExplainer module."""

import pytest
import pandas as pd
from chargeshield.data.generator import SyntheticTransactionGenerator
from chargeshield.explainability.explainer import RiskExplainer
from chargeshield.models.model_trainer import ChargeShieldModelTrainer


@pytest.fixture
def trained_system():
    gen = SyntheticTransactionGenerator(num_transactions=200, seed=42)
    df = gen.generate()
    trainer = ChargeShieldModelTrainer(n_estimators=30, max_depth=3)
    train_df, val_df, _ = trainer.split_temporal(df, 0.7, 0.3)
    trainer.train(train_df, val_df)
    explainer = RiskExplainer(trainer)
    return trainer, explainer, df


def test_explain_transaction_top_factors(trained_system):
    trainer, explainer, df = trained_system
    sample_txn = df.iloc[0].to_dict()

    explanation = explainer.explain_transaction(sample_txn, top_k=5)

    assert "risk_score" in explanation
    assert "risk_tier" in explanation
    assert "top_risk_factors" in explanation
    assert len(explanation["top_risk_factors"]) <= 5
    assert len(explanation["top_risk_factors"]) > 0

    first_factor = explanation["top_risk_factors"][0]
    assert "factor_title" in first_factor
    assert "description" in first_factor
    assert "category" in first_factor
    assert "severity" in first_factor
