"""Unit tests for ChargeShield Model Training and Threshold Optimizer."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from chargeshield.data.generator import SyntheticTransactionGenerator
from chargeshield.models.model_trainer import ChargeShieldModelTrainer
from chargeshield.models.threshold_optimizer import ThresholdOptimizer


@pytest.fixture
def dataset_splits():
    gen = SyntheticTransactionGenerator(
        num_transactions=300,
        num_users=40,
        num_merchants=8,
        days=20,
        target_fraud_rate=0.12,
        seed=42,
    )
    df = gen.generate()
    trainer = ChargeShieldModelTrainer()
    train_df, val_df, test_df = trainer.split_temporal(df, train_ratio=0.6, val_ratio=0.2)
    return train_df, val_df, test_df


def test_model_training_and_scoring(dataset_splits):
    train_df, val_df, test_df = dataset_splits
    trainer = ChargeShieldModelTrainer(n_estimators=30, max_depth=3)

    train_res = trainer.train(train_df, val_df)
    assert train_res["num_features"] >= 80

    # Predict risk scores on test slice
    scores = trainer.predict_risk_score(test_df)
    assert len(scores) == len(test_df)
    assert (scores >= 0.0).all() and (scores <= 100.0).all()

    # Predict single transaction
    single_res = trainer.predict_single(test_df.iloc[0].to_dict())
    assert "risk_score" in single_res
    assert "risk_tier" in single_res
    assert single_res["risk_tier"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert "recommended_action" in single_res


def test_threshold_optimizer():
    optimizer = ThresholdOptimizer(max_allowed_fpr=0.03)
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_scores = np.array([0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.85, 0.95])
    amounts = np.array([1000, 2000, 1500, 800, 1200, 3000, 2500, 4000, 25000, 45000])

    profile = optimizer.optimize(y_true, y_scores, amounts)
    assert "optimal_threshold" in profile
    assert profile["precision"] > 0
    assert profile["recall"] > 0
    assert profile["fpr"] <= 0.25
