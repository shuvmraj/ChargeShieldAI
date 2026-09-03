"""Unit tests for Feature Engineering Engine (80+ features guarantee)."""

import pytest
import pandas as pd
import numpy as np
from chargeshield.data.generator import SyntheticTransactionGenerator
from chargeshield.features.engineer import FeatureEngineer


@pytest.fixture
def sample_dataset():
    gen = SyntheticTransactionGenerator(
        num_transactions=100,
        num_users=20,
        num_merchants=5,
        days=10,
        seed=42,
    )
    return gen.generate()


def test_feature_count_exceeds_eighty(sample_dataset):
    fe = FeatureEngineer()
    X_mat = fe.fit_transform(sample_dataset)

    feature_cols = [c for c in X_mat.columns if c not in ["is_chargeback", "transaction_id", "timestamp"]]

    # Verify at least 80 features requirement
    assert len(feature_cols) >= 80, f"Expected >= 80 features, but got {len(feature_cols)}"
    assert len(fe.feature_names) >= 80

    # Ensure no NaN or infinite values
    assert not X_mat.isna().any().any(), "Transformed feature matrix contains NaNs"
    assert not np.isinf(X_mat[feature_cols].values).any(), "Transformed matrix contains Inf values"


def test_single_transaction_transform(sample_dataset):
    fe = FeatureEngineer()
    fe.fit(sample_dataset)

    single_txn = {
        "transaction_id": "pay_single_001",
        "amount_inr": 12500.0,
        "merchant_id": "mid_0001",
        "merchant_category": "electronics_gadgets",
        "payment_method": "upi",
        "user_id": "usr_0001",
        "ip_address": "103.21.244.18",
        "isp_name": "Reliance Jio Infocomm",
        "asn_code": "AS55836",
        "shipping_city": "Bengaluru",
        "billing_city": "Bengaluru",
        "is_vpn_proxy": 0,
        "is_emulator": 0,
        "is_rooted": 0,
        "fingerprint_entropy": 3.1,
        "session_duration_sec": 75,
        "time_to_checkout_sec": 20,
        "page_views": 3,
        "typing_speed_wpm": 55,
        "mouse_entropy": 0.8,
        "failed_attempts_1h": 0,
        "ip_to_shipping_dist_km": 5.0,
    }

    df_out = fe.transform_single(single_txn)
    assert len(df_out) == 1
    assert "amount_inr" in df_out.columns
    assert "log_amount" in df_out.columns
    assert not df_out.isna().any().any()
