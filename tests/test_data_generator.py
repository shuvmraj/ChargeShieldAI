"""Unit tests for synthetic Indian transaction data generator."""

import pytest
import pandas as pd
import numpy as np
from chargeshield.data.generator import SyntheticTransactionGenerator, haversine_distance_km


def test_haversine_distance():
    # Distance between Bengaluru (12.9716, 77.5946) and Mumbai (19.0760, 72.8777) is ~840 km
    dist = haversine_distance_km(12.9716, 77.5946, 19.0760, 72.8777)
    assert 800 < dist < 900
    assert haversine_distance_km(12.0, 77.0, 12.0, 77.0) == 0.0


def test_synthetic_data_generator():
    gen = SyntheticTransactionGenerator(
        num_transactions=150,
        num_users=30,
        num_merchants=10,
        days=15,
        target_fraud_rate=0.09,
        seed=42,
    )
    df = gen.generate()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 150
    assert "transaction_id" in df.columns
    assert "amount_inr" in df.columns
    assert "is_chargeback" in df.columns
    assert "fraud_archetype" in df.columns
    assert "payment_method" in df.columns

    # Check temporal ordering
    assert df["timestamp"].is_monotonic_increasing

    # Check realistic values
    assert (df["amount_inr"] > 0).all()
    assert df["is_chargeback"].isin([0, 1]).all()
    assert df["is_chargeback"].sum() > 0  # Contains positive chargeback cases
