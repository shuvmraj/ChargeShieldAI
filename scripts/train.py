#!/usr/bin/env python3
"""CLI script to train ChargeShield AI XGBoost + Isolation Forest hybrid model."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from chargeshield.data.generator import SyntheticTransactionGenerator
from chargeshield.models.model_trainer import ChargeShieldModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChargeShield AI Models")
    parser.add_argument("--data", type=str, default="data/raw_transactions.csv", help="Input dataset path")
    parser.add_argument("--artifacts-dir", type=str, default="models/artifacts", help="Directory to save model artifacts")
    parser.add_argument("--n-estimators", type=int, default=300, help="XGBoost n_estimators")
    parser.add_argument("--max-depth", type=int, default=5, help="XGBoost tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.04, help="Learning rate")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[*] Dataset not found at {data_path}. Auto-generating 30,000 synthetic transactions...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        gen = SyntheticTransactionGenerator(num_transactions=30000, seed=42)
        df = gen.generate()
        df.to_csv(data_path, index=False)
        print(f"[+] Generated and saved to {data_path}")
    else:
        print(f"[*] Loading transaction dataset from {data_path}...")
        df = pd.read_csv(data_path)

    trainer = ChargeShieldModelTrainer(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    print(f"[*] Performing strict temporal split (60% Train, 20% Val, 20% Held-Out Test)...")
    train_df, val_df, test_df = trainer.split_temporal(df, train_ratio=0.60, val_ratio=0.20)
    print(f"    - Train:      {len(train_df):,} txns ({train_df['is_chargeback'].mean():.2%} chargebacks)")
    print(f"    - Validation: {len(val_df):,} txns ({val_df['is_chargeback'].mean():.2%} chargebacks)")
    print(f"    - Test:       {len(test_df):,} txns ({test_df['is_chargeback'].mean():.2%} chargebacks)")

    # Save test set for clean standalone evaluation
    test_path = Path("data/test_transactions.csv")
    test_df.to_csv(test_path, index=False)
    print(f"[+] Saved held-out test split to {test_path}")

    # Train model
    trainer.train(train_df, val_df)

    # Save artifacts
    trainer.save_artifacts(args.artifacts_dir)
    print(f"\n✨ ChargeShield AI training successfully completed! Artifacts saved to '{args.artifacts_dir}'")


if __name__ == "__main__":
    main()
