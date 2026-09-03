#!/usr/bin/env python3
"""CLI script to generate synthetic Indian transaction data for ChargeShield AI."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chargeshield.data.generator import SyntheticTransactionGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Indian transaction dataset")
    parser.add_argument("--num-txns", type=int, default=30000, help="Number of transactions to generate")
    parser.add_argument("--num-users", type=int, default=4000, help="Number of unique user entities")
    parser.add_argument("--num-merchants", type=int, default=150, help="Number of merchants")
    parser.add_argument("--days", type=int, default=90, help="Timeline span in days")
    parser.add_argument("--fraud-rate", type=float, default=0.082, help="Target fraud / chargeback rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/raw_transactions.csv", help="Output file path")

    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Generating {args.num_txns:,} synthetic transactions spanning {args.days} days...")
    generator = SyntheticTransactionGenerator(
        num_transactions=args.num_txns,
        num_users=args.num_users,
        num_merchants=args.num_merchants,
        days=args.days,
        target_fraud_rate=args.fraud_rate,
        seed=args.seed,
    )
    df = generator.generate()

    df.to_csv(out_path, index=False)
    print(f"[+] Saved dataset to {out_path} ({len(df):,} rows, {len(df.columns)} columns)")
    print(f"    - Fraud / Chargeback cases: {df['is_chargeback'].sum():,} ({df['is_chargeback'].mean():.2%})")
    print(f"    - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"    - Unique Merchants: {df['merchant_id'].nunique()}, Unique Users: {df['user_id'].nunique()}")
    print("\nFraud Archetypes Distribution:")
    print(df["fraud_archetype"].value_counts().to_string())


if __name__ == "__main__":
    main()
