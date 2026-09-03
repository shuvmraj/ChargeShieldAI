#!/usr/bin/env python3
"""CLI script to run honest held-out evaluation for ChargeShield AI."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from chargeshield.models.evaluator import ModelEvaluator
from chargeshield.models.model_trainer import ChargeShieldModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChargeShield AI on Held-Out Test Data")
    parser.add_argument("--test-data", type=str, default="data/test_transactions.csv", help="Test dataset path")
    parser.add_argument("--artifacts-dir", type=str, default="models/artifacts", help="Model artifacts directory")
    parser.add_argument("--output-report", type=str, default="evaluation_report.md", help="Output markdown report path")
    parser.add_argument("--output-json", type=str, default="models/artifacts/evaluation_metrics.json", help="Output JSON metrics path")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"[!] Test dataset not found at {test_path}. Please run train.py first.")
        sys.exit(1)

    print(f"[*] Loading model artifacts from {args.artifacts_dir}...")
    trainer = ChargeShieldModelTrainer.load_artifacts(args.artifacts_dir)

    print(f"[*] Loading held-out test data from {test_path}...")
    df_test = pd.read_csv(test_path)

    print(f"[*] Evaluating {len(df_test):,} transactions on held-out test split...")
    evaluator = ModelEvaluator(trainer)
    report = evaluator.evaluate_test_set(df_test)

    # Save JSON report
    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_content = evaluator.generate_markdown_report(report)
    with open(args.output_report, "w") as f:
        f.write(md_content)

    print(f"[+] Saved evaluation metrics to {out_json_path}")
    print(f"[+] Saved full report to {args.output_report}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()
