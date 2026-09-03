"""Model training and hybrid risk scoring pipeline for ChargeShield AI.

Trains an XGBoost supervised risk classifier alongside an Isolation Forest
unsupervised anomaly detector, combining them into the calibrated ChargeShield Risk Score.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from chargeshield.features.engineer import FeatureEngineer
from chargeshield.models.threshold_optimizer import ThresholdOptimizer


class ChargeShieldModelTrainer:
    """Trains, calibrates, and ensembles XGBoost + Isolation Forest risk engines."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.04,
        contamination: float = 0.08,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state

        self.feature_engineer = FeatureEngineer()
        self.xgb_model: Optional[XGBClassifier] = None
        self.iso_forest: Optional[IsolationForest] = None
        self.iso_scaler: Optional[StandardScaler] = None
        self.threshold_optimizer = ThresholdOptimizer()

        self.feature_names: List[str] = []
        self.anomaly_feature_names: List[str] = [
            "ip_to_shipping_dist_km",
            "fingerprint_entropy",
            "session_duration_sec",
            "time_to_checkout_sec",
            "typing_speed_wpm",
            "mouse_entropy",
            "auth_friction_index",
            "amount_to_cat_avg_ratio",
            "telemetry_anomaly_score",
            "ip_risk_score",
        ]

    def split_temporal(
        self, df: pd.DataFrame, train_ratio: float = 0.60, val_ratio: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits transactions strictly chronologically into Train, Validation, and Test sets."""
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        return train_df, val_df, test_df

    def train(
        self,
        df_train_raw: pd.DataFrame,
        df_val_raw: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Trains feature engineer, XGBoost, Isolation Forest, and optimizes decision thresholds."""
        print("[*] Fitting Feature Engineering Engine on Training split...")
        X_train_df = self.feature_engineer.fit_transform(df_train_raw)
        y_train = X_train_df["is_chargeback"].values
        X_train = X_train_df.drop(columns=["is_chargeback"])
        self.feature_names = list(X_train.columns)

        print("[*] Transforming Validation split...")
        X_val_df = self.feature_engineer.transform(df_val_raw)
        y_val = X_val_df["is_chargeback"].values
        X_val = X_val_df.drop(columns=["is_chargeback"])[self.feature_names]

        # Class imbalance calculation
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        scale_pos_weight = float(num_neg / max(1, num_pos)) * 0.85
        print(f"    - Training class balance: {num_neg:,} legitimate vs {num_pos:,} chargeback (scale_pos_weight={scale_pos_weight:.2f})")

        # 1. Train Supervised XGBoost Classifier
        print("[*] Training XGBoost Risk Classifier...")
        self.xgb_model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # 2. Train Unsupervised Isolation Forest on Telemetry & Behavioral Anomaly Signals
        print("[*] Training Isolation Forest for Zero-Day Telemetry Anomaly Detection...")
        avail_anomaly_cols = [c for c in self.anomaly_feature_names if c in self.feature_names]
        self.iso_scaler = StandardScaler()
        X_train_anomaly = self.iso_scaler.fit_transform(X_train[avail_anomaly_cols])

        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.iso_forest.fit(X_train_anomaly)

        # 3. Predict on Validation Set and Optimize Thresholds
        print("[*] Optimizing Operational Thresholds on Validation set...")
        val_scores = self.predict_risk_score(df_val_raw) / 100.0
        val_amounts = df_val_raw["amount_inr"].values
        threshold_profile = self.threshold_optimizer.optimize(y_val, val_scores, val_amounts)

        print(f"[+] Optimization Complete! Optimal Threshold: {threshold_profile['optimal_threshold']:.4f}")
        print(f"    - Validation Precision: {threshold_profile['precision']:.2%}")
        print(f"    - Validation Recall:    {threshold_profile['recall']:.2%}")
        print(f"    - Validation F1:        {threshold_profile['f1_score']:.4f}")
        print(f"    - Validation FPR:       {threshold_profile['fpr']:.2%}")
        print(f"    - Net Saved Loss:       ₹{threshold_profile['net_financial_savings_inr']:,.2f}")

        return {
            "num_features": len(self.feature_names),
            "scale_pos_weight": scale_pos_weight,
            "threshold_profile": threshold_profile,
        }

    def predict_components(self, df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes supervised XGBoost probabilities, Isolation Forest anomaly scores, and rule penalties."""
        X_df = self.feature_engineer.transform(df_raw)
        X_mat = X_df[self.feature_names]

        # 1. Supervised XGBoost Probability
        p_xgb = self.xgb_model.predict_proba(X_mat)[:, 1]

        # 2. Unsupervised Isolation Forest Score (normalized 0 to 1, where 1 is highest anomaly)
        avail_anomaly_cols = [c for c in self.anomaly_feature_names if c in self.feature_names]
        X_anomaly = self.iso_scaler.transform(X_mat[avail_anomaly_cols])
        # decision_function returns negative for anomalies, positive for inliers
        raw_iso = self.iso_forest.decision_function(X_anomaly)
        # Invert and scale to [0, 1]
        s_iso = 1.0 / (1.0 + np.exp(raw_iso * 5.0))

        # 3. Rule Penalties (critical red flags)
        rule_penalties = np.zeros(len(df_raw))
        # Impossible geo distance + fast checkout
        rule_penalties += (X_mat.get("geo_velocity_flag", 0) == 1).astype(float) * 0.40
        # Datacenter VPN + Bot-like behavior
        rule_penalties += ((X_mat.get("is_datacenter_asn", 0) == 1) & (X_mat.get("is_bot_like_behavior", 0) == 1)).astype(float) * 0.45
        # High failed attempts in 1h
        rule_penalties += (X_mat.get("failed_attempts_1h", 0) >= 3).astype(float) * 0.30
        rule_penalties = np.clip(rule_penalties, 0.0, 1.0)

        return p_xgb, s_iso, rule_penalties

    def predict_risk_score(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Computes composite 0-100 ChargeShield Risk Score with adaptive zero-day weighting."""
        p_xgb, s_iso, s_rule = self.predict_components(df_raw)
        # Dynamic Anomaly Scaling: Elevate Isolation Forest weighting on severe anomalies to catch zero-day attacks
        iso_boost = np.where(s_iso > 0.70, 0.28, 0.18)
        xgb_weight = 0.90 - iso_boost
        composite = (xgb_weight * p_xgb) + (iso_boost * s_iso) + (0.10 * s_rule)
        risk_score = np.clip(composite * 100.0, 0.0, 100.0)
        return risk_score

    def predict_single(self, txn_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Performs real-time scoring and decision routing with trusted customer friction bypass."""
        df_single = pd.DataFrame([txn_dict])
        score = float(self.predict_risk_score(df_single)[0])
        p_xgb, s_iso, s_rule = self.predict_components(df_single)

        decision = self.threshold_optimizer.get_decision_tier(
            risk_score=score,
            user_account_age_days=int(txn_dict.get("user_account_age_days", 0) or 0),
            user_order_index=int(txn_dict.get("user_order_index", 1) or 1),
            has_delivery_pod=bool(txn_dict.get("delivery_status") == "DELIVERED_POD_CONFIRMED"),
        )
        confidence = float(np.abs(score - 50.0) / 50.0)

        return {
            "transaction_id": txn_dict.get("transaction_id", "N/A"),
            "risk_score": round(score, 1),
            "risk_tier": decision["tier"],
            "risk_label": decision["risk_label"],
            "recommended_action": decision["recommended_action"],
            "settlement_hold": decision["settlement_hold"],
            "badge_color": decision["badge_color"],
            "action_description": decision["action_description"],
            "friction_bypassed": decision.get("friction_bypassed", False),
            "confidence": round(confidence, 2),
            "components": {
                "xgboost_prob": round(float(p_xgb[0]), 4),
                "isolation_anomaly_score": round(float(s_iso[0]), 4),
                "rule_penalty_score": round(float(s_rule[0]), 4),
            },
        }

    def save_artifacts(self, artifact_dir: Union[str, Path]) -> None:
        """Saves all model components, scalers, and metadata."""
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.xgb_model, out_dir / "xgboost_model.joblib")
        joblib.dump(self.iso_forest, out_dir / "isolation_forest.joblib")
        joblib.dump(self.iso_scaler, out_dir / "iso_scaler.joblib")
        joblib.dump(self.feature_engineer, out_dir / "feature_engineer.joblib")

        self.feature_engineer.save_metadata(out_dir / "feature_meta.json")
        self.threshold_optimizer.save(out_dir / "thresholds.json")

        meta = {
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "anomaly_features": self.anomaly_feature_names,
            "optimal_threshold": self.threshold_optimizer.optimal_threshold,
        }
        with open(out_dir / "model_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[+] All ChargeShield model artifacts successfully saved to {out_dir}")

    @classmethod
    def load_artifacts(cls, artifact_dir: Union[str, Path]) -> ChargeShieldModelTrainer:
        """Loads fitted model trainer from saved artifacts."""
        in_dir = Path(artifact_dir)
        trainer = cls()
        trainer.xgb_model = joblib.load(in_dir / "xgboost_model.joblib")
        trainer.iso_forest = joblib.load(in_dir / "isolation_forest.joblib")
        trainer.iso_scaler = joblib.load(in_dir / "iso_scaler.joblib")
        trainer.feature_engineer = joblib.load(in_dir / "feature_engineer.joblib")
        trainer.threshold_optimizer = ThresholdOptimizer.load(in_dir / "thresholds.json")

        with open(in_dir / "model_config.json", "r") as f:
            meta = json.load(f)
        trainer.feature_names = meta["feature_names"]
        trainer.anomaly_feature_names = meta.get("anomaly_features", trainer.anomaly_feature_names)
        return trainer
