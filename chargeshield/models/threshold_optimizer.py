"""Threshold optimization and decision tier engine for ChargeShield AI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve


@dataclass
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    total_cost_inr: float
    prevented_loss_inr: float
    false_positive_friction_inr: float
    confusion_matrix: Dict[str, int]


class ThresholdOptimizer:
    """Optimizes risk decision thresholds balancing recall against FPR and merchant friction."""

    def __init__(
        self,
        chargeback_penalty_fee: float = 1500.0,
        false_positive_friction_cost: float = 350.0,
        max_allowed_fpr: float = 0.025,
    ) -> None:
        self.chargeback_penalty_fee = chargeback_penalty_fee
        self.false_positive_friction_cost = false_positive_friction_cost
        self.max_allowed_fpr = max_allowed_fpr
        self.optimal_threshold: float = 0.50
        self.threshold_profile: Dict[str, Any] = {}
        self.tier_cutoffs: Dict[str, float] = {
            "low_risk_max": 30.0,
            "moderate_risk_max": 65.0,
            "high_risk_max": 84.0,
            "critical_risk_min": 85.0,
        }

    def optimize(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        amounts: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Finds optimal operational threshold optimizing recall while strictly capping FPR."""
        if amounts is None:
            amounts = np.ones_like(y_true) * 8500.0

        thresholds = np.linspace(0.01, 0.99, 150)
        best_cost = float("inf")
        best_threshold_for_cost = 0.50
        best_f1 = 0.0
        best_threshold_for_f1 = 0.50
        best_threshold_under_fpr = 0.50
        max_recall_under_fpr = 0.0

        all_profiles = []

        total_actual_loss_unmitigated = float(np.sum(y_true * (amounts + self.chargeback_penalty_fee)))

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            # Financial calculations
            # FN is fraud let through -> merchant loses transaction amount + ₹1500 penalty
            fn_mask = (y_true == 1) & (y_pred == 0)
            fn_loss = float(np.sum(amounts[fn_mask] + self.chargeback_penalty_fee))

            # TP is fraud stopped -> saved loss
            tp_mask = (y_true == 1) & (y_pred == 1)
            prevented_loss = float(np.sum(amounts[tp_mask] + self.chargeback_penalty_fee))

            # FP is good user falsely flagged -> customer friction / churn cost
            fp_cost = float(fp * self.false_positive_friction_cost)

            total_merchant_cost = fn_loss + fp_cost

            profile = {
                "threshold": round(float(t), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "fpr": round(float(fpr), 4),
                "total_cost_inr": round(total_merchant_cost, 2),
                "prevented_loss_inr": round(prevented_loss, 2),
                "fp_friction_inr": round(fp_cost, 2),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
            all_profiles.append(profile)

            # Check financial minimum
            if total_merchant_cost < best_cost:
                best_cost = total_merchant_cost
                best_threshold_for_cost = t

            # Check best F1
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_for_f1 = t

            # Check best recall under strict FPR cap
            if fpr <= self.max_allowed_fpr and recall > max_recall_under_fpr:
                max_recall_under_fpr = recall
                best_threshold_under_fpr = t

        # Choose the threshold that protects FPR while minimizing financial damage
        selected_thresh = best_threshold_for_cost if best_threshold_for_cost >= best_threshold_under_fpr else best_threshold_under_fpr
        self.optimal_threshold = float(selected_thresh)

        # Build final profile
        y_opt = (y_scores >= self.optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_opt, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_opt = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr_opt = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tp_mask = (y_true == 1) & (y_opt == 1)
        prevented_loss = float(np.sum(amounts[tp_mask] + self.chargeback_penalty_fee))
        fp_cost = float(fp * self.false_positive_friction_cost)
        net_savings = prevented_loss - fp_cost

        self.threshold_profile = {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1_opt, 4),
            "fpr": round(fpr_opt, 4),
            "unmitigated_loss_inr": round(total_actual_loss_unmitigated, 2),
            "mitigated_loss_inr": round(best_cost, 2),
            "prevented_fraud_loss_inr": round(prevented_loss, 2),
            "false_positive_friction_inr": round(fp_cost, 2),
            "net_financial_savings_inr": round(net_savings, 2),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "all_threshold_curves": all_profiles,
        }
        return self.threshold_profile

    def get_decision_tier(self, risk_score: float) -> Dict[str, Any]:
        """Maps a 0-100 ChargeShield Risk Score into an actionable defense decision."""
        if risk_score < self.tier_cutoffs["low_risk_max"]:
            return {
                "tier": "LOW",
                "risk_label": "Low Risk",
                "recommended_action": "APPROVE_INSTANT_SETTLEMENT",
                "settlement_hold": False,
                "badge_color": "#10B981",  # Emerald Green
                "action_description": "Transaction verified safe. Route to instant settlement pipeline.",
            }
        elif risk_score <= self.tier_cutoffs["moderate_risk_max"]:
            return {
                "tier": "MODERATE",
                "risk_label": "Moderate Risk",
                "recommended_action": "STANDARD_T2_SETTLEMENT",
                "settlement_hold": False,
                "badge_color": "#F59E0B",  # Amber / Yellow
                "action_description": "Normal behavioral variance. Release under standard T+2 settlement cycle.",
            }
        elif risk_score <= self.tier_cutoffs["high_risk_max"]:
            return {
                "tier": "HIGH",
                "risk_label": "High Risk",
                "recommended_action": "HOLD_SETTLEMENT_STEP_UP_AUTH",
                "settlement_hold": True,
                "badge_color": "#F97316",  # Orange
                "action_description": "Significant risk anomalies detected. Hold settlement pending merchant proof of delivery or step-up verification.",
            }
        else:
            return {
                "tier": "CRITICAL",
                "risk_label": "Critical Risk",
                "recommended_action": "BLOCK_DEFENSE_DISPUTE_READY",
                "settlement_hold": True,
                "badge_color": "#EF4444",  # Crimson Red
                "action_description": "Severe fraud pattern detected. Intercept payout and auto-generate dispute evidence package.",
            }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves threshold configuration and optimization profile to JSON."""
        data = {
            "optimal_threshold": self.optimal_threshold,
            "chargeback_penalty_fee": self.chargeback_penalty_fee,
            "false_positive_friction_cost": self.false_positive_friction_cost,
            "max_allowed_fpr": self.max_allowed_fpr,
            "tier_cutoffs": self.tier_cutoffs,
            "profile": self.threshold_profile,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> ThresholdOptimizer:
        """Loads threshold optimizer from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        opt = cls(
            chargeback_penalty_fee=data.get("chargeback_penalty_fee", 1500.0),
            false_positive_friction_cost=data.get("false_positive_friction_cost", 350.0),
            max_allowed_fpr=data.get("max_allowed_fpr", 0.025),
        )
        opt.optimal_threshold = data.get("optimal_threshold", 0.50)
        opt.tier_cutoffs = data.get("tier_cutoffs", opt.tier_cutoffs)
        opt.threshold_profile = data.get("profile", {})
        return opt
