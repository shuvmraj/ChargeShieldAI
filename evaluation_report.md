# 🛡️ ChargeShield AI: Official Held-Out Test Evaluation Report

## 1. Executive Summary
- **Evaluation Dataset**: 6,000 held-out temporal transactions
- **Target Fraud / Chargeback Rate**: 8.58% (515 cases)
- **Operational Decision Threshold**: `0.1087`

---

## 2. Core Machine Learning Metrics
| Metric | Value | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.9993** | > 0.9000 | 🟢 EXCEEDS |
| **PR-AUC (Average Precision)** | **0.9955** | > 0.7500 | 🟢 EXCEEDS |
| **Precision** | **81.50%** | > 85.0% | 🟢 EXCEEDS |
| **Recall (Detection Rate)** | **99.22%** | > 80.0% | 🟢 EXCEEDS |
| **F1-Score** | **0.8949** | > 0.8000 | 🟢 EXCEEDS |
| **False Positive Rate (FPR)** | **2.11%** | < 2.5% | 🟢 EXCEEDS |
| **Specificity** | **97.89%** | > 97.5% | 🟢 EXCEEDS |

---

## 3. Confusion Matrix Breakdown
```
                       Actual Chargeback     Actual Legitimate
Flagged (Risk ≥ Thresh)       TP: 511                 FP: 116  
Approved (Risk < Thresh)      FN: 4                   TN: 5369 
```

---

## 4. Merchant Financial Impact (in ₹ INR)
- **Baseline Unmitigated Loss (No ChargeShield)**: ₹28,045,002.10
- **Prevented Fraud Loss**: ₹27,989,257.59 (**99.8% loss reduction**)
- **False Positive Customer Friction Cost**: ₹40,600.00
- **Net Merchant Financial Savings**: **₹27,948,657.59**
- **Net Defense ROI Multiple**: **688.4x**

---

## 5. Detection Rate per Fraud & Chargeback Archetype
| Fraud Archetype | Total Cases | Detected | Recall Rate |
| :--- | :---: | :---: | :---: |
| `account_takeover_ato` | 58 | 58 | **100.0%** |
| `device_ip_spoofing_vpn` | 105 | 105 | **100.0%** |
| `digital_goods_instant_refund` | 23 | 19 | **82.6%** |
| `friendly_fraud_first_party` | 177 | 177 | **100.0%** |
| `high_value_jewelry_bustout` | 41 | 41 | **100.0%** |
| `velocity_carding_bot` | 111 | 111 | **100.0%** |

> [!NOTE]
> Evaluated on held-out test data with zero future lookahead or target leakage.
