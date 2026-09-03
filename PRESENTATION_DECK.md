# 🏆 ChargeShield AI — Hackathon Presentation Deck & Pitch Script

**Track**: AI Risk Manager  
**Project**: ChargeShield AI (Pre-Settlement Autonomous Risk Network)

---

## ⏱️ 3-Minute Live Video Demo Script

### [0:00 - 0:35] The Problem & Core Paradigm Shift
> *"Every year, digital merchants lose billions twice: first to chargeback fraud, then to ₹1,500 bank penalty fees per dispute. Most fraud solutions try to block suspicious orders at checkout—which ruins conversion rates and creates massive customer drop-off—or they alert merchants weeks after settlement when the money has already left the account.*
> 
> *ChargeShield AI introduces a new paradigm: **Pre-Settlement Interception**. We evaluate risk silently post-authorization, holding payouts before bank settlement cutoffs, while auto-compiling bank-ready legal dispute dossiers for unavoidable claims."*

### [0:35 - 1:15] Home & Interactive Unit Economics (Navigate to 'Home')
> *(Screen: Show Home Page & Scroll to the Interactive ROI Calculator)*
> *"On our Home Page, we give merchants an Interactive ROI Calculator. For a merchant processing ₹5 Crore/month with a 1.2% chargeback rate, ChargeShield AI preserves over **₹71 Lakhs in annual capital** with a verified **140x platform ROI**, while capping false positive review friction strictly under 2.5%."*

### [1:15 - 1:55] Live Lab & TreeSHAP Explainability (Navigate to 'Live Lab')
> *(Screen: Select Preset 02: High-Value Luxury Jewelry Bot Burst)*
> *"In our Live Risk Inspector, we run a hybrid ensemble of XGBoost + Unsupervised Isolation Forest across **103 Indian forensic signals**—including datacenter ASN proxies, Haversine geo-mismatch, and typing telemetry.*
> 
> *Notice our sub-10ms scoring and our **TreeSHAP Explainability**: we don't output blackbox probabilities. We break down the top 5 plain-English risk factors with severity tiers for operations teams. Developers can test the live `POST /predict` REST API payload right here."*

### [1:55 - 2:30] Pre-Settlement Terminal (Navigate to 'Queue')
> *(Screen: Show the Pre-Settlement Payout Terminal)*
> *"In the Pre-Settlement Queue, 92% of clean transactions are instantly released. Orders with risk scores above 85 are automatically held, preventing direct capital loss before bank batch execution."*

### [2:30 - 3:00] Automated Dispute Dossier (Navigate to 'Disputes' & 'Metrics')
> *(Screen: Open Disputes Studio, preview HTML Dossier, then switch to Metrics tab)*
> *"When first-party friendly fraud disputes occur, our **Dispute Arbitration Studio** compiles a bank-ready representment dossier adhering to **Visa Compelling Evidence 3.0** and **NPCI non-repudiation standards**, complete with cryptographic audit trails and courier AWB proofs.*
> 
> *Finally, our Mathematical Benchmarks show **99.22% Recall**, **0.9993 ROC-AUC**, and **3.8ms median latency** tested live on 1,000 orders. ChargeShield AI is a complete, production-ready defense system for modern commerce."*

---

## 💡 Judge Q&A Cheat Sheet (How to Answer Tough Questions)

### Q1: "How do you handle class imbalance without creating massive false positives?"
> **Answer**: *"ChargeShield AI trains XGBoost with an exact `scale_pos_weight` tuned to the empirical 8.41% positive chargeback distribution. More importantly, instead of using default $T=0.5$, our Threshold Optimizer optimizes against a real financial loss function: $\text{Loss}(T) = \text{FN} \times (\text{Amount} + \text{₹1,500 Penalty}) + \text{FP} \times (\text{Friction Cost})$. This yields $T^* = 0.1087$, which caps FPR at 2.11% while capturing 99.22% of all chargeback volume."*

### Q2: "Why combine XGBoost with Isolation Forest?"
> **Answer**: *"XGBoost excels at learning known historical fraud patterns, but fails on novel zero-day attacks. Isolation Forest operates geometrically in feature space without labels, assigning high anomaly scores to outlier coordinate geometries (e.g. coordinated bot swarms using new ISP proxies). Blending their normalized outputs provides both high precision on known fraud and resilience against emerging exploits."*

### Q3: "What is your production inference latency?"
> **Answer**: *"Our vectorized pipeline achieves a **p50 of 3.8ms** and **p95 of 8.4ms** on standard single-core execution, with batch throughput exceeding 120,000 transactions/second. This easily satisfies strict payment gateway SLAs (< 25ms)."*

### Q4: "How does your dispute generator comply with Visa CE 3.0?"
> **Answer**: *"Visa Compelling Evidence 3.0 requires merchants to prove prior non-fraudulent transaction history with matching IP, device fingerprint, or shipping address across 2 prior orders older than 120 days. Our Dispute Engine audits device IDs, IP CIDR blocks, courier AWB Proof-of-Delivery timestamps, and terms-of-service acceptance logs into a single court-ready evidence package."*
