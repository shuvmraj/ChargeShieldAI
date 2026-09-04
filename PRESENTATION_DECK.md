# 🏆 ChargeShield AI — Complete Presentation & Demo Playbook

**Track**: AI Risk Manager  
**Project**: ChargeShield AI (Pre-Settlement Autonomous Risk Network)  
**Live Demo URL**: [http://localhost:8501](http://localhost:8501) | **Backend Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)  
**Pre-Recorded Voiceovers**: Available in `presentation_audio/` (`.m4a` files)

---

## 🎬 Quick Setup Before Recording

1. **Start the Platform**:
   ```bash
   ./run.sh all
   ```
2. **Open the Dashboard**: Go to [http://localhost:8501](http://localhost:8501) in your browser (Google Chrome or Safari). Set browser zoom to **100%** (`Cmd + 0`).
3. **Screen Recorder (Mac)**: Press **`Cmd + Shift + 5`**, choose *"Record Selected Portion"* or *"Record Entire Screen"*, and choose your browser window.
4. **Audio Option**:
   - **Option A (Hands-Free)**: Play [`presentation_audio/full_pitch_narration.m4a`](presentation_audio/full_pitch_narration.m4a) in the background while following the cursor actions below.
   - **Option B (Your Voice)**: Read the simple human scripts below naturally while following the on-screen actions.

---

## ⏱️ Screen-by-Screen 3-Minute Video Timeline

```
[0:00 - 0:35] Scene 1: Home Screen — The Problem & Big Idea
[0:35 - 1:10] Scene 2: Home Screen — Live ROI Calculator (Making the Business Case)
[1:10 - 1:55] Scene 3: Live Lab Tab — Catching a Bot Attack in Real Time with Plain English
[1:55 - 2:25] Scene 4: Queue Tab — The Pre-Settlement Payout Gate (Auto-Freeze)
[2:25 - 3:00] Scene 5: Disputes & Metrics Tabs — 1-Click Visa Dossier & Hard Math
```

---

### 📍 Scene 1: The Problem & The Core Paradigm Shift
* **Timestamp**: `0:00 - 0:35` (35 seconds)
* **Where to be**: **Home** tab (Top navigation bar: click **"Home"**).
* **What to do on screen**:
  1. Keep the camera at the top hero section: *"Eliminate Chargeback Losses Pre-Settlement"*.
  2. Slowly move your mouse across the 4 purple metric cards at the top:
     - **Detection Rate: 99.22%**
     - **False Positive Rate: 2.11%**
     - **Capital Preserved: ₹2.79 Cr**
     - **Telemetry Signals: 103**
  3. Scroll down slightly to show the *"Why Traditional Fraud Detection Fails"* comparison cards.

* **What to say (Natural Voiceover Script)**:
  > *"Every year, digital merchants lose billions twice: first to chargeback fraud, then to ₹1,500 bank penalty fees per dispute.*
  >
  > *Today's fraud tools force a painful tradeoff: if you block orders at checkout, you frustrate genuine buyers and kill conversion rates. But if you wait until the bank notifies you, the money has already left your account.*
  >
  > *ChargeShield AI introduces a new paradigm: **Pre-Settlement Interception**. We evaluate risk silently post-authorization, holding payouts before bank settlement cutoffs, while auto-compiling bank-ready legal dispute dossiers for unavoidable claims."*

---

### 📍 Scene 2: Unit Economics & Live ROI Calculator
* **Timestamp**: `0:35 - 1:10` (35 seconds)
* **Where to be**: **Home** tab — scroll down to **Section 01.4: Interactive Merchant ROI Calculator**.
* **What to do on screen**:
  1. Hover over the left card: *"Adjust Merchant Volume Parameters"*.
  2. Drag the **Monthly Processing Volume** slider to **₹5.0 Crores**.
  3. Drag the **Chargeback Rate** slider to **1.2%**.
  4. Point your cursor to the green results on the right side:
     - Point at: **₹71.3 Lakhs Annual Capital Preserved**.
     - Point at: **140x Net Defense ROI**.
     - Highlight that customer friction is strictly kept below 2.5%.

* **What to say (Natural Voiceover Script)**:
  > *"On our Home Page, we give merchants an interactive ROI Calculator.*
  >
  > *Take a typical Indian brand processing ₹5 Crore a month with a 1.2% chargeback rate. Without protection, they lose nearly ₹86 Lakhs a year between lost inventory and bank penalties.*
  >
  > *With ChargeShield AI, that merchant preserves over **₹71 Lakhs in annual capital** with a verified **140x platform ROI**, while capping false positive review friction strictly under 2.5%."*

---

### 📍 Scene 3: Live Risk Inspector & Plain-English SHAP
* **Timestamp**: `1:10 - 1:55` (45 seconds)
* **Where to be**: Click the **"Live Lab"** button in the top navigation bar.
* **What to do on screen**:
  1. In the dropdown **"SELECT ATTACK PRESET OR CUSTOM VECTOR:"**, select:
     👉 **`PRESET 02: High-Value Luxury Jewelry Bot Burst (Stolen Carding Script • Critical)`**
  2. Notice the screen update automatically:
     - Point your mouse at the big **Risk Score Gauge** showing **Critical Risk (~94 - 98)**.
     - Hover over the **Red Alert Box**: *"Action: BLOCK_DEFENSE_DISPUTE_READY (Settlement Payout Freeze Active)"*.
  3. Scroll down to the **"Top 5 Plain-English Risk Factors"** section:
     - Point to Factor #1: **Proxy / Datacenter VPN Detected (Hostinger Datacenter)**.
     - Point to Factor #2: **Geographic Mismatch (Frankfurt IP vs Jaipur Delivery, 6,200 km)**.
     - Point to Factor #3: **Automated Bot Cadence (230 WPM typing, 2-second checkout)**.
  4. Scroll down slightly to show the **Developer API Sandbox** (`POST /predict` JSON payload returning in 6.8ms).

* **What to say (Natural Voiceover Script)**:
  > *"In our Live Risk Inspector, we run a hybrid ensemble of XGBoost and Unsupervised Isolation Forest across **103 Indian forensic signals**—including datacenter ASN proxies, Haversine geo-mismatch, and typing telemetry.*
  >
  > *Let's load Preset 02: a high-value jewelry bot attack. Instantly, our system scores it as Critical Risk with a mandatory pre-settlement hold.*
  >
  > *Notice our sub-10ms scoring and our **TreeSHAP Explainability**: we don't output blackbox probabilities. We break down the top 5 plain-English risk factors with severity tiers for operations teams. Developers can test the live `POST /predict` REST API payload right here."*

---

### 📍 Scene 4: Pre-Settlement Payout Terminal (Auto-Hold Gate)
* **Timestamp**: `1:55 - 2:25` (30 seconds)
* **Where to be**: Click the **"Queue"** button in the top navigation bar.
* **What to do on screen**:
  1. Show the 3 top summary cards:
     - **Batch Queue Volume: 200 Txns**
     - **Payouts Held Pre-Settlement: ~17 Orders (Red Card)**
     - **Auto-Released for Payout: ~183 Orders (Green Card)**
  2. In the dropdown **"Filter Payout Decision"**, switch from *"ALL TRANSACTIONS"* to:
     👉 **`HELD PAYOUT ONLY (Score ≥ 85)`**
  3. Scroll smoothly through the filtered table to show that all high-risk items are safely flagged with **HOLD PAYOUT** and ready for dispute packaging.

* **What to say (Natural Voiceover Script)**:
  > *"Next is the Pre-Settlement Queue—the operational heartbeat of ChargeShield.*
  >
  > *Here, 92% of genuine, clean transactions are instantly cleared for same-day payout so merchants keep their working capital flowing.*
  >
  > *Meanwhile, orders scoring above 85 are automatically held before the bank batch execution cut-off. The merchant saves the cash before it leaves the bank."*

---

### 📍 Scene 5: Automated Dispute Dossier & Mathematical Proof
* **Timestamp**: `2:25 - 3:00` (35 seconds)
* **Where to be**: Click **"Disputes"**, then click **"Metrics"** for the finale.
* **What to do on screen**:
  1. On **Disputes**:
     - Point at the **Case Readiness Score (85-92%)**.
     - Scroll through the official preview of the **Dispute Evidence Dossier**: show the Visa CE 3.0 compliance badge, courier tracking AWB, and IP audit trail.
  2. In the top nav, click **"Metrics"**:
     - Move the **Operational Decision Threshold (T)** slider slightly between `0.08` and `0.15` to show the **Total Expected Loss Curve**.
     - Point out the benchmark badges: **99.22% Recall** and **0.9993 ROC-AUC**.

* **What to say (Natural Voiceover Script)**:
  > *"When first-party friendly fraud disputes occur, our **Dispute Arbitration Studio** compiles a bank-ready representment dossier adhering to **Visa Compelling Evidence 3.0** and **NPCI non-repudiation standards**, complete with cryptographic audit trails and courier AWB proofs.*
  >
  > *Finally, our Mathematical Benchmarks show **99.22% Recall**, **0.9993 ROC-AUC**, and **3.8ms median latency** tested live on 1,000 orders. ChargeShield AI is a complete, production-ready defense system for modern commerce."

---

## ⚡ 1-Minute Elevator Pitch Version (For Lightning Demos)

If a judge or interviewer asks for a quick 60-second walkthrough:
1. **[0:00 - 0:15] Open Home**: *"ChargeShield AI solves chargebacks by shifting fraud detection from checkout to pre-settlement. We catch 99.2% of chargebacks before the bank payout clears, saving Indian merchants ₹2.79 Crores in our test benchmark."*
2. **[0:15 - 0:35] Switch to Live Lab (Preset 02)**: *"We combine XGBoost with Isolation Forest over 103 signals. Watch this bot attack: sub-10ms scoring, automated settlement freeze, and exact plain-English explanations showing the datacenter proxy and 6,200 km geo-mismatch."*
3. **[0:35 - 0:50] Switch to Queue**: *"Clean orders settle instantly; dangerous orders are frozen before bank settlement."*
4. **[0:50 - 1:00] Switch to Disputes**: *"For unavoidable friendly fraud, we auto-generate Visa Compelling Evidence 3.0 packets with a 90%+ case readiness score. It's ready to deploy today."*

---

## 🎙️ Pre-Recorded Voiceover Audio Tracks

All voiceover clips are generated and ready in the project directory:

| Track Name | Location | Duration |
| :--- | :--- | :--- |
| **Full Continuous Pitch** | [`presentation_audio/full_pitch_narration.m4a`](presentation_audio/full_pitch_narration.m4a) | 2 min 45 sec |
| **01. Problem & Paradigm** | [`presentation_audio/01_problem_intro.m4a`](presentation_audio/01_problem_intro.m4a) | 35 sec |
| **02. ROI Calculator** | [`presentation_audio/02_home_unit_economics.m4a`](presentation_audio/02_home_unit_economics.m4a) | 35 sec |
| **03. Live Lab & SHAP** | [`presentation_audio/03_live_lab_explainability.m4a`](presentation_audio/03_live_lab_explainability.m4a) | 45 sec |
| **04. Settlement Queue** | [`presentation_audio/04_settlement_terminal.m4a`](presentation_audio/04_settlement_terminal.m4a) | 30 sec |
| **05. Disputes & Metrics** | [`presentation_audio/05_disputes_and_conclusion.m4a`](presentation_audio/05_disputes_and_conclusion.m4a) | 35 sec |

---

## 💡 Judge Q&A: Human Explanations

### Q1: *"Why pre-settlement instead of blocking at checkout?"*
> **Simple Answer**: *"Blocking at checkout adds friction and turns away good customers—checkout drops cost merchants more than fraud. Pre-settlement happens during the payment gateway's standard settlement clearing window (T+1 or T+2). The customer has a great checkout experience, and the merchant protects their money before the bank payout leaves."*

### Q2: *"How do you keep false positives low while catching 99% of fraud?"*
> **Simple Answer**: *"Most models use a default 50% cutoff. We built a Financial Loss Optimizer that calculates the exact cost of a missed fraud (order value + ₹1,500 bank fine) versus the friction cost of a false alarm (₹350). The mathematical sweet spot is threshold T = 0.1087, which catches 99.2% of fraud while keeping false alarms strictly below 2.11%."*

### Q3: *"Why use both XGBoost and Isolation Forest?"*
> **Simple Answer**: *"XGBoost is great at catching known fraud patterns it has seen before. But fraud rings invent new tricks every week. Isolation Forest is unsupervised—it spots weird geometric anomalies even if it has never seen that specific attack pattern before. Blending them gives merchants the best of both worlds."*

### Q4: *"What is Visa Compelling Evidence 3.0?"*
> **Simple Answer**: *"Visa CE 3.0 allows merchants to automatically win friendly fraud disputes if they can prove the customer previously made 2 undisputed purchases with matching device fingerprints or IP ranges. Our Dispute Studio audits those records automatically and compiles a ready-to-file legal PDF package in seconds."*
0