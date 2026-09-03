# 🚀 ChargeShield AI — Deployment Guide

This guide explains how to deploy **ChargeShield AI** to production:
1. **FastAPI Backend (REST API)** $\rightarrow$ **Vercel**
2. **Interactive Streamlit Studio (Dashboard)** $\rightarrow$ **Streamlit Community Cloud / Render / Railway**

---

## ⚡ Part 1: Deploying the FastAPI Backend on Vercel

The FastAPI backend is fully configured for Vercel Serverless Functions via [`api/index.py`](file:///Users/shubhamraj/Desktop/ChargeShieldAI/api/index.py) and [`vercel.json`](file:///Users/shubhamraj/Desktop/ChargeShieldAI/vercel.json).

### Method A: Deploy via GitHub (Recommended)
1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Deploy ChargeShield AI to Vercel"
   git push origin main
   ```
2. Go to **[vercel.com](https://vercel.com)** and log in with your GitHub account.
3. Click **"Add New..."** $\rightarrow$ **"Project"**.
4. Import your `ChargeShieldAI` repository.
5. In the project configuration:
   - **Framework Preset**: `Other`
   - **Root Directory**: `./`
   - Click **"Deploy"**.
6. Once deployed, you will receive a public URL like `https://chargeshield-ai.vercel.app`.
   - **Interactive API Docs (Swagger UI)**: `https://chargeshield-ai.vercel.app/docs`
   - **Real-Time Scoring**: `POST https://chargeshield-ai.vercel.app/predict`
   - **Health Check**: `GET https://chargeshield-ai.vercel.app/health`

### Method B: Deploy via Vercel CLI
1. Install Vercel CLI (if not already installed):
   ```bash
   npm i -g vercel
   ```
2. In the project root directory, run:
   ```bash
   vercel
   ```
3. Follow the CLI prompts (accept defaults). For production deployment, run:
   ```bash
   vercel --prod
   ```

---

## 🎨 Part 2: Deploying the Streamlit Studio (Interactive Dashboard)

> [!NOTE]
> **Why Streamlit needs a persistent WebSocket host**: Streamlit uses continuous WebSockets for real-time reactivity and state management. While Vercel is built for stateless REST APIs, the interactive dashboard is best hosted on **Streamlit Community Cloud** (100% Free) or **Render**.

### Option A: Streamlit Community Cloud (1-Click & 100% Free)
1. Ensure your repo is on GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New app"**.
4. Fill in:
   - **Repository**: `your-username/ChargeShieldAI`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
5. Click **"Deploy!"**.
6. You will receive a permanent public URL (e.g., `https://chargeshield-ai.streamlit.app`).

### Option B: Deploy on Render.com
1. Go to **[render.com](https://render.com)** and create a new **Web Service**.
2. Connect your GitHub repository.
3. Set the following settings:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **"Create Web Service"**.

---

## 🧪 Testing Your Deployed Vercel API

Once deployed on Vercel, test your live endpoint using `curl`:

```bash
curl -X POST "https://your-app.vercel.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_PROD_1001",
    "amount_inr": 85000.0,
    "merchant_category": "luxury_jewelry",
    "payment_method": "credit_card",
    "ip_address": "45.142.12.8",
    "isp_name": "Datacenter Hosting",
    "is_vpn_proxy": 1,
    "ip_to_shipping_dist_km": 1850.0,
    "failed_attempts_1h": 3
  }'
```
