"""Synthetic Indian transaction data generator for ChargeShield AI.

Generates highly realistic transaction streams tailored to the Indian e-commerce
and payments ecosystem (UPI, RuPay, Visa, Mastercard, NetBanking) with authentic
fraud patterns and chargeback typologies for Razorpay Buildathon.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Indian Context Reference Datasets
# ---------------------------------------------------------

INDIAN_CITIES: List[Dict[str, Any]] = [
    {"city": "Bengaluru", "state": "Karnataka", "tier": "Tier 1", "lat": 12.9716, "lon": 77.5946, "pincodes": ["560001", "560038", "560100", "560068"]},
    {"city": "Mumbai", "state": "Maharashtra", "tier": "Tier 1", "lat": 19.0760, "lon": 72.8777, "pincodes": ["400001", "400050", "400076", "400099"]},
    {"city": "Delhi NCR", "state": "Delhi", "tier": "Tier 1", "lat": 28.6139, "lon": 77.2090, "pincodes": ["110001", "110020", "110092", "122002"]},
    {"city": "Hyderabad", "state": "Telangana", "tier": "Tier 1", "lat": 17.3850, "lon": 78.4867, "pincodes": ["500001", "500081", "500034", "500032"]},
    {"city": "Chennai", "state": "Tamil Nadu", "tier": "Tier 1", "lat": 13.0827, "lon": 80.2707, "pincodes": ["600001", "600028", "600096", "600113"]},
    {"city": "Kolkata", "state": "West Bengal", "tier": "Tier 1", "lat": 22.5726, "lon": 88.3639, "pincodes": ["700001", "700020", "700091", "700156"]},
    {"city": "Pune", "state": "Maharashtra", "tier": "Tier 1", "lat": 18.5204, "lon": 73.8567, "pincodes": ["411001", "411014", "411045", "411057"]},
    {"city": "Ahmedabad", "state": "Gujarat", "tier": "Tier 1", "lat": 23.0225, "lon": 72.5714, "pincodes": ["380001", "380015", "380054", "382481"]},
    {"city": "Jaipur", "state": "Rajasthan", "tier": "Tier 2", "lat": 26.9124, "lon": 75.7873, "pincodes": ["302001", "302015", "302020", "302033"]},
    {"city": "Lucknow", "state": "Uttar Pradesh", "tier": "Tier 2", "lat": 26.8467, "lon": 80.9462, "pincodes": ["226001", "226010", "226016", "226024"]},
    {"city": "Indore", "state": "Madhya Pradesh", "tier": "Tier 2", "lat": 22.7196, "lon": 75.8577, "pincodes": ["452001", "452010", "452016", "453555"]},
    {"city": "Kochi", "state": "Kerala", "tier": "Tier 2", "lat": 9.9312, "lon": 76.2673, "pincodes": ["682001", "682016", "682030", "683101"]},
    {"city": "Chandigarh", "state": "Punjab", "tier": "Tier 2", "lat": 30.7333, "lon": 76.7794, "pincodes": ["160017", "160022", "160036", "160047"]},
    {"city": "Patna", "state": "Bihar", "tier": "Tier 2", "lat": 25.5941, "lon": 85.1376, "pincodes": ["800001", "800013", "800020", "800025"]},
    {"city": "Bhubaneswar", "state": "Odisha", "tier": "Tier 2", "lat": 20.2961, "lon": 85.8245, "pincodes": ["751001", "751010", "751024", "751030"]},
    {"city": "Guwahati", "state": "Assam", "tier": "Tier 3", "lat": 26.1445, "lon": 91.7362, "pincodes": ["781001", "781005", "781022", "781035"]},
    {"city": "Surat", "state": "Gujarat", "tier": "Tier 2", "lat": 21.1702, "lon": 72.8311, "pincodes": ["395001", "395003", "395007", "395010"]},
]

FOREIGN_PROXY_LOCATIONS: List[Dict[str, Any]] = [
    {"city": "Frankfurt", "state": "Hesse", "country": "DE", "lat": 50.1109, "lon": 8.6821, "isp": "Hostinger Datacenter", "asn": "AS47583"},
    {"city": "Ashburn", "state": "Virginia", "country": "US", "lat": 39.0438, "lon": -77.4874, "isp": "DigitalOcean LLC", "asn": "AS14061"},
    {"city": "Singapore", "state": "Central", "country": "SG", "lat": 1.3521, "lon": 103.8198, "isp": "M247 Europe Ltd", "asn": "AS9009"},
    {"city": "London", "state": "Greater London", "country": "GB", "lat": 51.5074, "lon": -0.1278, "isp": "Cloudflare WARP", "asn": "AS13335"},
]

INDIAN_ISPS: List[Dict[str, Any]] = [
    {"isp": "Reliance Jio Infocomm", "asn": "AS55836", "risk_weight": 0.05},
    {"isp": "Bharti Airtel Ltd", "asn": "AS45609", "risk_weight": 0.05},
    {"isp": "ACT Fibernet Broadband", "asn": "AS24309", "risk_weight": 0.08},
    {"isp": "Vodafone Idea Limited", "asn": "AS133694", "risk_weight": 0.07},
    {"isp": "Bharat Sanchar Nigam Ltd", "asn": "AS9829", "risk_weight": 0.09},
    {"isp": "Tata Teleservices", "asn": "AS4755", "risk_weight": 0.06},
    {"isp": "Excitel Broadband", "asn": "AS132203", "risk_weight": 0.12},
]

MERCHANT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "electronics_gadgets": {
        "mean_amount": 22500,
        "std_amount": 14000,
        "min_amount": 1200,
        "max_amount": 185000,
        "chargeback_baseline": 0.065,
        "refund_baseline": 0.08,
        "settlement_window_days": 2,
    },
    "luxury_jewelry": {
        "mean_amount": 42000,
        "std_amount": 28000,
        "min_amount": 4500,
        "max_amount": 350000,
        "chargeback_baseline": 0.085,
        "refund_baseline": 0.04,
        "settlement_window_days": 3,
    },
    "digital_goods_gaming": {
        "mean_amount": 1150,
        "std_amount": 1600,
        "min_amount": 99,
        "max_amount": 25000,
        "chargeback_baseline": 0.110,
        "refund_baseline": 0.14,
        "settlement_window_days": 1,
    },
    "fashion_apparel": {
        "mean_amount": 3400,
        "std_amount": 2900,
        "min_amount": 299,
        "max_amount": 45000,
        "chargeback_baseline": 0.045,
        "refund_baseline": 0.18,
        "settlement_window_days": 2,
    },
    "travel_airline": {
        "mean_amount": 16800,
        "std_amount": 12500,
        "min_amount": 1800,
        "max_amount": 120000,
        "chargeback_baseline": 0.075,
        "refund_baseline": 0.12,
        "settlement_window_days": 2,
    },
    "edtech_courses": {
        "mean_amount": 19500,
        "std_amount": 11000,
        "min_amount": 1499,
        "max_amount": 95000,
        "chargeback_baseline": 0.090,
        "refund_baseline": 0.10,
        "settlement_window_days": 2,
    },
    "quick_commerce_food": {
        "mean_amount": 750,
        "std_amount": 550,
        "min_amount": 99,
        "max_amount": 6500,
        "chargeback_baseline": 0.020,
        "refund_baseline": 0.06,
        "settlement_window_days": 1,
    },
}

UPI_HANDLES = ["@okhdfcbank", "@okicici", "@okaxis", "@oksbi", "@ybl", "@paytm", "@ibl", "@axl", "@barodampay", "@upi"]
CARD_NETWORKS = ["VISA", "MASTERCARD", "RUPAY", "AMEX"]
BANKS_INDIA = ["HDFC Bank", "ICICI Bank", "State Bank of India", "Axis Bank", "Kotak Mahindra Bank", "IndusInd Bank", "Federal Bank", "Punjab National Bank"]
COURIER_PARTNERS = ["Delhivery", "BlueDart Express", "Ecom Express", "Shadowfax", "XpressBees", "DTDC"]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great circle distance between two points in km."""
    r = 6371.0  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


class SyntheticTransactionGenerator:
    """Generates synthetic production-grade Indian transactions with realistic fraud vectors."""

    def __init__(
        self,
        num_transactions: int = 30000,
        num_users: int = 4000,
        num_merchants: int = 150,
        start_date: str = "2026-05-01 00:00:00",
        days: int = 90,
        target_fraud_rate: float = 0.082,
        seed: int = 42,
    ) -> None:
        self.num_transactions = num_transactions
        self.num_users = num_users
        self.num_merchants = num_merchants
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        self.days = days
        self.target_fraud_rate = target_fraud_rate
        self.seed = seed

        # Set seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        self._init_merchants()
        self._init_users()

    def _init_merchants(self) -> None:
        """Pre-computes realistic merchant profiles."""
        self.merchants: List[Dict[str, Any]] = []
        categories = list(MERCHANT_CATEGORIES.keys())
        weights = [0.20, 0.08, 0.18, 0.22, 0.12, 0.08, 0.12]

        for i in range(1, self.num_merchants + 1):
            category = np.random.choice(categories, p=weights)
            cat_cfg = MERCHANT_CATEGORIES[category]
            city_info = random.choice(INDIAN_CITIES)

            m_profile = {
                "merchant_id": f"mid_{i:04d}",
                "merchant_name": f"{category.replace('_', ' ').title()} Store #{i}",
                "merchant_category": category,
                "merchant_city": city_info["city"],
                "merchant_state": city_info["state"],
                "merchant_lat": city_info["lat"],
                "merchant_lon": city_info["lon"],
                "merchant_base_cb_rate": float(np.clip(np.random.normal(cat_cfg["chargeback_baseline"], 0.015), 0.005, 0.20)),
                "merchant_base_refund_rate": float(np.clip(np.random.normal(cat_cfg["refund_baseline"], 0.02), 0.01, 0.30)),
                "merchant_settlement_cycle": cat_cfg["settlement_window_days"],
                "merchant_account_age_months": int(np.random.randint(6, 60)),
            }
            self.merchants.append(m_profile)

    def _init_users(self) -> None:
        """Pre-computes realistic customer user profiles."""
        self.users: List[Dict[str, Any]] = []
        for i in range(1, self.num_users + 1):
            home_city = random.choice(INDIAN_CITIES)
            shipping_pincode = random.choice(home_city["pincodes"])
            vpa_prefix = f"user{i}"
            vpa_handle = random.choice(UPI_HANDLES)
            preferred_bank = random.choice(BANKS_INDIA)
            card_last4 = f"{random.randint(1000, 9999)}"
            card_network = np.random.choice(CARD_NETWORKS, p=[0.40, 0.30, 0.26, 0.04])
            default_device_id = f"dev_{uuid.uuid4().hex[:12]}"
            default_isp = np.random.choice(INDIAN_ISPS)

            u_profile = {
                "user_id": f"usr_{i:06d}",
                "user_name": f"Customer_{i}",
                "home_city": home_city["city"],
                "home_state": home_city["state"],
                "home_lat": home_city["lat"],
                "home_lon": home_city["lon"],
                "shipping_pincode": shipping_pincode,
                "billing_city": home_city["city"],
                "billing_state": home_city["state"],
                "billing_pincode": shipping_pincode,
                "upi_vpa": f"{vpa_prefix}{vpa_handle}",
                "card_id": f"card_{card_network.lower()}_{card_last4}",
                "card_network": card_network,
                "issuing_bank": preferred_bank,
                "default_device_id": default_device_id,
                "default_isp": default_isp["isp"],
                "default_asn": default_isp["asn"],
                "user_created_days_prior": int(np.random.exponential(scale=180) + 10),
                "is_chronic_friendly_fraudster": bool(np.random.rand() < 0.025),  # 2.5% chronic refund abusers
            }
            self.users.append(u_profile)

    def generate(self) -> pd.DataFrame:
        """Generates temporal transaction records with authentic distributions and fraud vectors."""
        records: List[Dict[str, Any]] = []

        # Time timeline spread over days
        total_seconds = self.days * 86400
        # Time distribution with peaks during Indian peak shopping hours (12 PM - 3 PM, 7 PM - 11 PM)
        timestamps_offsets = []
        for _ in range(self.num_transactions):
            day_offset = np.random.uniform(0, self.days)
            # Hour sampling with Indian shopping curves
            hour_p = [0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.07, 0.06, 0.05, 0.05, 0.06, 0.08, 0.09, 0.09, 0.06, 0.04, 0.02]
            hour_p_norm = np.array(hour_p) / np.sum(hour_p)
            hour = np.random.choice(range(24), p=hour_p_norm)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            timestamps_offsets.append(day_offset * 86400 + (hour * 3600 + minute * 60 + second))

        timestamps_offsets.sort()

        # Track rolling entities for realistic velocity
        recent_ip_history: Dict[str, List[datetime]] = {}
        recent_device_history: Dict[str, List[datetime]] = {}
        recent_user_history: Dict[str, List[datetime]] = {}
        user_order_counter: Dict[str, int] = {}

        # Fraud archetypes allocation
        # Target ~8.2% total fraud
        archetypes = [
            "legitimate",
            "friendly_fraud_first_party",
            "velocity_carding_bot",
            "device_ip_spoofing_vpn",
            "account_takeover_ato",
            "high_value_jewelry_bustout",
            "digital_goods_instant_refund",
        ]
        archetype_probs = [
            1.0 - self.target_fraud_rate,
            self.target_fraud_rate * 0.32,  # 32% of fraud is friendly fraud / chargeback abuse
            self.target_fraud_rate * 0.22,  # 22% carding / velocity bot
            self.target_fraud_rate * 0.18,  # 18% VPN / proxy / device mismatch
            self.target_fraud_rate * 0.12,  # 12% ATO
            self.target_fraud_rate * 0.10,  # 10% high-ticket jewelry / electronics bustout
            self.target_fraud_rate * 0.06,  # 6% digital goods instant drain
        ]
        # Normalize probabilities
        norm_probs = np.array(archetype_probs) / np.sum(archetype_probs)

        for i, offset_sec in enumerate(timestamps_offsets):
            txn_id = f"pay_{uuid.uuid4().hex[:14]}"
            rrn_utr = f"RRN{np.random.randint(100000000000, 999999999999)}"
            txn_time = self.start_date + timedelta(seconds=float(offset_sec))

            # Pick archetype
            arch = np.random.choice(archetypes, p=norm_probs)
            is_fraud = arch != "legitimate"

            # Base user and merchant selection
            user = random.choice(self.users)
            merchant = random.choice(self.merchants)
            user_order_counter[user["user_id"]] = user_order_counter.get(user["user_id"], 0) + 1

            cat_cfg = MERCHANT_CATEGORIES[merchant["merchant_category"]]
            base_mean = cat_cfg["mean_amount"]
            base_std = cat_cfg["std_amount"]

            # Initialize variables to defaults
            amount = float(np.clip(np.random.lognormal(mean=np.log(base_mean), sigma=0.6), cat_cfg["min_amount"], cat_cfg["max_amount"]))
            payment_method = np.random.choice(["upi", "credit_card", "debit_card", "netbanking", "wallet_emi"], p=[0.58, 0.24, 0.12, 0.04, 0.02])
            card_id = user["card_id"] if "card" in payment_method else ""
            card_network = user["card_network"] if "card" in payment_method else ""
            upi_vpa = user["upi_vpa"] if payment_method == "upi" else ""
            device_id = user["default_device_id"]
            isp_name = user["default_isp"]
            asn_code = user["default_asn"]
            ip_city = user["home_city"]
            ip_state = user["home_state"]
            ip_lat = user["home_lat"]
            ip_lon = user["home_lon"]
            shipping_city = user["home_city"]
            shipping_state = user["home_state"]
            shipping_pincode = user["shipping_pincode"]
            billing_city = user["billing_city"]
            billing_state = user["billing_state"]
            billing_pincode = user["billing_pincode"]

            is_vpn_proxy = False
            is_emulator = False
            is_rooted = False
            fingerprint_entropy = round(float(np.random.uniform(2.5, 4.2)), 2)
            session_duration_sec = int(np.random.gamma(shape=3.5, scale=40) + 15)
            time_to_checkout_sec = int(np.random.gamma(shape=2.5, scale=25) + 8)
            page_views = int(np.random.poisson(lam=5) + 1)
            failed_attempts_1h = 0
            failed_attempts_24h = 0
            cvv_retries = 0
            otp_delay_sec = int(np.random.uniform(4, 18))
            three_ds_status = "SUCCESS_CHALLENGE" if payment_method in ["credit_card", "debit_card"] else "NOT_APPLICABLE"
            three_ds_version = "2.2.0" if payment_method in ["credit_card", "debit_card"] else "N/A"
            user_account_age_days = user["user_created_days_prior"] + int(offset_sec / 86400)
            typing_speed_wpm = int(np.random.normal(55, 12))
            mouse_entropy = round(float(np.random.uniform(0.65, 0.95)), 2)
            screen_res = np.random.choice(["1920x1080", "1440x900", "390x844 (Mobile)", "412x915 (Mobile)", "2560x1440"])
            user_agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15" if "Mobile" in screen_res else "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            is_international_card = False
            is_international_ip = False
            delivery_status = "DELIVERED"
            delivery_awb = f"AWB_{np.random.randint(1000000000, 9999999999)}"
            courier_partner = random.choice(COURIER_PARTNERS)
            dispute_reason = "NONE"
            chargeback_stage = "NONE"
            terms_accepted_timestamp = (txn_time - timedelta(seconds=time_to_checkout_sec)).strftime("%Y-%m-%d %H:%M:%S")

            # Apply archetypal signatures for fraud / chargeback scenarios
            if arch == "friendly_fraud_first_party":
                # High ticket genuine looking purchase, but user claims unauthorized or item not received later
                amount = float(np.random.uniform(base_mean * 1.4, cat_cfg["max_amount"] * 0.95))
                # Authentic device and home IP
                session_duration_sec = int(np.random.uniform(180, 600))
                page_views = int(np.random.randint(4, 12))
                dispute_reason = np.random.choice([
                    "10.4 - Other Fraud (Cardholder Disputes Transaction)",
                    "13.1 - Merchandise/Services Not Received",
                    "13.3 - Defective / Not as Described",
                ])
                chargeback_stage = "CHARGEBACK_RECEIVED"

            elif arch == "velocity_carding_bot":
                # Machine bot testing stolen credentials rapidly
                amount = float(np.random.choice([99.0, 199.0, 499.0, 999.0, 1499.0]))
                payment_method = np.random.choice(["credit_card", "debit_card"])
                card_id = f"card_stolen_{np.random.randint(1000, 9999)}"
                card_network = np.random.choice(["VISA", "MASTERCARD"])
                session_duration_sec = int(np.random.uniform(3, 12))  # Extremely rapid
                time_to_checkout_sec = int(np.random.uniform(1, 4))
                page_views = 1
                typing_speed_wpm = int(np.random.uniform(180, 260))  # Automated script paste
                mouse_entropy = 0.05  # Linear bot movement
                failed_attempts_1h = int(np.random.randint(3, 8))
                failed_attempts_24h = int(np.random.randint(6, 18))
                cvv_retries = int(np.random.randint(2, 5))
                is_vpn_proxy = True
                proxy_loc = random.choice(FOREIGN_PROXY_LOCATIONS)
                ip_city = proxy_loc["city"]
                ip_state = proxy_loc["state"]
                ip_lat = proxy_loc["lat"]
                ip_lon = proxy_loc["lon"]
                isp_name = proxy_loc["isp"]
                asn_code = proxy_loc["asn"]
                is_international_ip = True
                dispute_reason = "10.4 - Unauthorized Transaction / Fraud Card Absent"
                chargeback_stage = "CHARGEBACK_RECEIVED"

            elif arch == "device_ip_spoofing_vpn":
                # Scammer operating over Datacenter VPN with rooted emulator
                amount = float(np.random.uniform(base_mean * 1.8, cat_cfg["max_amount"]))
                is_vpn_proxy = True
                is_emulator = bool(np.random.rand() < 0.8)
                is_rooted = bool(np.random.rand() < 0.7)
                fingerprint_entropy = round(float(np.random.uniform(4.8, 6.5)), 2)  # Inconsistent canvas/WebGL
                proxy_loc = random.choice(FOREIGN_PROXY_LOCATIONS)
                ip_city = proxy_loc["city"]
                ip_state = proxy_loc["state"]
                ip_lat = proxy_loc["lat"]
                ip_lon = proxy_loc["lon"]
                isp_name = proxy_loc["isp"]
                asn_code = proxy_loc["asn"]
                is_international_ip = True
                is_international_card = bool(np.random.rand() < 0.35)
                # Shipping to an Indian drop address mismatching home
                mismatch_city = random.choice([c for c in INDIAN_CITIES if c["city"] != user["home_city"]])
                shipping_city = mismatch_city["city"]
                shipping_state = mismatch_city["state"]
                shipping_pincode = random.choice(mismatch_city["pincodes"])
                dispute_reason = "10.4 - Counterfeit / Stolen Cardholder Data"
                chargeback_stage = "CHARGEBACK_RECEIVED"

            elif arch == "account_takeover_ato":
                # Hijacked account at odd night hours with sudden address & device change
                amount = float(np.random.uniform(base_mean * 2.2, cat_cfg["max_amount"]))
                device_id = f"dev_hijack_{uuid.uuid4().hex[:8]}"
                mismatch_city = random.choice([c for c in INDIAN_CITIES if c["city"] != user["home_city"]])
                shipping_city = mismatch_city["city"]
                shipping_state = mismatch_city["state"]
                shipping_pincode = random.choice(mismatch_city["pincodes"])
                session_duration_sec = int(np.random.uniform(45, 120))
                time_to_checkout_sec = int(np.random.uniform(15, 45))
                failed_attempts_1h = int(np.random.randint(1, 3))
                dispute_reason = "10.4 - Account Takeover / Unauthorized Order"
                chargeback_stage = "CHARGEBACK_RECEIVED"

            elif arch == "high_value_jewelry_bustout":
                # High-ticket jewelry or electronics purchase with maxed limits
                merchant = random.choice([m for m in self.merchants if m["merchant_category"] in ["luxury_jewelry", "electronics_gadgets"]])
                amount = float(np.random.uniform(45000, 195000))
                payment_method = "credit_card"
                card_id = f"card_bust_{np.random.randint(1000, 9999)}"
                card_network = "AMEX" if np.random.rand() < 0.5 else "VISA"
                cvv_retries = int(np.random.randint(1, 3))
                otp_delay_sec = int(np.random.uniform(35, 75))
                dispute_reason = "10.4 - Unrecognized Transaction / High Risk Bustout"
                chargeback_stage = "CHARGEBACK_RECEIVED"

            elif arch == "digital_goods_instant_refund":
                # Instant digital goods consumption followed by chargeback
                merchant = random.choice([m for m in self.merchants if m["merchant_category"] in ["digital_goods_gaming", "edtech_courses"]])
                amount = float(np.random.uniform(899, 14999))
                delivery_status = "INSTANT_DIGITAL_KEY_DELIVERED"
                courier_partner = "Digital Delivery API (Server-to-Server)"
                delivery_awb = f"DIGITAL_GRANT_{uuid.uuid4().hex[:10].upper()}"
                dispute_reason = "13.1 - Product Not As Described / Unfulfilled Digital Item"
                chargeback_stage = "CHARGEBACK_RECEIVED"

            # Compute geo distances in km
            ip_to_shipping_dist = haversine_distance_km(ip_lat, ip_lon, user["home_lat"], user["home_lon"])
            billing_to_shipping_dist = haversine_distance_km(user["home_lat"], user["home_lon"], ip_lat, ip_lon)

            # Generate realistic IP
            if is_vpn_proxy:
                ip_addr = f"{np.random.randint(45, 195)}.{np.random.randint(10, 240)}.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}"
            else:
                ip_addr = f"103.{np.random.randint(10, 250)}.{np.random.randint(1, 254)}.{np.random.randint(1, 254)}"

            rec = {
                "transaction_id": txn_id,
                "rrn_utr": rrn_utr,
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user["user_id"],
                "merchant_id": merchant["merchant_id"],
                "merchant_name": merchant["merchant_name"],
                "merchant_category": merchant["merchant_category"],
                "merchant_city": merchant["merchant_city"],
                "merchant_state": merchant["merchant_state"],
                "merchant_base_cb_rate": merchant["merchant_base_cb_rate"],
                "merchant_base_refund_rate": merchant["merchant_base_refund_rate"],
                "merchant_settlement_cycle": merchant["merchant_settlement_cycle"],
                "amount_inr": round(amount, 2),
                "payment_method": payment_method,
                "card_id": card_id,
                "card_network": card_network,
                "issuing_bank": user["issuing_bank"],
                "is_international_card": int(is_international_card),
                "upi_vpa": upi_vpa,
                "device_id": device_id,
                "ip_address": ip_addr,
                "isp_name": isp_name,
                "asn_code": asn_code,
                "ip_city": ip_city,
                "ip_state": ip_state,
                "is_international_ip": int(is_international_ip),
                "is_vpn_proxy": int(is_vpn_proxy),
                "is_emulator": int(is_emulator),
                "is_rooted": int(is_rooted),
                "fingerprint_entropy": fingerprint_entropy,
                "shipping_city": shipping_city,
                "shipping_state": shipping_state,
                "shipping_pincode": shipping_pincode,
                "billing_city": billing_city,
                "billing_state": billing_state,
                "billing_pincode": billing_pincode,
                "ip_to_shipping_dist_km": round(ip_to_shipping_dist, 2),
                "session_duration_sec": session_duration_sec,
                "time_to_checkout_sec": time_to_checkout_sec,
                "page_views": page_views,
                "typing_speed_wpm": typing_speed_wpm,
                "mouse_entropy": mouse_entropy,
                "failed_attempts_1h": failed_attempts_1h,
                "failed_attempts_24h": failed_attempts_24h,
                "cvv_retries": cvv_retries,
                "otp_delay_sec": otp_delay_sec,
                "three_ds_status": three_ds_status,
                "three_ds_version": three_ds_version,
                "user_account_age_days": user_account_age_days,
                "user_order_index": user_order_counter[user["user_id"]],
                "delivery_status": delivery_status,
                "delivery_awb": delivery_awb,
                "courier_partner": courier_partner,
                "dispute_reason": dispute_reason,
                "chargeback_stage": chargeback_stage,
                "terms_accepted_timestamp": terms_accepted_timestamp,
                "fraud_archetype": arch,
                "is_chargeback": int(is_fraud),
            }
            records.append(rec)

        df = pd.DataFrame(records)
        # Ensure strict temporal sort
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
