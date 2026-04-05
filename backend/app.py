import os
import joblib
import numpy as np
import pandas as pd
import math
import logging
import threading
import json
import time
import re
import sys
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
from web3 import Web3
from flask_socketio import SocketIO, emit
from utils.shap_explainer import FEATURE_MEANINGS, explain_prediction, load_explainer as load_shap_explainer
from utils.time_model import update_wallet_score
from config import Config
from database.db import init_db, SessionLocal
import database.crud as crud
from database.models_db import Transaction, Wallet
from graph.graph_builder import GraphBuilder
from graph.risk_engine import RiskPropagationEngine
from graph.graph_api import graph_bp, init_graph_api
import google.generativeai as genai  # OR import openai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # <--- NEW

base_dir = os.path.abspath(os.path.dirname(__file__))
frontend_static_dir = os.path.join(base_dir, '../frontend/static')


app = Flask(__name__, template_folder=frontend_static_dir, static_folder=frontend_static_dir)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

genai.configure(api_key=Config.GOOGLE_API_KEY)

logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
app.logger.handlers = logging.getLogger().handlers
app.logger.setLevel(getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

ETH_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
TX_HASH_RE = re.compile(r"^0x[a-fA-F0-9]{64}$")
PLACEHOLDER_TX_HASH_RE = re.compile(r"^0x([a-fA-F0-9])\1{63}$")

GRAPH_BOOTSTRAP_LIMIT = int(os.getenv("GRAPH_BOOTSTRAP_LIMIT", "5000"))
GRAPH_PROPAGATION_DEPTH = int(os.getenv("GRAPH_PROPAGATION_DEPTH", "3"))

graph_builder = GraphBuilder(
    funding_threshold=int(os.getenv("GRAPH_FUNDING_THRESHOLD", "3")),
    related_threshold=int(os.getenv("GRAPH_RELATED_THRESHOLD", "5")),
    burst_window_seconds=int(os.getenv("GRAPH_BURST_WINDOW_SECONDS", "300")),
    burst_threshold=int(os.getenv("GRAPH_BURST_THRESHOLD", "8")),
)
risk_engine = RiskPropagationEngine(
    graph_builder=graph_builder,
    alpha=float(os.getenv("GRAPH_ALPHA", "0.65")),
    hop_decay=float(os.getenv("GRAPH_HOP_DECAY", "0.55")),
    max_depth=GRAPH_PROPAGATION_DEPTH,
)

with app.app_context():
    try:
        app.logger.info("Creating database tables if not exist...")
        init_db()
        app.logger.info("Database initialized!")
    except Exception as e:
        app.logger.exception("DB init failed: %s", e)

    try:
        bootstrap_stats = graph_builder.bootstrap_from_database(SessionLocal, limit=GRAPH_BOOTSTRAP_LIMIT)
        app.logger.info("Graph bootstrap complete: %s", bootstrap_stats)

        seed_wallets = [
            row["address"]
            for row in graph_builder.get_base_risk_wallets(limit=25, min_score=0.3)
        ]
        if seed_wallets:
            seeded = risk_engine.propagate_from_sources(seed_wallets, depth=GRAPH_PROPAGATION_DEPTH)
            app.logger.info(
                "Initial graph propagation complete (sources=%d, affected=%d)",
                len(seeded.get("sources", [])),
                len(seeded.get("scores", {})),
            )
    except Exception as graph_bootstrap_exc:
        app.logger.exception("Graph bootstrap failed: %s", graph_bootstrap_exc)

init_graph_api(graph_builder, risk_engine, SessionLocal, logger=logger)
app.register_blueprint(graph_bp)

# -----------------------------------------------------------------------------
# DEFINITION REQUIRED FOR JOBLIB LOADING (DO NOT MODIFY)
# -----------------------------------------------------------------------------
class ProductionXGBoostFraudDetector:
    def __init__(self, model, feature_names, threshold=0.5):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold

    def predict_risk_score(self, tx_features: dict) -> dict:
        df = pd.DataFrame([tx_features])
        
        # Feature Engineering
        df['Value_to_Fee_Ratio'] = df['Transaction_Value'] / (df['Transaction_Fees'].replace(0, 1e-8) + 1e-8)
        df['Gas_Efficiency'] = df['Transaction_Value'] / (df['Gas_Price'].replace(0, 1e-8) + 1e-8)
        
        if 'Final_Balance' not in df.columns: df['Final_Balance'] = df.get('Wallet_Balance', 0)
        if 'BMax_BMin_per_NT' not in df.columns: df['BMax_BMin_per_NT'] = 0.0

        df = df.fillna(0)
        for col in self.feature_names:
            if col not in df.columns: df[col] = 0.0
                
        X = df[self.feature_names]

        try:
            prob_fraud = float(self.model.predict_proba(X)[:, 1][0])
        except Exception as e:
            return {"error": f"Prediction failed: {e}", "risk_score": 1}

        if prob_fraud <= 0.01: risk_score = 1
        else:
            risk_score = math.ceil(prob_fraud * 10)
            if risk_score > 10: risk_score = 10

        is_fraud = 1 if prob_fraud >= self.threshold else 0

        return {
            "fraud_probability": round(prob_fraud, 4),
            "risk_score": int(risk_score),
            "is_fraud_label": is_fraud,
            "threshold_used": self.threshold
        }

class XGBoostFraudDetector(ProductionXGBoostFraudDetector):
    pass


# Support legacy pickles that reference detector classes under __main__.
_main_module = sys.modules.get("__main__")
if _main_module is not None:
    if not hasattr(_main_module, "ProductionXGBoostFraudDetector"):
        setattr(_main_module, "ProductionXGBoostFraudDetector", ProductionXGBoostFraudDetector)
    if not hasattr(_main_module, "XGBoostFraudDetector"):
        setattr(_main_module, "XGBoostFraudDetector", XGBoostFraudDetector)

# --- Model Loading ---
MODEL_PATH = os.path.abspath(os.path.join(base_dir, Config.MODEL_PATH))
model = None
detector = None
model_feature_order = []

def load_model():
    global model, detector, model_feature_order
    try:
        detector = joblib.load(MODEL_PATH)
        model = detector.model if hasattr(detector, 'model') else detector

        if hasattr(detector, 'feature_names') and detector.feature_names:
            model_feature_order = list(detector.feature_names)
        elif hasattr(model, 'feature_names_in_'):
            model_feature_order = list(model.feature_names_in_)
        else:
            model_feature_order = []

        try:
            load_shap_explainer(model, feature_order=model_feature_order)
        except Exception as shap_exc:
            logger.warning("SHAP explainer not initialized: %s", shap_exc)

        try:
            if hasattr(model, 'feature_names_in_'):
                logger.info(f"Model expects features: {list(model.feature_names_in_)}")
        except Exception:
            pass
        logger.info(f"ML Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f" Error: Model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# --- WEB3 PROXY SETUP (Required for Live Search) ---
proxy_w3 = Web3(Web3.HTTPProvider(Config.SEPOLIA_RPC_URL))

# --- Helpers ---
def tx_to_front(tx_row):
    if not tx_row: return None
    tx_hash = normalize_tx_hash(getattr(tx_row, "tx_hash", None))
    if not is_valid_tx_hash(tx_hash):
        return None
    ts = None
    try: ts = int(tx_row.timestamp.timestamp())
    except: ts = None

    payload = tx_row.raw_payload if isinstance(tx_row.raw_payload, dict) else {}
    fraud_probability = float(tx_row.risk_score or 0.0)
    temporal_normalized = float(tx_row.wallet_trust_score or 0.0)

    risk_score_10 = payload.get("risk_score") if isinstance(payload, dict) else None
    try:
        risk_score_10 = int(risk_score_10) if risk_score_10 is not None else int(math.ceil(fraud_probability * 10))
    except Exception:
        risk_score_10 = int(math.ceil(fraud_probability * 10)) if fraud_probability > 0 else 1
    risk_score_10 = max(1, min(10, risk_score_10))

    temporal_score = payload.get("temporal_score") if isinstance(payload, dict) else None
    if temporal_score is None:
        temporal_score = round(temporal_normalized * 10, 3)

    shap_explanation = payload.get("shap_explanation") if isinstance(payload, dict) else None
    transaction_shap_explanation = (
        payload.get("transaction_shap_explanation") if isinstance(payload, dict) else None
    ) or shap_explanation
    wallet_shap_explanation = payload.get("wallet_shap_explanation") if isinstance(payload, dict) else None
    shap_summary = ""
    if isinstance(transaction_shap_explanation, dict):
        shap_summary = str(transaction_shap_explanation.get("summary") or "").strip()
    elif isinstance(transaction_shap_explanation, str):
        shap_summary = transaction_shap_explanation.strip()
    rule_explanation = (
        shap_summary
        or (payload.get("rule_explanation") if isinstance(payload, dict) else None)
    )

    return {
        "id": tx_hash,
        "transaction_hash": tx_hash,
        "from": (tx_row.from_address or "").lower(),
        "to": (tx_row.to_address or "").lower(),
        "wallet_address": (tx_row.from_address or "").lower(),
        "value": float(tx_row.amount_eth or 0.0),
        "riskScore": fraud_probability,
        "fraud_probability": fraud_probability,
        "risk_score": risk_score_10,
        "walletScore": temporal_normalized,
        "temporal_score_normalized": temporal_normalized,
        "temporal_score": float(temporal_score or 0.0),
        "explanation": rule_explanation,
        "shap_explanation": transaction_shap_explanation,
        "transaction_shap_explanation": transaction_shap_explanation,
        "wallet_shap_explanation": wallet_shap_explanation,
        "rule_explanation": rule_explanation,
        "timestamp": ts,
        "status": tx_row.status,
        "saved_to_chain": bool(tx_row.saved_to_chain or False),
        "onchain_record_txhash": tx_row.onchain_record_txhash,
        "raw_transaction": tx_row.raw_payload or (tx_row.metadata_json or {}),
    }

def wallet_to_front(wallet_row):
    if not wallet_row: return None
    metadata = wallet_row.metadata_json if isinstance(wallet_row.metadata_json, dict) else {}
    wallet_shap_explanation = metadata.get("wallet_shap_explanation")
    if wallet_shap_explanation is None:
        try:
            wallet_shap_explanation = graph_builder.get_wallet_shap(wallet_row.address)
        except Exception:
            wallet_shap_explanation = None
    return {
        "address": (wallet_row.address or "").lower(),
        "first_seen": int(wallet_row.first_seen.timestamp()) if wallet_row.first_seen else None,
        "last_seen": int(wallet_row.last_seen.timestamp()) if wallet_row.last_seen else None,
        "age_days": wallet_row.age_days,
        "score": float(wallet_row.trust_score or 0.0) if wallet_row.trust_score is not None else None,
        "avg_risk": float(wallet_row.avg_risk or 0.0) if wallet_row.avg_risk is not None else None,
        "labels": wallet_row.labels or {},
        "metadata": metadata,
        "wallet_shap_explanation": wallet_shap_explanation,
    }

def get_risk_level(risk_score):
    if risk_score is None: return "Unknown"
    if risk_score >= 0.8: return "Very High"
    elif risk_score >= 0.6: return "High"
    elif risk_score >= 0.4: return "Medium"
    elif risk_score >= 0.2: return "Low"
    else: return "Very Low"

def get_trust_level(trust_score):
    if trust_score is None: return "Unknown"
    if trust_score >= 0.8: return "Very High"
    elif trust_score >= 0.6: return "High"
    elif trust_score >= 0.4: return "Medium"
    elif trust_score >= 0.2: return "Low"
    else: return "Very Low"


def build_shap_analysis_text(shap_explanation: dict, risk_score: int) -> str:
    if not isinstance(shap_explanation, dict):
        return "SHAP explanation unavailable."

    summary = str(shap_explanation.get("summary") or "").strip()
    top = shap_explanation.get("top_features") if isinstance(shap_explanation.get("top_features"), list) else []

    def _parse_impact(value):
        try:
            return float(str(value).replace("+", "").strip())
        except Exception:
            return 0.0

    features = []
    for item in top[:5]:
        if not isinstance(item, dict):
            continue
        meaning = str(item.get("meaning") or item.get("feature") or "Unknown feature").strip()
        impact_raw = item.get("impact") or "0.0000"
        impact_val = _parse_impact(impact_raw)
        direction = item.get("direction")
        if direction not in {"increases_risk", "reduces_risk"}:
            direction = "increases_risk" if impact_val >= 0 else "reduces_risk"
        features.append(
            {
                "meaning": meaning,
                "impact_raw": str(impact_raw),
                "impact_val": impact_val,
                "direction": direction,
            }
        )

    positive = [f for f in features if f["direction"] == "increases_risk"]
    negative = [f for f in features if f["direction"] == "reduces_risk"]

    net_top_impact = round(sum(f["impact_val"] for f in features), 4)
    positive_sum = round(sum(abs(f["impact_val"]) for f in positive), 4)
    negative_sum = round(sum(abs(f["impact_val"]) for f in negative), 4)

    severity = "Critical" if risk_score >= 7 else "High" if risk_score >= 5 else "Medium" if risk_score >= 3 else "Low"
    base_value = shap_explanation.get("base_value")
    base_text = ""
    try:
        if base_value is not None:
            base_text = f" The model baseline prior to feature effects is {float(base_value):.4f}."
    except Exception:
        base_text = ""

    if features:
        ranked_text = "; ".join(
            [
                f"{f['meaning']} ({f['impact_raw']}, {'increases' if f['direction'] == 'increases_risk' else 'decreases'} risk)"
                for f in features
            ]
        )
    else:
        ranked_text = "No feature-level SHAP attributions were returned."

    amplifier_text = (
        ", ".join([f["meaning"] for f in positive[:3]])
        if positive else "no dominant risk amplifiers"
    )
    mitigator_text = (
        ", ".join([f["meaning"] for f in negative[:3]])
        if negative else "no strong mitigators"
    )

    default_summary = (
        f"Top attributions combine to a net impact of {net_top_impact:+.4f}, "
        f"with +{positive_sum:.4f} risk pressure and -{negative_sum:.4f} mitigation pressure."
    )

    summary_text = summary or default_summary

    return (
        f"{severity} risk profile (model score {risk_score}/10). "
        f"{summary_text}{base_text} "
        f"Detailed SHAP decomposition: {ranked_text}. "
        f"Primary risk amplifiers: {amplifier_text}; primary mitigators: {mitigator_text}. "
        f"This narrative is generated directly from SHAP values and directional contribution magnitudes, "
        f"not from static rule heuristics."
    )


def normalize_wallet_address(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    if not v.startswith("0x"):
        v = f"0x{v}"
    return v


def is_valid_wallet_address(value):
    if value is None:
        return False
    return bool(ETH_ADDRESS_RE.match(str(value).strip()))


def normalize_tx_hash(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    if not v.startswith("0x"):
        v = f"0x{v}"
    return v


def is_valid_tx_hash(value):
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if not TX_HASH_RE.match(normalized):
        return False
    if PLACEHOLDER_TX_HASH_RE.match(normalized):
        return False
    return True


def probability_to_risk_score(probability: float) -> int:
    p = max(0.0, min(1.0, float(probability)))
    if p < 0.05: return 1
    elif p < 0.15: return 2
    elif p < 0.25: return 3
    elif p < 0.35: return 4
    elif p < 0.45: return 5
    elif p < 0.55: return 6
    elif p < 0.65: return 7
    elif p < 0.75: return 8
    elif p < 0.85: return 9
    else: return 10

def extract_features_from_tx(raw_tx):
    defaults = {
        "Transaction_Value": 0.0, "Transaction_Fees": 0.0,
        "Number_of_Inputs": 0.0, "Number_of_Outputs": 0.0,
        "Gas_Price": 0.0, "Wallet_Age_Days": 0.0, "Wallet_Balance": 0.0,
        "Transaction_Velocity": 0.0, "Exchange_Rate": 0.0,
        "Final_Balance": 0.0, "BMax_BMin_per_NT": 0.0
    }
    if not raw_tx:
        return defaults.copy()

    alias_map = {
        "transaction_value": "Transaction_Value",
        "value": "Transaction_Value",
        "transaction_fees": "Transaction_Fees",
        "fee": "Transaction_Fees",
        "number_of_inputs": "Number_of_Inputs",
        "inputs": "Number_of_Inputs",
        "number_of_outputs": "Number_of_Outputs",
        "outputs": "Number_of_Outputs",
        "gas_price": "Gas_Price",
        "wallet_age_days": "Wallet_Age_Days",
        "wallet_balance": "Wallet_Balance",
        "transaction_velocity": "Transaction_Velocity",
        "exchange_rate": "Exchange_Rate",
        "final_balance": "Final_Balance",
        "bmax_bmin_per_nt": "BMax_BMin_per_NT",
    }

    normalized = {}
    for key, value in raw_tx.items():
        nk = str(key).strip().lower().replace(" ", "_")
        normalized[nk] = value

    f = {}
    for k in defaults.keys():
        if k in raw_tx:
            try: f[k] = float(raw_tx.get(k) or 0.0)
            except: f[k] = defaults[k]
        else:
            lk = k.lower()
            mapped_key = alias_map.get(lk, k)
            candidate = None
            if lk in normalized:
                candidate = normalized.get(lk)
            else:
                for alias, target in alias_map.items():
                    if target == mapped_key and alias in normalized:
                        candidate = normalized.get(alias)
                        break

            if candidate is not None:
                try: f[k] = float(candidate or 0.0)
                except: f[k] = defaults[k]
            else:
                f[k] = defaults[k]

    return f


def _engineer_fallback_features(raw_features_dict):
    tx_value = float(raw_features_dict.get("Transaction_Value", 0.0) or 0.0)
    tx_fees = float(raw_features_dict.get("Transaction_Fees", 0.0) or 0.0)
    gas_price = float(raw_features_dict.get("Gas_Price", 0.0) or 0.0)
    num_inputs = float(raw_features_dict.get("Number_of_Inputs", 0.0) or 0.0)
    num_outputs = float(raw_features_dict.get("Number_of_Outputs", 0.0) or 0.0)
    balance = float(raw_features_dict.get("Wallet_Balance", 0.0) or 0.0)
    age_days = float(raw_features_dict.get("Wallet_Age_Days", 0.0) or 0.0)
    velocity = float(raw_features_dict.get("Transaction_Velocity", 0.0) or 0.0)
    volatility = float(raw_features_dict.get("BMax_BMin_per_NT", 0.0) or 0.0)
    eps = 1e-8

    engineered = dict(raw_features_dict)
    engineered["Value_to_Fee_Ratio"] = tx_value / (tx_fees + eps)
    engineered["Gas_Efficiency"] = tx_value / (gas_price + eps)
    engineered["Input_Output_Ratio"] = num_inputs / (num_outputs + eps)
    engineered["Balance_Utilization"] = tx_value / (balance + eps)
    engineered["Tx_Frequency_Score"] = velocity / (age_days + 1.0)
    engineered["Value_Velocity_Interaction"] = tx_value * velocity
    engineered["Volatility_Age_Interaction"] = volatility * np.log1p(age_days)
    engineered["Gas_Complexity"] = gas_price * num_inputs
    engineered["Log_Transaction_Value"] = np.log1p(tx_value)
    engineered["Log_Wallet_Balance"] = np.log1p(balance)
    engineered["Log_Wallet_Age"] = np.log1p(age_days)
    engineered["Is_Young_Wallet"] = int(age_days < 30)
    engineered["Is_High_Velocity"] = int(velocity > 5.0)
    return engineered


def build_model_ready_frame(raw_features_dict):
    base = extract_features_from_tx(raw_features_dict)

    engineered = dict(base)
    try:
        if detector and hasattr(detector, "_engineer_features"):
            engineered = detector._engineer_features(dict(base))
        else:
            engineered = _engineer_fallback_features(base)
    except Exception:
        engineered = _engineer_fallback_features(base)

    df = pd.DataFrame([engineered])
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    expected_order = list(model_feature_order or [])
    if expected_order:
        for col in expected_order:
            if col not in df.columns:
                df[col] = 0.0
        df = df[expected_order]

    return df, base, engineered


def build_complete_feature_vector(raw_features_dict):
    # Backward-compatible wrapper kept for older callers.
    frame, _, _ = build_model_ready_frame(raw_features_dict)
    arr = frame.to_numpy(dtype=float)[0].tolist()
    return arr, list(frame.columns)

# ---------------------------
# API ROUTES
# ---------------------------

@app.route('/')
def serve_index(): return send_from_directory(frontend_static_dir, 'index.html')

@app.route('/shield')
def serve_shield_page(): return send_from_directory(frontend_static_dir, 'activity.html')

@app.route('/activity')
def serve_activity_page(): return redirect('/shield', code=302)

@app.route('/live')
def serve_live_page(): return send_from_directory(frontend_static_dir, 'live.html')

@app.route('/graph')
def serve_graph_page(): return send_from_directory(frontend_static_dir, 'graph.html')

@app.route('/<path:path>')
def serve_static_files(path):
    try: return send_from_directory(frontend_static_dir, path)
    except: return send_from_directory(frontend_static_dir, 'index.html')


@app.route('/health', methods=['GET'])
def health_check():
    model_loaded = bool(detector is not None and model is not None)
    return jsonify({
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "graph_ready": bool(graph_builder is not None and risk_engine is not None),
        "timestamp": int(time.time()),
    }), 200 if model_loaded else 503


# --- 1. DB ROUTES (FETCHING) ---

@app.route('/transactions', methods=['GET'])
def api_get_transactions():
    try:
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        min_risk = request.args.get('min_risk', None)
        from_addr = request.args.get('from_address', None)
        to_addr = request.args.get('to_address', None)
        tx_hash = request.args.get('tx_hash', None)
        sort = request.args.get('sort', 'time')
        predicted_only_raw = request.args.get('predicted_only', '1')
        predicted_only = str(predicted_only_raw).strip().lower() not in {'0', 'false', 'no', 'off'}

        if from_addr is not None:
            from_addr = normalize_wallet_address(from_addr)
            if from_addr and not is_valid_wallet_address(from_addr):
                return jsonify({"error": "Invalid from_address format"}), 400
        if to_addr is not None:
            to_addr = normalize_wallet_address(to_addr)
            if to_addr and not is_valid_wallet_address(to_addr):
                return jsonify({"error": "Invalid to_address format"}), 400
        if tx_hash is not None:
            tx_hash = normalize_tx_hash(tx_hash)
            if tx_hash and not is_valid_tx_hash(tx_hash):
                return jsonify({"error": "Invalid tx_hash format"}), 400

        session = SessionLocal()
        q = session.query(Transaction)
        if predicted_only:
            q = q.filter(Transaction.risk_score.isnot(None))
            q = q.filter(Transaction.wallet_trust_score.isnot(None))
        if min_risk is not None:
            try: q = q.filter(Transaction.risk_score >= float(min_risk))
            except: pass
        if from_addr: q = q.filter(Transaction.from_address == from_addr.lower())
        if to_addr: q = q.filter(Transaction.to_address == to_addr.lower())
        if tx_hash: q = q.filter(Transaction.tx_hash == tx_hash)
        if sort == 'risk': q = q.order_by(Transaction.risk_score.desc())
        else: q = q.order_by(Transaction.timestamp.desc())
        rows = q.offset(offset).limit(min(limit * 3, 3000)).all()
        session.close()

        front_rows = []
        for row in rows:
            front = tx_to_front(row)
            if not front:
                continue
            front_rows.append(front)
            if len(front_rows) >= limit:
                break

        response = jsonify(front_rows)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        app.logger.exception("Error in /transactions: %s", e)
        return jsonify({"error": "internal server error"}), 500

@app.route('/wallets', methods=['GET'])
def get_wallets():
    try:
        limit = int(request.args.get('limit', 100))
        predicted_only_raw = request.args.get('predicted_only', '1')
        predicted_only = str(predicted_only_raw).strip().lower() not in {'0', 'false', 'no', 'off'}
        session = SessionLocal()
        q = session.query(Wallet)
        if predicted_only:
            q = q.filter(Wallet.trust_score.isnot(None))
        rows = q.order_by(Wallet.last_seen.desc()).limit(limit).all()
        session.close()
        response = jsonify([wallet_to_front(w) for w in rows])
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 2. ORACLE INGESTION (FIXED 405 ERROR) ---

@app.route('/transactions', methods=['POST'])
def api_add_transaction():
    """
    Endpoint for Oracle to push processed transactions.
    Saves to DB and emits WebSocket event to frontend.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        session = SessionLocal()
        try:
            tx_row = crud.create_transaction(session, data)
            
            # Format for Frontend
            front_data = tx_to_front(tx_row)

            # Graph update + risk propagation (non-fatal to core persistence path).
            graph_payload = front_data or {
                "transaction_hash": tx_row.tx_hash,
                "from": tx_row.from_address,
                "to": tx_row.to_address,
                "value": tx_row.amount_eth,
                "fraud_probability": tx_row.risk_score,
                "wallet_trust_score": tx_row.wallet_trust_score,
                "timestamp": int(tx_row.timestamp.timestamp()) if tx_row.timestamp else None,
                "transaction_shap_explanation": (
                    tx_row.raw_payload.get("transaction_shap_explanation")
                    if isinstance(tx_row.raw_payload, dict)
                    else None
                ),
                "wallet_shap_explanation": (
                    tx_row.raw_payload.get("wallet_shap_explanation")
                    if isinstance(tx_row.raw_payload, dict)
                    else None
                ),
            }

            if isinstance(tx_row.raw_payload, dict):
                raw_payload = tx_row.raw_payload
                graph_payload["transaction_shap_explanation"] = (
                    graph_payload.get("transaction_shap_explanation")
                    or graph_payload.get("shap_explanation")
                    or raw_payload.get("transaction_shap_explanation")
                    or raw_payload.get("shap_explanation")
                )
                graph_payload["wallet_shap_explanation"] = (
                    graph_payload.get("wallet_shap_explanation")
                    or raw_payload.get("wallet_shap_explanation")
                )
                if raw_payload.get("temporal_state") is not None:
                    graph_payload["temporal_state"] = raw_payload.get("temporal_state")

            graph_status = {"updated": False, "affected_wallets": 0}
            try:
                graph_builder.add_transaction(graph_payload, session=session, persist_edges=True)
                propagation = risk_engine.update_after_transaction(
                    graph_payload,
                    session=session,
                    depth=GRAPH_PROPAGATION_DEPTH,
                )

                source_wallet = (
                    graph_payload.get("from")
                    or graph_payload.get("from_address")
                    or tx_row.from_address
                )
                wallet_shap = None
                if source_wallet:
                    try:
                        wallet_shap = graph_builder.get_wallet_shap(source_wallet)
                    except Exception:
                        wallet_shap = None

                tx_raw_payload = tx_row.raw_payload if isinstance(tx_row.raw_payload, dict) else {}
                if not isinstance(tx_raw_payload, dict):
                    tx_raw_payload = {}

                tx_shap = (
                    graph_payload.get("transaction_shap_explanation")
                    or graph_payload.get("shap_explanation")
                    or tx_raw_payload.get("transaction_shap_explanation")
                    or tx_raw_payload.get("shap_explanation")
                )

                if tx_shap is not None:
                    tx_raw_payload["transaction_shap_explanation"] = tx_shap
                    tx_raw_payload["shap_explanation"] = tx_shap

                if wallet_shap is not None:
                    tx_raw_payload["wallet_shap_explanation"] = wallet_shap

                tx_row.raw_payload = tx_raw_payload
                session.add(tx_row)

                if source_wallet:
                    wallet_row = session.query(Wallet).filter(Wallet.address == str(source_wallet).lower()).one_or_none()
                    if wallet_row:
                        wallet_meta = wallet_row.metadata_json if isinstance(wallet_row.metadata_json, dict) else {}
                        if wallet_shap is not None:
                            wallet_meta["wallet_shap_explanation"] = wallet_shap
                        wallet_row.metadata_json = wallet_meta
                        session.add(wallet_row)

                session.commit()
                front_data = tx_to_front(tx_row)
                graph_status = {
                    "updated": True,
                    "affected_wallets": len(propagation.get("scores", {})),
                }
            except Exception as graph_exc:
                session.rollback()
                app.logger.exception("Graph update failed for tx %s: %s", tx_row.tx_hash, graph_exc)
                graph_status = {
                    "updated": False,
                    "affected_wallets": 0,
                    "error": str(graph_exc),
                }
            
            # EMIT LIVE EVENT (Updates dashboard instantly)
            socketio.emit('new_transaction', front_data)
            
            return jsonify({
                "status": "success",
                "tx_hash": tx_row.tx_hash,
                "graph": graph_status,
            }), 201
        except ValueError as e:
            session.rollback()
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    except Exception as e:
        app.logger.error(f"Failed to add transaction: {e}")
        return jsonify({"error": str(e)}), 500

def generate_ai_explanation(features: dict, risk_score: int, risk_prob: float):
    """
    Uses an LLM to generate a 'Thinking' Executive Summary.
    """
    try:
        # 1. Construct the Analyst Prompt
        # We give the AI a persona and the raw data
        prompt = f"""
        You are a Senior Blockchain Fraud Analyst for DeTrust. 
        Analyze the following Ethereum transaction data and provide a 2-sentence Executive Summary for a non-technical stakeholder.
        
        CONTEXT:
        - Risk Score: {risk_score}/10 (Probability: {risk_prob:.2%})
        - Transaction Value: {features.get('Transaction_Value')} ETH
        - Wallet Age: {features.get('Wallet_Age_Days')} days
        - Velocity: {features.get('Transaction_Velocity')} tx/day
        - Max/Min Balance Ratio (Volatility): {features.get('BMax_BMin_per_NT')}
        - Gas Efficiency: {features.get('Gas_Efficiency')}
        - Value to Fee Ratio: {features.get('Value_to_Fee_Ratio')}
        
        INSTRUCTIONS:
        1. Explain WHY the score is high or low based on the features.
        2. Highlight specific anomalies (e.g., "The high velocity combined with a new wallet suggests bot activity").
        3. Be professional, concise, and definitive. Do not use bullet points. Write it as a narrative.
        4. If the risk is Low, explain why it appears safe (e.g., "established history").
        """
        
        # 2. Call the AI Model (Gemini Example)
        model = genai.GenerativeModel('gemini-2.5-flash')
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        
        # --- NEW: SAFE RESPONSE HANDLING ---
        # Sometimes response.text fails if the model refused to answer
        if hasattr(response, 'text'):
            return response.text.replace('*', '').strip()
        elif response.parts:
            return response.parts[0].text.replace('*', '').strip()
        else:
            return "AI Analysis produced no text (Safety Block or Empty)."

    except Exception as e:
        # Log the specific error to your terminal so you can see it
        print(f"------------ AI ERROR DEBUG ------------\n{e}\n----------------------------------------")
        return f"AI Analysis unavailable: {str(e)}"
    
@app.route('/explain/transaction', methods=['POST'])
def explain_transaction_ai():
    try:
        data = request.get_json(force=True)
        # We expect the frontend to send the transaction details
        features = data.get('features')
        risk_score = data.get('risk_score')
        risk_prob = data.get('risk_prob')
        
        if not features:
            return jsonify({"error": "Missing features"}), 400

        # Call the AI function
        explanation = generate_ai_explanation(features, risk_score, risk_prob)
        
        return jsonify({"explanation": explanation})
    except Exception as e:
        logger.error(f"AI Explain Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- RISK EXPLANATION ---

# -----------------------------------------------------------------------------
# PRODUCTION-GRADE EXPLANATION ENGINE
# -----------------------------------------------------------------------------
def get_risk_explanation_rules(features: dict, risk_score: int):
    """
    Context-Aware Explanation Engine (v4.0 Production)
    
    Generates a dynamic, non-binary narrative by analyzing the 'net pressure' 
    of risk vs. trust factors. Uses logistic scaling to avoid brittle hard thresholds.
    
    Returns: [Narrative String, Detailed Dictionary]
    """

    # --- 1. INTERNAL HELPER UTILITIES ---
    def safe_float(x, default=0.0):
        try:
            if x is None: return float(default)
            val = float(x)
            return val if math.isfinite(val) else float(default)
        except:
            return float(default)

    def logistic_scale(x, x_mid, steep=1.0, out_min=0.0, out_max=2.0):
        """
        Maps input 'x' to a score between [out_min, out_max] using a S-curve.
        - x_mid: The point where the score flips or reaches the middle.
        - steep: How fast the score changes (higher = sharper transition).
        """
        try:
            val = 1.0 / (1.0 + math.exp(-steep * (x - x_mid)))
        except OverflowError:
            val = 0.0 if (steep * (x - x_mid)) < 0 else 1.0
        return out_min + (out_max - out_min) * val

    def add_factor(container, score, desc, category):
        # Clamp score reasonable bounds (0 to 3) for safety
        score = max(0.0, min(3.0, float(score)))
        container.append((score, desc, category))

    # --- 2. PARSE & CLEAN INPUTS (Defensive Coding) ---
    age = safe_float(features.get("Wallet_Age_Days", 0))
    velocity = safe_float(features.get("Transaction_Velocity", 0))
    balance = safe_float(features.get("Final_Balance", 0))
    value = safe_float(features.get("Transaction_Value", 0))
    inputs = safe_float(features.get("Number_of_Inputs", 1))
    outputs = safe_float(features.get("Number_of_Outputs", 1))
    gas_eff = safe_float(features.get("Gas_Efficiency", 0))
    volatility = safe_float(features.get("BMax_BMin_per_NT", 0))
    gas_price = safe_float(features.get("Gas_Price", 0))

    # Placeholders for future graph/context features (Defaults to False/0 for now)
    is_contract = bool(features.get("Is_Contract", False))
    cluster_score = safe_float(features.get("Known_Cluster_Score", 0.0)) 
    
    # --- 3. IMPACT ANALYSIS (Fuzzy Logic Containers) ---
    # Stores tuples: (ImpactScore, Description, Category['risk'|'trust'])
    factors = [] 

    # === LOGIC BLOCK A: WALLET AGE (Contextual) ===
    # "Newness" Risk: High if age < 7, fades to 0 as age approaches 30
    age_risk = logistic_scale(-age, -7, steep=0.25, out_min=0.0, out_max=1.8)
    
    # "History" Trust: Low if age < 90, grows strong as age approaches 365
    age_trust = logistic_scale(age, 365, steep=0.02, out_min=0.0, out_max=1.8)

    if age_risk > 0.2:
        add_factor(factors, age_risk, f"newly created wallet ({int(age)} days)", "risk")
    
    if age_trust > 0.5:
        add_factor(factors, age_trust * 0.9, f"long-established history ({int(age)} days)", "trust")

    # === LOGIC BLOCK B: VELOCITY (Dynamic) ===
    # High velocity is risky, BUT if wallet is ancient (Trust > 1.0), we tolerate it more.
    tolerance = 1.0 if age_trust > 1.0 else 0.3
    
    # Logistic curve: Velocity becomes risky around 10-15 tx/day
    vel_score = logistic_scale(velocity, 15.0, steep=0.15, out_min=0.0, out_max=2.0)
    
    if vel_score > (0.5 * tolerance): 
        add_factor(factors, vel_score, f"high transaction frequency ({int(velocity)} tx/day)", "risk")
    elif velocity < 2 and age > 30:
        add_factor(factors, 0.4, "stable transaction pacing", "trust")

    # === LOGIC BLOCK C: FINANCIALS (Balance & Value) ===
    # Low Balance Risk (Dust)
    if balance < 0.005:
        add_factor(factors, 1.2, "dust/near-zero balance", "risk")
    # High Balance Trust (Whale)
    elif balance > 10.0:
        add_factor(factors, 1.5, f"substantial reserves ({balance:.1f} ETH)", "trust")
    
    # Value Transfer Magnitude
    val_score = logistic_scale(value, 5.0, steep=0.4, out_min=0.0, out_max=1.8)
    if val_score > 0.4:
        desc = f"significant value transfer ({value:.1f} ETH)"
        # Interaction: New wallet moving big money is VERY bad
        if age < 30: 
            val_score *= 1.5
            desc += " via new wallet"
        add_factor(factors, val_score, desc, "risk")

    # === LOGIC BLOCK D: BEHAVIORAL PATTERNS ===
    # Input/Output Complexity (Layering)
    if inputs > 5:
        add_factor(factors, 0.8, "complex input mixing", "risk")
    
    if outputs > 10:
        add_factor(factors, 1.2, "mass-distribution output behavior", "risk")
    
    # Volatility (Stability check)
    vol_score = logistic_scale(volatility, 0.4, steep=5.0, out_min=0.0, out_max=1.2)
    if vol_score > 0.6:
        add_factor(factors, vol_score, "highly volatile balance history", "risk")
    elif volatility < 0.05 and age > 30:
        add_factor(factors, 0.5, "consistent accumulation pattern", "trust")

    # Gas Aggression (Bot check)
    if gas_price > 100: # Gwei
        add_factor(factors, 1.0, "aggressive gas bidding", "risk")

    # --- 4. CATEGORIZE & SORT ---
    drivers = [f for f in factors if f[2] == "risk"]
    mitigators = [f for f in factors if f[2] == "trust"]

    drivers.sort(key=lambda x: x[0], reverse=True)
    mitigators.sort(key=lambda x: x[0], reverse=True)

    # --- 5. COMPUTE RULE-DERIVED SCORE (for Comparison) ---
    # We calculate a 'Shadow Score' from these rules to see if it agrees with the ML model
    def dim_sum(items):
        total = 0.0
        for i, (sc, _, _) in enumerate(items):
            total += sc * (0.85 ** i) # Diminishing returns for multiple factors
        return total

    rule_risk_total = dim_sum(drivers)
    rule_trust_total = dim_sum(mitigators)
    
    # Net Score (0-10 scale approximation)
    net_rule_score = max(0.0, min(10.0, (rule_risk_total - (rule_trust_total * 0.8)) * 2.5))

    # --- 6. NARRATIVE GENERATION ---
    narrative_parts = []
    
    # A. Severity Header
    severity = "Critical" if risk_score >= 7 else "High" if risk_score >= 5 else "Medium" if risk_score >= 3 else "Low"
    narrative_parts.append(f"**{severity} Risk** Analysis (Model Score: {risk_score}/10).")

    # B. Driver Explanation (Why is it risky?)
    if risk_score >= 5:
        # High Risk Scenario
        if drivers:
            top = drivers[0]
            narrative_parts.append(f"The primary driver is **{top[1]}**.")
            if len(drivers) > 1:
                narrative_parts.append(f"Risk is compounded by **{drivers[1][1]}**.")
        else:
            narrative_parts.append("Risk is driven by a combination of minor behavioral anomalies.")
            
        if mitigators:
            narrative_parts.append(f"While **{mitigators[0][1]}** is positive, it does not outweigh the risk factors.")
            
    else:
        # Low/Medium Risk Scenario
        if mitigators:
            top = mitigators[0]
            narrative_parts.append(f"The score is suppressed by **{top[1]}**.")
            if len(mitigators) > 1:
                narrative_parts.append(f"Supported by **{mitigators[1][1]}**.")
            
            if drivers:
                narrative_parts.append(f"Although **{drivers[0][1]}** was detected, it remains within safe tolerances.")
        elif drivers:
            narrative_parts.append(f"Minor risk signals like **{drivers[0][1]}** were detected but are insufficient to trigger an alert.")
        else:
            narrative_parts.append("The transaction exhibits standard behavioral patterns.")

    # C. Model Alignment Check (Sanity Check)
    # If ML says 9/10 but Rules say 2/10, warn the user.
    if abs(risk_score - net_rule_score) > 3.5:
        narrative_parts.append("(Note: The ML model detects hidden patterns not fully explained by standard heuristics).")

    # Combine into single string
    full_narrative = " ".join(narrative_parts)

    # --- 7. RETURN STRUCTURE ---
    # Returns a list: [Narrative String, Debug Details Dictionary]
    # Frontend only uses index 0, but index 1 is available for future 'Advanced View'
    return [
        full_narrative,
        {
            "rule_score": round(net_rule_score, 2),
            "drivers": drivers,
            "mitigators": mitigators
        }
    ]

# --- 3. PREDICTION ENDPOINT (Used by Oracle) ---
@app.route('/predict/transaction', methods=['POST'])
def predict_transaction_risk():
    if not detector:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json(force=True)
        features_dict = data.get('features') or data.get('input_features') or data
        if not isinstance(features_dict, dict):
            return jsonify({"error": "Invalid features payload"}), 400

        wallet_address = data.get("wallet_address") or data.get("from_address") or features_dict.get("from_address") or features_dict.get("from")
        wallet_address = normalize_wallet_address(wallet_address) if wallet_address else None
        if wallet_address and not is_valid_wallet_address(wallet_address):
            return jsonify({"error": "Invalid wallet_address format"}), 400

        tx_timestamp = data.get("timestamp")
        update_state = data.get("update_state")
        if update_state is None:
            update_state = bool(wallet_address)

        # 1. Build model-aligned frame (prevents feature-order drift)
        X, raw_feats, engineered_feats = build_model_ready_frame(features_dict)

        # 2. Predict from resolved model object
        risk_probability = None
        m = model if model is not None else getattr(detector, 'model', detector)
        if hasattr(m, "predict_proba"):
            prob = m.predict_proba(X)
            risk_probability = float(prob[0][1])
        elif hasattr(m, "predict"):
            out = m.predict(X)
            risk_probability = 1.0 if int(out[0]) == 1 else 0.0
        else:
            raise ValueError("Loaded model does not expose predict_proba/predict")

        risk_probability = float(max(0.0, min(1.0, risk_probability)))
        threshold_used = float(getattr(detector, "threshold", 0.5) or 0.5)
        is_fraud_label = 1 if risk_probability >= threshold_used else 0
        risk_score = probability_to_risk_score(risk_probability)
        prediction_confidence = round(max(risk_probability, 1.0 - risk_probability), 6)
        
        # 3. SHAP explanation (singleton explainer initialized at startup).
        shap_features = engineered_feats.copy()
        try:
            if hasattr(detector, "_engineer_features"):
                shap_features = detector._engineer_features(raw_feats.copy())
        except Exception:
            pass

        shap_explanation = explain_prediction(
            shap_features,
            feature_order=model_feature_order,
            top_k=5,
        )

        # Build user-facing narrative directly from SHAP instead of scripted rule heuristics.
        risk_explanation = build_shap_analysis_text(shap_explanation, risk_score)

        # 4. Temporal state update (enabled for oracle calls carrying wallet address).
        temporal_payload = {
            "temporal_score_normalized": round(max(0.0, min(1.0, 1.0 - risk_probability)), 6),
            "temporal_score": round((1.0 - risk_probability) * 10.0, 3),
            "temporal_previous_score": None,
            "decay": 1.0,
            "time_gap_seconds": 0,
            "burst_detected": False,
            "tx_count_last_1_min": 0,
            "dormant_spike": False,
            "rolling_avg_risk": round(risk_probability, 6),
            "adjusted_risk_probability": round(risk_probability, 6),
        }

        if update_state and wallet_address:
            session = SessionLocal()
            try:
                temporal_payload = update_wallet_score(
                    session,
                    wallet_address=wallet_address,
                    risk_probability=risk_probability,
                    current_timestamp=tx_timestamp,
                )
                session.commit()
            except Exception as temporal_exc:
                session.rollback()
                logger.exception("Temporal scoring failed for wallet %s: %s", wallet_address, temporal_exc)
            finally:
                session.close()

        wallet_shap_explanation = None
        if wallet_address:
            try:
                wallet_shap_explanation = graph_builder.get_wallet_shap(wallet_address)
            except Exception:
                wallet_shap_explanation = None
        
        return jsonify({
            "wallet_address": wallet_address,
            "risk_probability": risk_probability,
            "fraud_probability": risk_probability,
            "risk_level": get_risk_level(risk_probability),
            "is_high_risk": risk_score >= 7,
            "risk_score": risk_score,
            "is_fraud_label": is_fraud_label,
            "threshold_used": threshold_used,
            "prediction_confidence": prediction_confidence,
            "temporal_score": temporal_payload.get("temporal_score"),
            "temporal_score_normalized": temporal_payload.get("temporal_score_normalized"),
            "temporal_previous_score": temporal_payload.get("temporal_previous_score"),
            "temporal_state": {
                "decay": temporal_payload.get("decay"),
                "time_gap_seconds": temporal_payload.get("time_gap_seconds"),
                "burst_detected": temporal_payload.get("burst_detected"),
                "tx_count_last_1_min": temporal_payload.get("tx_count_last_1_min"),
                "dormant_spike": temporal_payload.get("dormant_spike"),
                "rolling_avg_risk": temporal_payload.get("rolling_avg_risk"),
                "adjusted_risk_probability": temporal_payload.get("adjusted_risk_probability"),
                "formula_raw_score": temporal_payload.get("formula_raw_score"),
                "formula_normalized_score": temporal_payload.get("formula_normalized_score"),
                "update_alpha": temporal_payload.get("update_alpha"),
            },
            "explanation": risk_explanation,
            "shap_explanation": shap_explanation,
            "transaction_shap_explanation": shap_explanation,
            "wallet_shap_explanation": wallet_shap_explanation,
            "input_features": raw_feats,
            "model_features_used": list(X.columns),
            "feature_mapping": FEATURE_MEANINGS,
        }), 200

    except Exception as e:
        logger.error(f"Error during transaction prediction: {e}")
        return jsonify({"error": str(e)}), 400
    
# --- 4. PROXY ENDPOINTS (For Live Frontend Search) ---

@app.route('/api/proxy/transaction/<tx_hash>', methods=['GET'])
def proxy_tx_details(tx_hash):
    try:
        if not proxy_w3.is_connected():
            return jsonify({"error": "Backend RPC not connected"}), 500
            
        tx = proxy_w3.eth.get_transaction(tx_hash)
        receipt = proxy_w3.eth.get_transaction_receipt(tx_hash)
        block = proxy_w3.eth.get_block(tx['blockNumber'])
        
        session = SessionLocal()
        db_tx = session.query(Transaction).filter(Transaction.tx_hash == tx_hash.lower()).one_or_none()
        session.close()

        return jsonify({
            "hash": tx_hash,
            "status": "Success" if receipt['status'] == 1 else "Failed",
            "block": tx['blockNumber'],
            "timestamp": block['timestamp'],
            "from": tx['from'],
            "to": tx['to'],
            "value": float(proxy_w3.from_wei(tx['value'], 'ether')),
            "fee": float(proxy_w3.from_wei(tx['gasPrice'] * receipt['gasUsed'], 'ether')),
            "risk_score": db_tx.risk_score if db_tx else None,
            "risk_level": get_risk_level(db_tx.risk_score) if db_tx else "Not Analyzed"
        })
    except Exception as e:
        return jsonify({"error": "Transaction not found"}), 404

@app.route('/api/proxy/wallet/<address>', methods=['GET'])
def proxy_wallet_details(address):
    try:
        address = Web3.to_checksum_address(address)
        balance = proxy_w3.eth.get_balance(address)
        count = proxy_w3.eth.get_transaction_count(address)
        
        session = SessionLocal()
        db_wallet = crud.get_wallet_by_address(session, address)
        session.close()

        return jsonify({
            "address": address,
            "balance": float(proxy_w3.from_wei(balance, 'ether')),
            "tx_count": count,
            "trust_score": db_wallet.trust_score if db_wallet else None,
            "trust_level": get_trust_level(db_wallet.trust_score) if db_wallet else "Unknown"
        })
    except Exception as e:
        return jsonify({"error": "Wallet not found"}), 404

# --- MAIN EXECUTION ---
load_model()

if __name__ == '__main__':
    logger.info("Ethereum Fraud Detection API starting...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
    