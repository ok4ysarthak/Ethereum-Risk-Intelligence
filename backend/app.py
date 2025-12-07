import os
import joblib
import numpy as np
import pandas as pd
import math
import logging
import threading
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from web3 import Web3
from flask_socketio import SocketIO, emit
from wallet_updater import WalletScoreUpdater, INITIAL_SCORE
from config import Config
from Database.db import init_db, SessionLocal
import Database.crud as crud
from Database.models_db import Transaction, Wallet

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


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

with app.app_context():
    try:
        app.logger.info("Creating database tables if not exist...")
        init_db()
        app.logger.info("Database initialized!")
    except Exception as e:
        app.logger.exception("DB init failed: %s", e)

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

# --- Model Loading ---
MODEL_PATH = Config.MODEL_PATH
model = None
detector = None

def load_model():
    global model, detector
    try:
        detector = joblib.load(MODEL_PATH)
        model = detector.model if hasattr(detector, 'model') else detector
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
    ts = None
    try: ts = int(tx_row.timestamp.timestamp())
    except: ts = None
    return {
        "id": tx_row.tx_hash,
        "transaction_hash": tx_row.tx_hash,
        "from": (tx_row.from_address or "").lower(),
        "to": (tx_row.to_address or "").lower(),
        "value": float(tx_row.amount_eth or 0.0),
        "riskScore": float(tx_row.risk_score or 0.0),
        "walletScore": float(tx_row.wallet_trust_score or 0.0),
        "timestamp": ts,
        "status": tx_row.status,
        "saved_to_chain": bool(tx_row.saved_to_chain or False),
        "onchain_record_txhash": tx_row.onchain_record_txhash,
        "raw_transaction": tx_row.raw_payload or (tx_row.metadata_json or {}),
    }

def wallet_to_front(wallet_row):
    if not wallet_row: return None
    return {
        "address": (wallet_row.address or "").lower(),
        "first_seen": int(wallet_row.first_seen.timestamp()) if wallet_row.first_seen else None,
        "last_seen": int(wallet_row.last_seen.timestamp()) if wallet_row.last_seen else None,
        "age_days": wallet_row.age_days,
        "score": float(wallet_row.trust_score or 0.0) if wallet_row.trust_score is not None else None,
        "avg_risk": float(wallet_row.avg_risk or 0.0) if wallet_row.avg_risk is not None else None,
        "labels": wallet_row.labels or {},
        "metadata": wallet_row.metadata_json or {},
    }

def get_risk_level(risk_score):
    if not risk_score: return "Unknown"
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

def extract_features_from_tx(raw_tx):
    defaults = {
        "Transaction_Value": 0.0, "Transaction_Fees": 0.0,
        "Number_of_Inputs": 0.0, "Number_of_Outputs": 0.0,
        "Gas_Price": 0.0, "Wallet_Age_Days": 0.0, "Wallet_Balance": 0.0,
        "Transaction_Velocity": 0.0, "Exchange_Rate": 0.0,
        "Final_Balance": 0.0, "BMax_BMin_per_NT": 0.0
    }
    if not raw_tx: return defaults.copy()
    f = {}
    for k in defaults.keys():
        if k in raw_tx:
            try: f[k] = float(raw_tx.get(k) or 0.0)
            except: f[k] = defaults[k]
        else:
            lk = k.lower()
            if lk in raw_tx:
                try: f[k] = float(raw_tx.get(lk) or 0.0)
                except: f[k] = defaults[k]
            else:
                f[k] = defaults[k] # simplified fallback
    return f

def build_complete_feature_vector(raw_features_dict):
    feature_order = [
        'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs',
        'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days',
        'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate',
        'Final_Balance', 'BMax_BMin_per_NT'
    ]
    arr = [float(raw_features_dict.get(k, 0.0) or 0.0) for k in feature_order]
    
    # Engineering (Ratios)
    tx_value = float(raw_features_dict.get("Transaction_Value", 0.0))
    tx_fees = float(raw_features_dict.get("Transaction_Fees", 0.0))
    gas_price = float(raw_features_dict.get("Gas_Price", 0.0))
    
    eps = 1e-8
    value_to_fee_ratio = tx_value / (tx_fees + eps)
    gas_efficiency = tx_value / (gas_price + eps)
    
    arr_extended = arr + [value_to_fee_ratio, gas_efficiency]
    full_names = feature_order + ["value_to_fee_ratio", "gas_efficiency"]
    
    return arr_extended, full_names

# ---------------------------
# API ROUTES
# ---------------------------

@app.route('/')
def serve_index(): return send_from_directory(app.static_folder, 'index.html')

@app.route('/activity')
def serve_activity_page(): return send_from_directory(app.static_folder, 'activity.html')

@app.route('/live')
def serve_live_page(): return send_from_directory(app.static_folder, 'live.html')

@app.route('/<path:path>')
def serve_static_files(path):
    try: return send_from_directory(app.static_folder, path)
    except: return send_from_directory(app.static_folder, 'index.html')

# --- 1. DB ROUTES (FETCHING) ---

@app.route('/transactions', methods=['GET'])
def api_get_transactions():
    try:
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        min_risk = request.args.get('min_risk', None)
        from_addr = request.args.get('from_address', None)
        to_addr = request.args.get('to_address', None)
        sort = request.args.get('sort', 'time')
        session = SessionLocal()
        q = session.query(Transaction)
        if min_risk is not None:
            try: q = q.filter(Transaction.risk_score >= float(min_risk))
            except: pass
        if from_addr: q = q.filter(Transaction.from_address == from_addr.lower())
        if to_addr: q = q.filter(Transaction.to_address == to_addr.lower())
        if sort == 'risk': q = q.order_by(Transaction.risk_score.desc())
        else: q = q.order_by(Transaction.timestamp.desc())
        rows = q.offset(offset).limit(limit).all()
        session.close()
        return jsonify([tx_to_front(r) for r in rows])
    except Exception as e:
        app.logger.exception("Error in /transactions: %s", e)
        return jsonify({"error": "internal server error"}), 500

@app.route('/wallets', methods=['GET'])
def get_wallets():
    try:
        limit = int(request.args.get('limit', 100))
        session = SessionLocal()
        rows = session.query(Wallet).order_by(Wallet.last_seen.desc()).limit(limit).all()
        session.close()
        return jsonify([wallet_to_front(w) for w in rows])
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
            
            # EMIT LIVE EVENT (Updates dashboard instantly)
            socketio.emit('new_transaction', front_data)
            
            return jsonify({"status": "success", "tx_hash": tx_row.tx_hash}), 201
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    except Exception as e:
        app.logger.error(f"Failed to add transaction: {e}")
        return jsonify({"error": str(e)}), 500

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
            
        # 1. Extract and Build Vector
        raw_feats = extract_features_from_tx(features_dict)
        complete_vector, full_feature_names = build_complete_feature_vector(raw_feats)
        
        # 2. Predict
        X = np.array([complete_vector], dtype=float)
        risk_probability = None
        
        # Try method from new class first
        if hasattr(detector, 'predict_risk_score'):
             try:
                res = detector.predict_risk_score(raw_feats)
                risk_probability = res['fraud_probability']
             except Exception as e:
                logger.debug(f"detector.predict_risk_score failed: {e}")

        # Fallback
        if risk_probability is None:
            m = model if model is not None else detector
            if hasattr(m, "predict_proba"):
                prob = m.predict_proba(X)
                risk_probability = float(prob[0][1])
            elif hasattr(m, "predict"):
                out = m.predict(X)
                risk_probability = 1.0 if out[0] == 1 else 0.0

        risk_probability = float(risk_probability)
        return jsonify({
            "risk_probability": risk_probability,
            "risk_level": get_risk_level(risk_probability),
            "is_high_risk": risk_probability > Config.HIGH_RISK_THRESHOLD,
            "risk_score": int(math.ceil(risk_probability * 10)) if risk_probability > 0.01 else 1
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
if __name__ == '__main__':
    load_model()
    logger.info("Ethereum Fraud Detection API starting...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)