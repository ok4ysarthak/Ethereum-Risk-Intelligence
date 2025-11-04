# backend/app.py
# backend/app.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
import logging
from datetime import datetime
import threading
import time

# Import your other modules
from data_fetcher import EthereumDataFetcher
from wallet_scorer import WalletScorer
from batch_processor import BatchProcessor
from config import Config

app = Flask(__name__)

# --- Setup Logging (Fixed for Windows) ---
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),  # Add encoding
        logging.StreamHandler()
    ]
)
# Set encoding for console handler to avoid emoji issues on Windows
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.stream.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


# --- Model Loading ---
MODEL_PATH = Config.MODEL_PATH

model = None
detector = None
data_fetcher = None
wallet_scorer = None
batch_processor = None
batch_thread = None

def load_model():
    """Load the model from disk."""
    global model, detector, data_fetcher, wallet_scorer, batch_processor
    try:
        detector = joblib.load(MODEL_PATH)
        model = detector.model if hasattr(detector, 'model') else detector
        
        # Initialize other components AFTER loading the model
        data_fetcher = EthereumDataFetcher(Config.get_rpc_url())
        wallet_scorer = WalletScorer(detector)  # Now this will work
        batch_processor = BatchProcessor(data_fetcher, detector, wallet_scorer)
        
        logger.info(f"✅ Model and components loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.error(f"❌ Error: Model file not found at {MODEL_PATH}")
        logger.error("Please make sure you have trained the model first.")
        exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        exit(1)


# --- API Endpoints ---

@app.route('/health')
def health_check():
    """Simple health check to see if the server is running."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

@app.route('/predict/transaction', methods=['POST'])
def predict_transaction_risk():
    """
    Predicts the risk of a transaction.
    Expects JSON data like:
    {
      "features": {
        "Transaction_Value": 1.5,
        "Transaction_Fees": 0.002,
        "Number_of_Inputs": 2,
        "Number_of_Outputs": 3,
        "Gas_Price": 20,
        "Wallet_Age_Days": 180,
        "Wallet_Balance": 5.2,
        "Transaction_Velocity": 2.5,
        "Exchange_Rate": 2000.0
      }
    }
    """
    if not detector:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        features_dict = data['features']
        
        # Feature order for the new dataset
        feature_order = [
            'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs', 
            'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days', 
            'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate'
        ]
        
        # Create the feature array in the correct order
        features_array = []
        for key in feature_order:
            if key in features_dict:
                features_array.append(float(features_dict[key]))
            else:
                features_array.append(0.0)  # Default value for missing features
        
        # Add engineered features
        tx_value = features_array[0]
        tx_fees = features_array[1]
        gas_price = features_array[4]
        wallet_balance = features_array[6]
        num_inputs = features_array[2]
        num_outputs = features_array[3]
        
        # Engineered features (same as in training)
        value_to_fee_ratio = tx_value / (tx_fees + 1e-8)
        input_output_ratio = num_inputs / (num_outputs + 1e-8)
        gas_efficiency = tx_value / (gas_price + 1e-8)
        balance_turnover = tx_value / (wallet_balance + 1e-8)
        
        # Complete feature array with engineered features
        complete_features = features_array + [
            value_to_fee_ratio, input_output_ratio, gas_efficiency, balance_turnover
        ]
        
        # Make prediction
        risk_score = detector.predict_fraud_risk(dict(zip(feature_order, features_array)))
        
        logger.info(f"Transaction risk prediction complete. Risk: {risk_score:.4f}")

        return jsonify({
            "risk_probability": float(risk_score),
            "risk_level": get_risk_level(risk_score),
            "is_high_risk": risk_score > Config.HIGH_RISK_THRESHOLD,
            "input_features": features_dict
        })

    except Exception as e:
        logger.error(f"❌ Error during transaction prediction: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/predict/wallet/<address>', methods=['GET'])
def get_wallet_score(address):
    """
    Get the trust score for a wallet address.
    """
    try:
        if not wallet_scorer:
            return jsonify({"error": "System not initialized"}), 500
        
        # Get current wallet score
        wallet_score = wallet_scorer.get_wallet_score(address)
        
        return jsonify({
            "wallet_address": address,
            "trust_score": float(wallet_score),
            "trust_level": get_trust_level(wallet_score),
            "is_trusted": wallet_score >= Config.LOW_TRUST_THRESHOLD
        })

    except Exception as e:
        logger.error(f"❌ Error getting wallet score: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/wallet/history/<address>', methods=['GET'])
def get_wallet_history(address):
    """
    Get transaction history for a wallet.
    """
    try:
        if not wallet_scorer:
            return jsonify({"error": "System not initialized"}), 500
        
        history = wallet_scorer.wallet_history.get(address, [])
        
        return jsonify({
            "wallet_address": address,
            "transaction_count": len(history),
            "recent_transactions": history[-10:] if history else []  # Last 10 transactions
        })

    except Exception as e:
        logger.error(f"❌ Error getting wallet history: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/batch/start', methods=['POST'])
def start_batch_processing():
    """
    Start the batch processing in a separate thread.
    """
    global batch_thread
    
    try:
        if batch_thread and batch_thread.is_alive():
            return jsonify({"message": "Batch processing already running"}), 200
        
        def run_batch_processor():
            while True:
                try:
                    batch_processor.process_new_transactions()
                    time.sleep(Config.BATCH_INTERVAL_MINUTES * 60)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        batch_thread = threading.Thread(target=run_batch_processor, daemon=True)
        batch_thread.start()
        
        return jsonify({"message": "Batch processing started", "interval_minutes": Config.BATCH_INTERVAL_MINUTES})

    except Exception as e:
        logger.error(f"❌ Error starting batch processing: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/batch/status', methods=['GET'])
def batch_status():
    """
    Get the status of batch processing.
    """
    is_running = batch_thread and batch_thread.is_alive()
    return jsonify({
        "batch_processing_running": is_running,
        "interval_minutes": Config.BATCH_INTERVAL_MINUTES,
        "last_processed_count": len(getattr(batch_processor, 'processed_transactions', []))
    })

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score >= 0.8:
        return "Very High"
    elif risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    elif risk_score >= 0.2:
        return "Low"
    else:
        return "Very Low"

def get_trust_level(trust_score):
    """Convert trust score to trust level"""
    if trust_score >= 0.8:
        return "Very High"
    elif trust_score >= 0.6:
        return "High"
    elif trust_score >= 0.4:
        return "Medium"
    elif trust_score >= 0.2:
        return "Low"
    else:
        return "Very Low"

# --- Main Execution ---
if __name__ == '__main__':
    load_model()  # Load the model when the app starts
    logger.info("🚀 Ethereum Fraud Detection API starting...")
    logger.info(f"📊 Model loaded successfully")
    logger.info(f"🌐 API available at http://0.0.0.0:5000")
    logger.info(f"⚡ Batch processing interval: {Config.BATCH_INTERVAL_MINUTES} minutes")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
