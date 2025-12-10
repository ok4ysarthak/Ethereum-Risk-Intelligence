import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

# --- IMPORTS ---
from backend.feature_calculator import RealTimeFeatureCalculator

SEPOLIA_RPC = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"
ETHERSCAN_KEY = "862HRU5VPZ5ENFKX2F9A2C44AZQVHR81TP"
MODEL_PATH = "../ml_models/saved_models/fraud_detector_model.pkl" 
TEST_TX_HASH = "0xd31b673de81905c829b3920ff851784aec9523a9de20a43209b7661ce69334b5"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Verification")

class ProductionXGBoostFraudDetector:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.threshold = 0.5

    def _engineer_features(self, raw_features):
        features = raw_features.copy()
        features["Value_to_Fee_Ratio"] = features["Transaction_Value"] / (features["Transaction_Fees"] + 1e-8)
        features["Gas_Efficiency"] = features["Transaction_Value"] / (features["Gas_Price"] + 1e-8)
        features["Input_Output_Ratio"] = features["Number_of_Inputs"] / (features["Number_of_Outputs"] + 1e-8)
        features["Balance_Utilization"] = features["Transaction_Value"] / (features["Wallet_Balance"] + 1e-8)
        features["Tx_Frequency_Score"] = features["Transaction_Velocity"] / (features["Wallet_Age_Days"] + 1.0)
        features["Value_Velocity_Interaction"] = (features["Transaction_Value"] * features["Transaction_Velocity"])
        features["Volatility_Age_Interaction"] = features["BMax_BMin_per_NT"] * np.log1p(features["Wallet_Age_Days"])
        features["Gas_Complexity"] = (features["Gas_Price"] * features["Number_of_Inputs"])
        features["Log_Transaction_Value"] = np.log1p(features["Transaction_Value"])
        features["Log_Wallet_Balance"] = np.log1p(features["Wallet_Balance"])
        features["Log_Wallet_Age"] = np.log1p(features["Wallet_Age_Days"])
        features["Is_Young_Wallet"] = int(features["Wallet_Age_Days"] < 30)
        features["Is_High_Velocity"] = int(features["Transaction_Velocity"] > 5.0)
        return features

    def predict_risk_score(self, tx_features):
        if self.model is None: raise ValueError("Model not loaded")
        features = self._engineer_features(tx_features)
        
        cols = [c for c in self.feature_names if c in features]
        missing = [c for c in self.feature_names if c not in features]
        
        X_dict = {col: [features.get(col, 0.0)] for col in self.feature_names}
        X = pd.DataFrame(X_dict)
        
        proba = self.model.predict_proba(X)[0]
        fraud_prob = float(proba[1])
        
        if fraud_prob < 0.05: risk_score = 1
        elif fraud_prob < 0.15: risk_score = 2
        elif fraud_prob < 0.25: risk_score = 3
        elif fraud_prob < 0.35: risk_score = 4
        elif fraud_prob < 0.45: risk_score = 5
        elif fraud_prob < 0.55: risk_score = 6
        elif fraud_prob < 0.65: risk_score = 7
        elif fraud_prob < 0.75: risk_score = 8
        elif fraud_prob < 0.85: risk_score = 9
        else: risk_score = 10
        
        return fraud_prob, risk_score


def verify_system():
    print("\n🔍 STARTING SYSTEM VERIFICATION...")
    print("=" * 60)

    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found! Check path: {MODEL_PATH}")
        return

    try:
        detector = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Connect
    try:
        calc = RealTimeFeatureCalculator(
            rpc_url=SEPOLIA_RPC,
            etherscan_api_key=ETHERSCAN_KEY,
            network='sepolia'
        )
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return

    # 3. Fetch Data
    print(f"\n📡 Fetching Live Data for TX: {TEST_TX_HASH}")
    features = calc.get_features_for_tx(TEST_TX_HASH)
    
    if not features:
        logger.error("Failed to fetch transaction features.")
        return

    # 4. Display Stats
    print("\nWALLET STATISTICS:")
    print(f"   --------------------------------")
    print(f"B_max (Max Balance):  {features.get('B_max', 0):.4f} ETH")
    print(f"B_min (Min Balance):  {features.get('B_min', 0):.4f} ETH")
    print(f"NT (Tx Count):        {features.get('NT', 0)}")
    print(f"   --------------------------------")
    print(f"Volatility:           {features.get('BMax_BMin_per_NT', 0):.6f}")
    print(f"Wallet Age:           {features.get('Wallet_Age_Days', 0):.2f} Days")
    
    # 5. Predict
    print("\n Running ML Prediction...")
    try:
        prob, score = detector.predict_risk_score(features)
        
        print("-" * 30)
        print(f"Fraud Probability: {prob:.4f} ({prob*100:.2f}%)")
        print(f"Risk Score:        {score} / 10")
        
        if score > 7:
            print("VERDICT: HIGH RISK")
        elif score > 4:
            print("VERDICT: MEDIUM RISK")
        else:
            print("VERDICT: LOW RISK (LEGIT)")
        print("-" * 30)
        
    except Exception as e:
        logger.error(f"Prediction Logic Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_system()