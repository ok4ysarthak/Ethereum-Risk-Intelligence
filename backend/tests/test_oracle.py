import joblib
import pandas as pd
import numpy as np
import math

MODEL_PATH = "../ml_models/saved_models/fraud_detector_model.pkl"

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
        
        # Ensure columns exist and order matches training
        X_dict = {col: [features.get(col, 0.0)] for col in self.feature_names}
        X = pd.DataFrame(X_dict)
        
        proba = self.model.predict_proba(X)[0]
        fraud_prob = float(proba[1])
        
        # Score Logic
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

def run_simulation():
    print(f"📂 Loading Model: {MODEL_PATH}")
    try:
        detector = joblib.load(MODEL_PATH)
        print("Model Loaded!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    boring_tx = {
        "Transaction_Value": 0.0004,
        "Transaction_Fees": 0.00003,
        "Number_of_Inputs": 1.0,
        "Number_of_Outputs": 1.0,
        "Gas_Price": 1.5,
        "Wallet_Age_Days": 4.0,
        "Wallet_Balance": 2.2,
        "Transaction_Velocity": 7.0,
        "Exchange_Rate": 3000.0,
        "Final_Balance": 2.2,       # No draining
        "BMax_BMin_per_NT": 0.08    # Low volatility
    }

    thief_tx = {
        "Transaction_Value": 50.0,  # Stealing 50 ETH
        "Transaction_Fees": 0.005,
        "Number_of_Inputs": 1.0,
        "Number_of_Outputs": 1.0,
        "Gas_Price": 50.0,          # Rushing the tx
        "Wallet_Age_Days": 0.1,     # Brand new wallet
        "Wallet_Balance": 50.1,     # Had 50.1
        "Transaction_Velocity": 20.0,
        "Exchange_Rate": 3000.0,
        "Final_Balance": 0.01,      # DRAINED to near zero!
        "BMax_BMin_per_NT": 25.0    # Massive volatility (0 -> 50 -> 0)
    }

    print("\n---TEST 1: Boring Sepolia Transaction ---")
    prob, score = detector.predict_risk_score(boring_tx)
    print(f"Result: Risk {score}/10 (Prob: {prob:.4f})")
    print("Expected: LOW Risk (1-3)")

    print("\n---TEST 2: The 'Thief' Scenario ---")
    prob, score = detector.predict_risk_score(thief_tx)
    print(f"Result: Risk {score}/10 (Prob: {prob:.4f})")
    print("Expected: HIGH Risk (8-10)")

if __name__ == "__main__":
    run_simulation()