import os
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from feature_calculator import RealTimeFeatureCalculator

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FraudTest")

class ProductionXGBoostFraudDetector:
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance_dict: Dict[str, float] = {}
        self.training_metadata: Dict[str, Any] = {}
        self.threshold: float = 0.5
        self.model_version: str = "3.0"

    def _engineer_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        features = raw_features.copy()

        # Derived ratios (Safe Division)
        features["Value_to_Fee_Ratio"] = features["Transaction_Value"] / (features["Transaction_Fees"] + 1e-8)
        features["Gas_Efficiency"] = features["Transaction_Value"] / (features["Gas_Price"] + 1e-8)
        features["Input_Output_Ratio"] = features["Number_of_Inputs"] / (features["Number_of_Outputs"] + 1e-8)
        features["Balance_Utilization"] = features["Transaction_Value"] / (features["Wallet_Balance"] + 1e-8)
        features["Tx_Frequency_Score"] = features["Transaction_Velocity"] / (features["Wallet_Age_Days"] + 1.0)

        # Interaction terms
        features["Value_Velocity_Interaction"] = (features["Transaction_Value"] * features["Transaction_Velocity"])
        features["Volatility_Age_Interaction"] = features["BMax_BMin_per_NT"] * np.log1p(features["Wallet_Age_Days"])
        features["Gas_Complexity"] = (features["Gas_Price"] * features["Number_of_Inputs"])

        # Log transforms
        features["Log_Transaction_Value"] = np.log1p(features["Transaction_Value"])
        features["Log_Wallet_Balance"] = np.log1p(features["Wallet_Balance"])
        features["Log_Wallet_Age"] = np.log1p(features["Wallet_Age_Days"])

        # Binary risk flags
        features["Is_Young_Wallet"] = int(features["Wallet_Age_Days"] < 30)
        features["Is_High_Velocity"] = int(features["Transaction_Velocity"] > 5.0)

        return features

    def _probability_to_risk_score(self, probability: float) -> int:
        if probability < 0.05: return 1
        elif probability < 0.15: return 2
        elif probability < 0.25: return 3
        elif probability < 0.35: return 4
        elif probability < 0.45: return 5
        elif probability < 0.55: return 6
        elif probability < 0.65: return 7
        elif probability < 0.75: return 8
        elif probability < 0.85: return 9
        else: return 10

    def _get_risk_category(self, risk_score: int) -> str:
        if risk_score <= 3: return "LOW"
        elif risk_score <= 6: return "MEDIUM"
        elif risk_score <= 8: return "HIGH"
        else: return "CRITICAL"

    def predict_risk_score(self, tx_features: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not loaded / trained.")

        # 1. Engineer Features
        features = self._engineer_features(tx_features)
        
        # 2. Prepare DataFrame
        try:
            X = pd.DataFrame([features])[self.feature_names]
        except KeyError as e:
            missing = list(set(self.feature_names) - set(features.keys()))
            logger.warning(f"Feature mismatch! Filling missing with 0: {missing}")
            for col in missing:
                features[col] = 0.0
            X = pd.DataFrame([features])[self.feature_names]

        # 3. Predict
        proba = self.model.predict_proba(X)[0]
        fraud_prob = float(proba[1])
        confidence = float(max(proba))

        risk_score = self._probability_to_risk_score(fraud_prob)
        is_fraud = int(fraud_prob >= self.threshold)
        risk_category = self._get_risk_category(risk_score)

        return {
            "fraud_probability": fraud_prob,
            "risk_score": risk_score,
            "is_fraud_label": is_fraud,
            "risk_category": risk_category,
            "input_features": features 
        }

def calculate_trust_score(current_trust: float, risk_prob: float, tx_value: float) -> float:
    """
    Simple decay logic:
    - High risk transactions lower the score drastically.
    - Low risk transactions slowly increase it.
    - Large value transactions have higher impact.
    """
    alpha = 0.1
    
    if risk_prob > 0.5:
        penalty = (risk_prob - 0.5) * 200 * alpha  
        new_score = max(0, current_trust - penalty)
    else:
        reward = (0.5 - risk_prob) * 10 * alpha
        new_score = min(100, current_trust + reward)
        
    return round(new_score, 2)


def run_test():
    MODEL_FILE = "../ml_models/saved_models/fraud_detector_model.pkl" 
    SEPOLIA_RPC = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"
    ETHERSCAN_KEY = "862HRU5VPZ5ENFKX2F9A2C44AZQVHR81TP"
    
    TX_HASH = "0xd31b673de81905c829b3920ff851784aec9523a9de20a43209b7661ce69334b5"

    print("\n--- 1. Loading Model ---")
    if not os.path.exists(MODEL_FILE):
        logger.error(f"Model file not found at {MODEL_FILE}. Please check path.")
        return

    try:
        detector = joblib.load(MODEL_FILE)
        print(f"Model loaded! Version: {detector.model_version}")
        print(f"Expecting {len(detector.feature_names)} features.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    print("\n--- 2. Fetching Live Features ---")
    calculator = RealTimeFeatureCalculator(
        rpc_url=SEPOLIA_RPC,
        etherscan_api_key=ETHERSCAN_KEY,
        network='sepolia'
    )
    
    features = calculator.get_features_for_tx(TX_HASH)
    
    if not features:
        logger.error("Failed to fetch features. Check RPC/Etherscan keys.")
        return

    print("Features Fetched Successfully:")
    print(f"   - Transaction Value: {features['Transaction_Value']} ETH")
    print(f"   - Wallet Age: {features['Wallet_Age_Days']:.2f} Days")
    print(f"   - Volatility: {features['BMax_BMin_per_NT']:.4f}")
    print(f"   - Velocity: {features['Transaction_Velocity']:.4f}")

    # 3. RUN PREDICTION
    print("\n--- 3. Predicting Risk ---")
    try:
        result = detector.predict_risk_score(features)
        
        prob = result['fraud_probability']
        score = result['risk_score']
        label = "FRAUD" if result['is_fraud_label'] == 1 else "LEGIT"
        
        print(f"FRAUD PROBABILITY: {prob:.4f} ({prob*100:.2f}%)")
        print(f"RISK SCORE: {score}/10")
        print(f"VERDICT: {label} ({result['risk_category']})")
        
        current_trust = 50.0 
        new_trust = calculate_trust_score(current_trust, prob, features['Transaction_Value'])
        
        print("\n--- 4. Wallet Trust Update ---")
        print(f"   - Previous Trust Score: {current_trust}")
        print(f"   - New Trust Score:      {new_trust}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()