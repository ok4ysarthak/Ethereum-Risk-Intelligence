"""
Production-Grade XGBoost Ethereum Fraud Detection Model
========================================================
- Tuned for Mainnet/Sepolia generalization
- Adjusted Scoring: "Uncertain" transactions (30-50% prob) are now Low Risk (Score 3)
- Fixes "Average 5/10" issue by creating stricter thresholds for Medium/High risk
"""

import os
import json
import math
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
import joblib
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PRODUCTION MODEL CLASS
# =============================================================================

class ProductionXGBoostFraudDetector:
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance_dict: Dict[str, float] = {}
        self.training_metadata: Dict[str, Any] = {}
        self.threshold: float = 0.6  # Increased threshold for binary "Is Fraud" label
        self.model_version: str = "5.1-Calibrated"

    def _engineer_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        features = raw_features.copy()

        # Ratios (Safe Division)
        features["Value_to_Fee_Ratio"] = features["Transaction_Value"] / (features["Transaction_Fees"] + 1e-8)
        features["Gas_Efficiency"] = features["Transaction_Value"] / (features["Gas_Price"] + 1e-8)
        features["Input_Output_Ratio"] = features["Number_of_Inputs"] / (features["Number_of_Outputs"] + 1e-8)
        features["Balance_Utilization"] = features["Transaction_Value"] / (features["Wallet_Balance"] + 1e-8)
        features["Tx_Frequency_Score"] = features["Transaction_Velocity"] / (features["Wallet_Age_Days"] + 1.0)

        # Interactions
        features["Value_Velocity_Interaction"] = (features["Transaction_Value"] * features["Transaction_Velocity"])
        features["Volatility_Age_Interaction"] = features["BMax_BMin_per_NT"] * np.log1p(features["Wallet_Age_Days"])
        features["Gas_Complexity"] = (features["Gas_Price"] * features["Number_of_Inputs"])

        # Logs
        features["Log_Transaction_Value"] = np.log1p(features["Transaction_Value"])
        features["Log_Wallet_Balance"] = np.log1p(features["Wallet_Balance"])
        features["Log_Wallet_Age"] = np.log1p(features["Wallet_Age_Days"])

        # Flags
        features["Is_Young_Wallet"] = int(features["Wallet_Age_Days"] < 30)
        features["Is_High_Velocity"] = int(features["Transaction_Velocity"] > 5.0)

        return features

    # ---------- NEW SCORING LOGIC (The Fix) ----------
    def _probability_to_risk_score(self, probability: float) -> int:
        # We widen the "Safe" bucket so 40-50% probability is still Low Risk
        if probability < 0.15: return 1   # Very Safe
        elif probability < 0.30: return 2 # Safe
        elif probability < 0.50: return 3 # Low Risk (Most 'boring' txs land here)
        elif probability < 0.65: return 4 # Warning
        elif probability < 0.75: return 5 # Medium Risk
        elif probability < 0.80: return 6
        elif probability < 0.85: return 7
        elif probability < 0.90: return 8 # High Risk
        elif probability < 0.95: return 9
        else: return 10                   # Critical

    def _get_risk_category(self, risk_score: int) -> str:
        if risk_score <= 3: return "LOW"
        elif risk_score <= 5: return "MEDIUM"
        elif risk_score <= 8: return "HIGH"
        else: return "CRITICAL"

    def predict_risk_score(self, tx_features: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None: raise ValueError("Model not loaded.")
        
        features = self._engineer_features(tx_features)
        
        # Build DataFrame with correct column order
        X = pd.DataFrame([features])
        for col in self.feature_names:
            if col not in X.columns: X[col] = 0.0
        X = X[self.feature_names]

        proba = self.model.predict_proba(X)[0]
        fraud_prob = float(proba[1])
        confidence = float(max(proba))

        risk_score = self._probability_to_risk_score(fraud_prob)
        is_fraud = int(fraud_prob >= self.threshold)

        return {
            "fraud_probability": fraud_prob,
            "risk_score": risk_score,
            "is_fraud_label": is_fraud,
            "confidence": confidence,
            "risk_category": self._get_risk_category(risk_score),
            "model_version": self.model_version,
        }

    def predict_batch(self, transactions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        results = []
        for tx in transactions:
            results.append(self.predict_risk_score(tx))
        return results

# =============================================================================
# 2. FEATURE ENGINEERING (TRAINING)
# =============================================================================
def advanced_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print("\nAdvanced feature engineering...")
    df = df.copy()

    # Ratios
    df["Value_to_Fee_Ratio"] = df["Transaction_Value"] / (df["Transaction_Fees"] + 1e-8)
    df["Gas_Efficiency"] = df["Transaction_Value"] / (df["Gas_Price"] + 1e-8)
    df["Input_Output_Ratio"] = df["Number_of_Inputs"] / (df["Number_of_Outputs"] + 1e-8)
    df["Balance_Utilization"] = df["Transaction_Value"] / (df["Wallet_Balance"] + 1e-8)
    df["Tx_Frequency_Score"] = df["Transaction_Velocity"] / (df["Wallet_Age_Days"] + 1.0)

    # Interactions
    df["Value_Velocity_Interaction"] = df["Transaction_Value"] * df["Transaction_Velocity"]
    df["Volatility_Age_Interaction"] = df["BMax_BMin_per_NT"] * np.log1p(df["Wallet_Age_Days"])
    df["Gas_Complexity"] = df["Gas_Price"] * df["Number_of_Inputs"]

    # Logs
    df["Log_Transaction_Value"] = np.log1p(df["Transaction_Value"])
    df["Log_Wallet_Balance"] = np.log1p(df["Wallet_Balance"])
    df["Log_Wallet_Age"] = np.log1p(df["Wallet_Age_Days"])

    # Flags
    df["Is_Young_Wallet"] = (df["Wallet_Age_Days"] < 30).astype(int)
    df["Is_High_Velocity"] = (df["Transaction_Velocity"] > 5.0).astype(int)

    # Cleanup
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    feature_columns = [
        "Transaction_Value", "Transaction_Fees", "Number_of_Inputs", "Number_of_Outputs",
        "Gas_Price", "Wallet_Age_Days", "Wallet_Balance", "Transaction_Velocity",
        "Exchange_Rate", "Final_Balance", "BMax_BMin_per_NT", 
        "Value_to_Fee_Ratio", "Gas_Efficiency", "Input_Output_Ratio", "Balance_Utilization",
        "Tx_Frequency_Score", "Value_Velocity_Interaction", "Volatility_Age_Interaction",
        "Gas_Complexity", "Log_Transaction_Value", "Log_Wallet_Balance", "Is_Young_Wallet", 
        "Is_High_Velocity",
    ]
    return df, feature_columns

# =============================================================================
# 3. TRAINING PIPELINE
# =============================================================================
def train_production_model(csv_path: str, log_dir: str = "training_logs"):
    print("\n" + "=" * 60)
    print("TRAINING RE-CALIBRATED XGBOOST MODEL")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Check Required Columns
    required = ["Transaction_Value", "Transaction_Fees", "Number_of_Inputs", "Number_of_Outputs",
                "Gas_Price", "Wallet_Age_Days", "Wallet_Balance", "Transaction_Velocity",
                "Exchange_Rate", "Final_Balance", "BMax_BMin_per_NT", "Is_Scam"]
    
    for col in required:
        if col not in df.columns:
            print(f"⚠️ Missing {col}, filling with 0")
            df[col] = 0.0

    # Engineer Features
    df, feature_cols = advanced_feature_engineering(df)
    
    X = df[feature_cols]
    y = df["Is_Scam"].astype(int)

    # Calculate Imbalance
    fraud_count = y.sum()
    valid_count = len(y) - fraud_count
    ratio = (valid_count / max(fraud_count, 1)) * 0.8
    print(f"⚖️  Adjusted Class Weight: {ratio:.2f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Model Params
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio, # Uses the dampened ratio
        n_jobs=-1,
        random_state=42,
        eval_metric='auc',
        tree_method='hist'
    )

    # Train
    print("\nTraining Model...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    print("\nEvaluation Results:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ROC-AUC:  {auc:.4f}")
    print("   Confusion Matrix:")
    print(cm)

    # Save
    detector = ProductionXGBoostFraudDetector()
    detector.model = model
    detector.feature_names = feature_cols
    detector.feature_importance_dict = dict(zip(feature_cols, model.feature_importances_))
    
    os.makedirs("saved_models", exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("saved_models", f"xgb_fraud_detector_v5.1_calibrated_{ts_file}.pkl")
    
    joblib.dump(detector, model_path)
    print(f"\n✅ Model Saved: {model_path}")
    print("👉 Update config.py with this filename to fix the '5/10' issue.")

if __name__ == "__main__":
    import sys
    DATA_PATH = "../data/mock_ethereum_fraud_dataset.csv"
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
        
    train_production_model(DATA_PATH)