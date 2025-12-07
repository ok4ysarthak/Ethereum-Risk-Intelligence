"""
Production-Grade XGBoost Ethereum Fraud Detection Model (Robust Version)
========================================================================

- IMPROVED: Implements "Domain Randomization" to simulate real-world noise.
- IMPROVED: Simulates "Feature Dropout" (missing data) so the model survives API failures.
- Preserves exact inference logic for compatibility with app.py/oracle.py.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score
)
import joblib
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PRODUCTION MODEL CLASS (UNCHANGED)
# =============================================================================
class ProductionXGBoostFraudDetector:
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importance_dict: Dict[str, float] = {}
        self.training_metadata: Dict[str, Any] = {}
        self.threshold: float = 0.5
        self.model_version: str = "5.0-Robust"

    def _engineer_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        features = raw_features.copy()
        
        # Helper to safely get values
        def get(k): return features.get(k, 0.0)

        # Derived ratios (Robust to division by zero)
        features["Value_to_Fee_Ratio"] = get("Transaction_Value") / (get("Transaction_Fees") + 1e-8)
        features["Gas_Efficiency"] = get("Transaction_Value") / (get("Gas_Price") + 1e-8)
        features["Input_Output_Ratio"] = get("Number_of_Inputs") / (get("Number_of_Outputs") + 1e-8)
        features["Balance_Utilization"] = get("Transaction_Value") / (get("Wallet_Balance") + 1e-8)
        features["Tx_Frequency_Score"] = get("Transaction_Velocity") / (get("Wallet_Age_Days") + 1.0)

        # Interaction terms
        features["Value_Velocity_Interaction"] = get("Transaction_Value") * get("Transaction_Velocity")
        features["Volatility_Age_Interaction"] = get("BMax_BMin_per_NT") * np.log1p(get("Wallet_Age_Days"))
        features["Gas_Complexity"] = get("Gas_Price") * get("Number_of_Inputs")

        # Log transforms
        features["Log_Transaction_Value"] = np.log1p(get("Transaction_Value"))
        features["Log_Wallet_Balance"] = np.log1p(get("Wallet_Balance"))
        features["Log_Wallet_Age"] = np.log1p(get("Wallet_Age_Days"))

        # Binary risk flags
        features["Is_Young_Wallet"] = int(get("Wallet_Age_Days") < 30)
        features["Is_High_Velocity"] = int(get("Transaction_Velocity") > 5.0)

        return features

    def predict_risk_score(self, tx_features: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None: raise ValueError("Model not loaded.")
        features = self._engineer_features(tx_features)
        X = pd.DataFrame([features])[self.feature_names]
        proba = self.model.predict_proba(X)[0]
        return {
            "fraud_probability": float(proba[1]),
            "risk_score": self._probability_to_risk_score(float(proba[1])),
            "is_fraud_label": int(float(proba[1]) >= self.threshold),
            "model_version": self.model_version
        }

    def predict_batch(self, transactions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        if self.model is None: raise ValueError("Model not loaded.")
        features_list = [self._engineer_features(tx) for tx in transactions]
        X = pd.DataFrame(features_list)[self.feature_names]
        probs = self.model.predict_proba(X)
        return [{"fraud_probability": float(p[1]), "is_fraud_label": int(p[1] >= self.threshold)} for p in probs]

    def _probability_to_risk_score(self, p: float) -> int:
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

    def _get_risk_category(self, score: int) -> str:
        if score <= 3: return "LOW"
        elif score <= 6: return "MEDIUM"
        elif score <= 8: return "HIGH"
        else: return "CRITICAL"

# =============================================================================
# 2. ROBUSTNESS AUGMENTATION FUNCTIONS (NEW!)
# =============================================================================

def add_training_noise(X: pd.DataFrame, noise_level: float = 0.02) -> pd.DataFrame:
    """
    Injects random Gaussian noise into continuous features to simulate
    real-world volatility and prevent overfitting to mock data.
    """
    print(f"   >>> Injecting {noise_level*100}% Gaussian Noise for Robustness...")
    X_noisy = X.copy()
    
    # Only add noise to float columns, ignore binary flags
    float_cols = X.select_dtypes(include=['float64', 'float32']).columns
    
    for col in float_cols:
        # Calculate standard deviation for scale
        std = X[col].std()
        if std == 0: continue
        
        # Generate noise: Mean=0, Sigma=noise_level * column_std
        noise = np.random.normal(0, std * noise_level, size=len(X))
        X_noisy[col] = X[col] + noise
        
    return X_noisy

def simulate_missing_data(X: pd.DataFrame, drop_prob: float = 0.05) -> pd.DataFrame:
    """
    Randomly sets features to 0 to simulate API failures (Missing Data).
    Forces model to rely on partial information.
    """
    print(f"   >>> Simulating {drop_prob*100}% Feature Dropout (API Failures)...")
    X_dropped = X.copy()
    
    # Create a mask of values to drop
    mask = np.random.rand(*X_dropped.shape) < drop_prob
    
    # Apply mask (Set to 0, which is the standard fillna value)
    X_dropped[mask] = 0
    return X_dropped

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def advanced_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print("\nAdvanced feature engineering...")
    df = df.copy()

    # Calculations
    df["Value_to_Fee_Ratio"] = df["Transaction_Value"] / (df["Transaction_Fees"] + 1e-8)
    df["Gas_Efficiency"] = df["Transaction_Value"] / (df["Gas_Price"] + 1e-8)
    df["Input_Output_Ratio"] = df["Number_of_Inputs"] / (df["Number_of_Outputs"] + 1e-8)
    df["Balance_Utilization"] = df["Transaction_Value"] / (df["Wallet_Balance"] + 1e-8)
    df["Tx_Frequency_Score"] = df["Transaction_Velocity"] / (df["Wallet_Age_Days"] + 1.0)
    df["Value_Velocity_Interaction"] = df["Transaction_Value"] * df["Transaction_Velocity"]
    df["Volatility_Age_Interaction"] = df["BMax_BMin_per_NT"] * np.log1p(df["Wallet_Age_Days"])
    df["Gas_Complexity"] = df["Gas_Price"] * df["Number_of_Inputs"]
    df["Log_Transaction_Value"] = np.log1p(df["Transaction_Value"])
    df["Log_Wallet_Balance"] = np.log1p(df["Wallet_Balance"])
    df["Log_Wallet_Age"] = np.log1p(df["Wallet_Age_Days"])
    df["Is_Young_Wallet"] = (df["Wallet_Age_Days"] < 30).astype(int)
    df["Is_High_Velocity"] = (df["Transaction_Velocity"] > 5.0).astype(int)

    feature_columns = [
        "Transaction_Value", "Transaction_Fees", "Number_of_Inputs", "Number_of_Outputs",
        "Gas_Price", "Wallet_Age_Days", "Wallet_Balance", "Transaction_Velocity",
        "Exchange_Rate", 
        # "Final_Balance",
        "BMax_BMin_per_NT",
        "Value_to_Fee_Ratio", "Gas_Efficiency", "Input_Output_Ratio", "Balance_Utilization", 
        "Tx_Frequency_Score", "Value_Velocity_Interaction", "Volatility_Age_Interaction", 
        "Gas_Complexity", "Log_Transaction_Value", "Log_Wallet_Balance", 
        "Is_Young_Wallet", "Is_High_Velocity"
    ]
    return df, feature_columns

# =============================================================================
# 4. XGBOOST CONFIG (BALANCED)
# =============================================================================

def optimize_xgboost_params(scale_pos_weight: float) -> Dict[str, Any]:
    print("\nConfiguring XGBoost hyperparameters (Robust Mode)...")
    return {
        "n_estimators": 600,        # More trees to learn subtle patterns
        "learning_rate": 0.015,     # Slower learning
        "max_depth": 5,             
        "colsample_bytree": 0.4,    
        "colsample_bylevel": 0.5,
        "subsample": 0.7,
        "reg_alpha": 2.0,           
        "reg_lambda": 10.0,        
        "min_child_weight": 10,
        "gamma": 1.0,               
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "auc",
    }

# =============================================================================
# 5. TRAINING PIPELINE
# =============================================================================

def train_production_model(csv_path: str, log_dir: str = "training_logs") -> ProductionXGBoostFraudDetector:
    print("\n" + "=" * 80)
    print("TRAINING ROBUST XGBOOST FRAUD MODEL")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    required = ["Transaction_Value", "Transaction_Fees", "Number_of_Inputs", "Number_of_Outputs",
                "Gas_Price", "Wallet_Age_Days", "Wallet_Balance", "Transaction_Velocity",
                "Exchange_Rate", "BMax_BMin_per_NT", "Is_Scam"]
    df[required] = df[required].fillna(0)
    df["Is_Scam"] = df["Is_Scam"].astype(int)
    print("Using actual Final_Balance/Volatility from dataset.")
    df, feature_columns = advanced_feature_engineering(df)

    X = df[feature_columns].copy().fillna(0)
    y = df["Is_Scam"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
# =========================================================================
# 4. APPLY ROBUSTNESS AUGMENTATION (Training Set Only)
# =========================================================================
    print("\nApplying Data Augmentation to Training Set:")
    
    X_train_aug = add_training_noise(X_train, noise_level=0.03) # 3% jitter
    
    X_train_broken = simulate_missing_data(X_train_aug, drop_prob=0.05) # 5% missing
    
    X_train_final = pd.concat([X_train, X_train_aug, X_train_broken])
    y_train_final = pd.concat([y_train, y_train, y_train])
    
    print(f"Augmented Training Set: {len(X_train)} -> {len(X_train_final)} rows")
    # =========================================================================

    # 5. Train Model
    fraud_ratio = (y == 1).sum() / len(y)
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    params = optimize_xgboost_params(scale_pos_weight)
    
    print("\nTraining final XGBoost model...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_final, y_train_final, eval_set=[(X_test, y_test)], verbose=False)
    
    metrics = evaluate_model(model, X_test, y_test, feature_columns)
    
    detector = ProductionXGBoostFraudDetector()
    detector.model = model
    detector.feature_names = feature_columns
    detector.feature_importance_dict = {fi["feature"]: float(fi["importance"]) for fi in metrics["feature_importance"]}
    detector.training_metadata = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "metrics": metrics}

    os.makedirs("saved_models", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("saved_models", f"xgb_fraud_detector_robust_{ts}.pkl")
    joblib.dump(detector, path)
    print(f"\n✅ Robust Model saved to: {path}")
    return detector

def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nEvaluation Results (Clean Test Set):")
    print(f"Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    fi = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    print("\nTop Features:")
    print(fi.head(5))
    
    return {"accuracy": float(acc), "auc": float(auc), "feature_importance": fi.to_dict("records")}

if __name__ == "__main__":
    train_production_model("../data/mock_ethereum_fraud_dataset.csv")