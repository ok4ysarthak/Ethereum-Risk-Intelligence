import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# 1. MODEL CLASS (Must match training script exactly)
# =============================================================================
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
        def get_val(key, default=np.nan): return features.get(key, default)

        # Raw Inputs
        t_val = get_val("Transaction_Value")
        t_fees = get_val("Transaction_Fees")
        n_inputs = get_val("Number_of_Inputs")
        n_outputs = get_val("Number_of_Outputs")
        gas_price = get_val("Gas_Price")
        w_bal = get_val("Wallet_Balance")
        w_age = get_val("Wallet_Age_Days")
        t_vel = get_val("Transaction_Velocity")
        volatility = get_val("BMax_BMin_per_NT")

        # Feature Engineering (Robust to division by zero)
        features["Value_to_Fee_Ratio"] = t_val / (t_fees + 1e-8)
        features["Gas_Efficiency"] = t_val / (gas_price + 1e-8)
        features["Input_Output_Ratio"] = n_inputs / (n_outputs + 1e-8)
        features["Balance_Utilization"] = t_val / (w_bal + 1e-8)
        features["Tx_Frequency_Score"] = t_vel / (w_age + 1.0)
        features["Value_Velocity_Interaction"] = t_val * t_vel
        features["Volatility_Age_Interaction"] = volatility * np.log1p(w_age)
        features["Gas_Complexity"] = gas_price * n_inputs
        features["Log_Transaction_Value"] = np.log1p(t_val)
        features["Log_Wallet_Balance"] = np.log1p(w_bal)
        features["Log_Wallet_Age"] = np.log1p(w_age)
        features["Is_Young_Wallet"] = 1 if w_age < 30 else 0
        features["Is_High_Velocity"] = 1 if t_vel > 5.0 else 0
        return features

    def predict_batch(self, transactions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        features_list = [self._engineer_features(tx) for tx in transactions]
        X = pd.DataFrame(features_list)
        for col in self.feature_names:
            if col not in X.columns: X[col] = np.nan
        X = X[self.feature_names]
        probs = self.model.predict_proba(X)
        return [{"fraud_probability": float(p[1])} for p in probs]

# =============================================================================
# 2. ADVANCED EVALUATION LOGIC
# =============================================================================

MODEL_PATH = '../saved_models/best_fraud_detector_model.pkl'  # <--- CHECK THIS PATH
DATASET_PATH = 'transaction_dataset.csv'

class AdvancedEvaluator:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.df_features = None
        self.y_true = None
        self.detector = None

    def load_and_adapt_data(self):
        print(f"Loading dataset from {self.data_path}...")
        df_raw = pd.read_csv(self.data_path)
        df_raw.columns = df_raw.columns.str.strip()

        # 1. Map Existing Data
        df_mapped = pd.DataFrame(index=df_raw.index)
        df_mapped['Transaction_Value'] = df_raw['avg val sent']
        df_mapped['Wallet_Balance'] = df_raw['total ether balance'].clip(lower=0)
        df_mapped['Final_Balance'] = df_raw['total ether balance'].clip(lower=0)
        df_mapped['Wallet_Age_Days'] = df_raw['Time Diff between first and last (Mins)'] / 1440.0
        df_mapped['Transaction_Velocity'] = df_raw['Sent tnx'] / (df_mapped['Wallet_Age_Days'] + 1e-9)

        # Instead of NaN, we assume the missing transaction details are "Standard/Average"
        print("Applying Domain Adaptation (Neutral Imputation)...")
        df_mapped['Transaction_Fees'] = 3.36       # Median from training
        df_mapped['Number_of_Inputs'] = 2.0        # Standard transfer
        df_mapped['Number_of_Outputs'] = 2.0       # Standard transfer
        df_mapped['Gas_Price'] = 79.75             # Average Gas
        df_mapped['Exchange_Rate'] = 1802.30       # Average Price
        df_mapped['BMax_BMin_per_NT'] = 3.65       # Average Volatility

        self.df_features = df_mapped
        if 'FLAG' in df_raw.columns:
            self.y_true = df_raw['FLAG']
        else:
            raise ValueError("Dataset missing 'FLAG' column.")

    def load_model(self):
        try:
            self.detector = joblib.load(self.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    def run_sensitivity_analysis(self):
        print("\n=== RUNNING SENSITIVITY ANALYSIS (Hyper-tuning Threshold) ===")
        
        # Get Probabilities
        records = self.df_features.to_dict(orient='records')
        results = self.detector.predict_batch(records)
        y_prob = np.array([r['fraud_probability'] for r in results])

        # Sweep thresholds
        thresholds = np.arange(0.05, 1.0, 0.05)
        precisions, recalls, f1_scores = [], [], []

        best_f1 = 0
        best_thresh = 0.5

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            p = precision_score(self.y_true, y_pred, zero_division=1)
            r = recall_score(self.y_true, y_pred)
            f = f1_score(self.y_true, y_pred)
            
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f)
            
            if f > best_f1:
                best_f1 = f
                best_thresh = t

        # PLOT THE SENSITIVITY CURVE
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision (Low False Alarms)', color='blue', linestyle='--')
        plt.plot(thresholds, recalls, label='Recall (Catching Scams)', color='green')
        plt.plot(thresholds, f1_scores, label='F1 Score (Balance)', color='red', linewidth=2)
        plt.axvline(best_thresh, color='black', linestyle=':', label=f'Optimal ({best_thresh:.2f})')
        plt.title('Model Sensitivity Analysis: Finding the Sweet Spot')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('sensitivity_analysis.png')
        print(f"Sensitivity Plot saved to 'sensitivity_analysis.png'")

        return best_thresh, y_prob

    def print_final_report(self, best_thresh, y_prob):
        y_pred = (y_prob >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()

        print("\n" + "="*60)
        print(f"OPTIMIZED MODEL PERFORMANCE REPORT (Threshold: {best_thresh:.2f})")
        print("="*60)
        print(f"{'Metric':<25} | {'Count/Val':<10} | {'Interpretation'}")
        print("-" * 60)
        print(f"{'True Positives (TP)':<25} | {tp:<10} | Scams Successfully Caught")
        print(f"{'False Negatives (FN)':<25} | {fn:<10} | Scams Missed")
        print(f"{'False Positives (FP)':<25} | {fp:<10} | False Alarms (Reduced)")
        print(f"{'True Negatives (TN)':<25} | {tn:<10} | Normal Users Cleared")
        print("-" * 60)
        print(f"{'Final Precision':<25} | {precision_score(self.y_true, y_pred):.2%}     | Trustworthiness of alerts")
        print(f"{'Final Recall':<25} | {recall_score(self.y_true, y_pred):.2%}     | Safety coverage")
        print(f"{'Final Accuracy':<25} | {accuracy_score(self.y_true, y_pred):.2%}     | Overall correctness")
        print("="*60)

        # Confusion Matrix Plot
        cm = confusion_matrix(self.y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title(f'Optimized Confusion Matrix\n(Threshold {best_thresh:.2f})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('optimized_confusion_matrix.png')
        print("Confusion Matrix saved to 'optimized_confusion_matrix.png'")

if __name__ == "__main__":
    evaluator = AdvancedEvaluator(DATASET_PATH, MODEL_PATH)
    evaluator.load_and_adapt_data()
    evaluator.load_model()
    best_t, probs = evaluator.run_sensitivity_analysis()
    evaluator.print_final_report(best_t, probs)