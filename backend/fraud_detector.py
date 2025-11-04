# backend/fraud_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class EthereumFraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def preprocess_single_transaction(self, transaction_data):
        """Preprocess single transaction data"""
        # Convert to DataFrame if it's a dict
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame(transaction_data)
            
        # Feature engineering (same as training)
        df['Value_to_Fee_Ratio'] = df['Transaction_Value'] / (df['Transaction_Fees'] + 1e-8)
        df['Input_Output_Ratio'] = df['Number_of_Inputs'] / (df['Number_of_Outputs'] + 1e-8)
        df['Gas_Efficiency'] = df['Transaction_Value'] / (df['Gas_Price'] + 1e-8)
        df['Balance_Turnover'] = df['Transaction_Value'] / (df['Wallet_Balance'] + 1e-8)
        
        return df
    
    def predict_fraud_risk(self, transaction_data):
        """Predict fraud risk for a transaction"""
        try:
            # Preprocess transaction data
            df = self.preprocess_single_transaction(transaction_data)
            
            feature_columns = [
                'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs', 
                'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days', 
                'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate',
                'Value_to_Fee_Ratio', 'Input_Output_Ratio', 
                'Gas_Efficiency', 'Balance_Turnover'
            ]
            
            # Ensure all features are present (fill missing with 0)
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            X = df[feature_columns].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get prediction probability (fraud probability)
            risk_score = self.model.predict_proba(X_scaled)[0][1]
            
            return float(risk_score)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.5  # Return neutral score on error
