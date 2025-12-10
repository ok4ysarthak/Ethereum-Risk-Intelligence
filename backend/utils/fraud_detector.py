import joblib
import os
import sys
import pandas as pd
import math
from dotenv import load_dotenv
import requests
import time
from typing import Dict, Tuple

load_dotenv()


# ---------------------------------------------------------
# LiveWalletFeatureExtractor (from live_features.py)
# ---------------------------------------------------------
class LiveWalletFeatureExtractor:
    def __init__(self, ETHERSCAN_API_KEY: str):
        self.api_key = ETHERSCAN_API_KEY
        self.base_url = "https://api.etherscan.io/KJ8SVFE6Z6N1VW9P5ENKU4HABN1R2B1ET3"
        self.session = requests.Session()

    def get_features(self, address: str) -> Dict[str, float]:
        """
        Fetches live data and calculates:
        - Final_Balance (Current ETH balance)
        - BMax_BMin_per_NT (Volatility)
        - Wallet_Age_Days
        - Transaction_Velocity
        """
        # 1. Fetch Current Balance (Source of Truth)
        current_balance = self._fetch_current_balance(address)
        
        # 2. Fetch Full History (Normal + Internal Txs)
        # We need both to accurately reconstruct the balance curve
        normal_txs = self._fetch_tx_list(address, "txlist")
        internal_txs = self._fetch_tx_list(address, "txlistinternal")
        
        # 3. Merge and Sort by Timestamp
        all_txs = normal_txs + internal_txs
        # Sort is critical for replaying history in order
        all_txs.sort(key=lambda x: int(x['timeStamp']))
        
        # 4. Reconstruct Balance History
        if not all_txs:
            # New/Empty wallet case
            return {
                "Final_Balance": current_balance,
                "BMax_BMin_per_NT": 0.0,
                "Wallet_Age_Days": 0.0,
                "Transaction_Velocity": 0.0
            }

        b_max, b_min, count = self._calculate_volatility_stats(address, all_txs, current_balance)
        
        # 5. Calculate Research Features
        # Formula: (B_max - B_min) / N_T
        volatility = (b_max - b_min) / count if count > 0 else 0.0
        
        # Calculate Age
        first_tx_time = int(all_txs[0]['timeStamp'])
        last_tx_time = int(all_txs[-1]['timeStamp'])
        age_days = (last_tx_time - first_tx_time) / 86400
        if age_days < 0.01: age_days = 0.01 # Prevent div by zero
        
        return {
            "Final_Balance": current_balance,
            "BMax_BMin_per_NT": volatility,
            "Wallet_Age_Days": age_days,
            "Transaction_Velocity": count / age_days
        }

    def _fetch_current_balance(self, address):
        """Get the absolute current balance in ETH"""
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
            "apikey": self.api_key
        }
        try:
            resp = self.session.get(self.base_url, params=params, timeout=5).json()
            return float(resp['result']) / 1e18
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0

    def _fetch_tx_list(self, address, action):
        """Fetch list of transactions"""
        params = {
            "module": "account",
            "action": action,
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": self.api_key
        }
        try:
            resp = self.session.get(self.base_url, params=params, timeout=10).json()
            if resp['message'] == 'OK':
                return resp['result']
            return []
        except Exception:
            return []

    def _calculate_volatility_stats(self, address, sorted_txs, final_balance) -> Tuple[float, float, int]:
        """
        Reconstructs the balance timeline BACKWARDS from the current known balance.
        This is more accurate than starting from 0 because of complex edge cases (airdrops, genesis).
        """
        current_bal = final_balance
        balances = [current_bal]
        address_lower = address.lower()
        
        # Iterate BACKWARDS (from newest to oldest)
        for tx in reversed(sorted_txs):
            value = float(tx['value']) / 1e18
            
            # Internal transactions often don't have gasPrice/gasUsed fields in the same way,
            # so we treat gas carefully.
            gas_cost = 0.0
            if 'gasPrice' in tx and 'gasUsed' in tx:
                gas_cost = (float(tx['gasPrice']) * float(tx['gasUsed'])) / 1e18

            # Logic: We are stepping BACK in time.
            # If money WAS received (to), in the past we had LESS.
            if tx['to'].lower() == address_lower:
                current_bal -= value
            
            # If money WAS sent (from), in the past we had MORE.
            elif tx['from'].lower() == address_lower:
                current_bal += (value + gas_cost)
            
            # Sanity check: Balance shouldn't technically be negative, 
            # but API data gaps happen. We clamp to 0 for stability.
            if current_bal < 0: current_bal = 0
            
            balances.append(current_bal)
            
        # Metrics
        b_max = max(balances)
        b_min = min(balances)
        count = len(sorted_txs)
        
        return b_max, b_min, count


# ---------------------------------------------------------
# 1. HELPER CLASS (MUST MATCH TRAINING SCRIPT)
# ---------------------------------------------------------
# This class definition allows joblib to load the custom object
class XGBoostFraudDetector:
    def __init__(self, model, feature_names, threshold=0.5):
        self.model = model
        self.feature_names = feature_names
        self.threshold = threshold

    def predict_risk_score(self, tx_features: dict) -> dict:
        # Convert to DataFrame
        df = pd.DataFrame([tx_features])
        
        # Feature Engineering (On-the-fly)
        # Ratios
        df['Value_to_Fee_Ratio'] = df['Transaction_Value'] / (df['Transaction_Fees'].replace(0, 1e-8) + 1e-8)
        df['Gas_Efficiency'] = df['Transaction_Value'] / (df['Gas_Price'].replace(0, 1e-8) + 1e-8)
        
        # Fallbacks for research columns if missing (handled by wrapper)
        if 'Final_Balance' not in df.columns: df['Final_Balance'] = 0.0
        if 'BMax_BMin_per_NT' not in df.columns: df['BMax_BMin_per_NT'] = 0.0
        if 'Wallet_Age_Days' not in df.columns: df['Wallet_Age_Days'] = 0.0
        if 'Transaction_Velocity' not in df.columns: df['Transaction_Velocity'] = 0.0
        
        df = df.fillna(0)
        
        # Select ordered features
        try:
            X = df[self.feature_names]
            # Ensure all cols exist
            for col in self.feature_names:
                if col not in X.columns: X[col] = 0.0
            X = X[self.feature_names]
        except KeyError as e:
            return {"error": str(e), "risk_score": 1}

        # Predict
        prob_fraud = float(self.model.predict_proba(X)[:, 1][0])
        
        # Risk Score (1-10)
        if prob_fraud <= 0.01: risk_score = 1
        else:
            risk_score = math.ceil(prob_fraud * 10)
            if risk_score > 10: risk_score = 10

        is_fraud = 1 if prob_fraud >= self.threshold else 0

        return {
            "fraud_probability": round(prob_fraud, 4),
            "risk_score": int(risk_score),
            "is_fraud_label": is_fraud
        }


# ---------------------------------------------------------
# 2. MAIN BACKEND INTERFACE
# ---------------------------------------------------------
class EthereumFraudDetector:
    def __init__(self, model_path='../ml_models/saved_models/noise_fraud_detector_model.pkl'):
        """
        Initialize the detector and the live stats fetcher.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_file = os.path.join(base_dir, model_path)
        
        # Initialize Etherscan Fetcher
        api_key = os.getenv("ETHERSCAN_API_KEY")
        self.stats_fetcher = LiveWalletFeatureExtractor(api_key)

        try:
            self.detector_backend = joblib.load(self.model_file)
            print(f"✅ Loaded XGBoost Model from {self.model_file}")
        except Exception as e:
            print(f"🚨 Model Load Error: {e}")
            self.detector_backend = None

    def predict_fraud_risk(self, transaction_data: dict):
        """
        Main entry point.
        1. Checks for 'Address' in input.
        2. If live metadata is missing, FETCH it from Etherscan.
        3. Runs Prediction.
        """
        if self.detector_backend is None:
            return {"error": "Model not loaded", "risk_score": 0}

        # --- STEP 1: ENRICH DATA ---
        address = transaction_data.get('Address') or transaction_data.get('from')
        
        needed_keys = ['Final_Balance', 'BMax_BMin_per_NT', 'Wallet_Age_Days', 'Transaction_Velocity']
        needs_live = any(k not in transaction_data for k in needed_keys)

        if address and needs_live:
            print(f"🔍 Fetching live stats for {address}...")
            live_stats = self.stats_fetcher.get_features(address)
            
            # Merge live stats into transaction data
            transaction_data.update(live_stats)
            print("   ✅ Live stats merged.")

        # Ensure all required live metadata keys exist
        for k in needed_keys:
            transaction_data.setdefault(k, 0.0)

        # --- STEP 2: PREDICT ---
        return self.detector_backend.predict_risk_score(transaction_data)


# Test Block
if __name__ == "__main__":
    detector = EthereumFraudDetector()
    
    # Example: Just raw transaction info, NO research features
    raw_tx = {
        'Address': '0xdAC17F958D2ee523a2206206994597C13D831ec7', # USDT Contract (Example)
        'Transaction_Value': 5.0,
        'Transaction_Fees': 0.005,
        'Number_of_Inputs': 1,
        'Number_of_Outputs': 1,
        'Gas_Price': 20.0,
        'Exchange_Rate': 3000.0,
        # Note: Wallet_Age, Final_Balance, BMax are MISSING. 
        # The detector should fetch them automatically.
    }
    
    print("\nTest Prediction (Auto-Fetch):")
    result = detector.predict_fraud_risk(raw_tx)
    print(result)
