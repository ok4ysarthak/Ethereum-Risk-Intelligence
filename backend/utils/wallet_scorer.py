# backend/wallet_scorer.py
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class WalletScorer:
    def __init__(self, fraud_detector):
        self.fraud_detector = fraud_detector
        self.wallet_history = defaultdict(list)
        self.wallet_scores = {}
    
    def calculate_wallet_score(self, address, transaction_history):
        """Calculate comprehensive wallet score"""
        if not transaction_history:
            return 0.5  # Neutral score
        
        # Historical fraud risk
        fraud_risks = []
        for tx in transaction_history:
            try:
                risk = self.fraud_detector.predict_fraud_risk(tx)
                fraud_risks.append(risk)
            except Exception as e:
                logger.warning(f"Error calculating risk for transaction: {e}")
                continue
        
        avg_fraud_risk = np.mean(fraud_risks) if fraud_risks else 0.5
        
        # Transaction patterns
        total_transactions = len(transaction_history)
        total_value = sum(tx.get('Transaction_Value', 0) for tx in transaction_history)
        
        # Frequency score (more transactions = higher trust, up to a point)
        frequency_score = min(total_transactions / 100, 1.0)
        
        # Value consistency score
        values = [tx.get('Transaction_Value', 0) for tx in transaction_history]
        value_std = np.std(values) if len(values) > 1 else 0
        value_consistency = 1 / (1 + value_std / (np.mean(values) + 1e-8))
        
        # Wallet age factor (assuming it's in the first transaction)
        wallet_age = transaction_history[0].get('Wallet_Age_Days', 1)
        age_factor = min(wallet_age / 365, 1.0)  # Normalize to 1 year
        
        # Combine factors (lower fraud risk = higher score)
        wallet_score = (
            (1 - avg_fraud_risk) * 0.4 +  # Fraud risk component
            frequency_score * 0.2 +        # Transaction frequency
            value_consistency * 0.2 +      # Value consistency
            age_factor * 0.2               # Wallet age
        )
        
        return max(0, min(1, wallet_score))  # Clamp between 0 and 1
    
    def update_wallet_history(self, address, transaction_data):
        """Update wallet transaction history"""
        self.wallet_history[address].append(transaction_data)
        # Keep only recent transactions (last 1000)
        if len(self.wallet_history[address]) > 1000:
            self.wallet_history[address] = self.wallet_history[address][-1000:]
    
    def get_wallet_score(self, address):
        """Get current wallet score"""
        if address in self.wallet_scores:
            return self.wallet_scores[address]
        return 0.5  # Default neutral score

# Don't create an instance here - this was causing the error
# wallet_scorer = WalletScorer(detector)  # <-- This line should be removed
