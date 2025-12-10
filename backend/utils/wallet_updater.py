# backend/wallet_updater.py
"""
This module defines the logic for updating a wallet's trust score.
It is a heuristic algorithm, not a trained ML model.
"""

# Define scoring constants
INITIAL_SCORE = 0.5         # Score for a brand new wallet (0.0 to 1.0 scale)
MAX_SCORE = 1.0             # Maximum possible score
MIN_SCORE = 0.0             # Minimum possible score

# Penalties & Bonuses (adjusted for 0-1 scale)
HIGH_RISK_TX_PENALTY_FACTOR = 0.3   # 30% penalty for high risk
MEDIUM_RISK_TX_PENALTY_FACTOR = 0.1 # 10% penalty for medium risk
LOW_RISK_TX_BONUS = 0.05            # 5% bonus for low risk
ACTIVITY_BONUS = 0.01               # 1% bonus for activity

# Value-based bonus (can be tuned)
# We add a small fraction of the normalized tx value to the score
VALUE_BONUS_FACTOR = 0.0001         # e.g., $1000 tx adds 0.1 points (10% of max)

class WalletScoreUpdater:
    
    def __init__(self):
        """Initializes the score updater."""
        # In the future, this could load dynamic config
        pass

    def calculate_new_score(self, current_score, tx_risk_probability, tx_value_usd):
        """
        Calculates a wallet's new trust score based on a new transaction.
        
        Args:
            current_score (float): The wallet's score before this transaction (0.0 to 1.0).
            tx_risk_probability (float): A value from 0.0 to 1.0 from Model 1.
            tx_value_usd (float): The USD value of the transaction.

        Returns:
            float: The new, clamped trust score (0.0 to 1.0).
        """
        new_score = float(current_score)

        # 1. Apply Penalties based on risk
        if tx_risk_probability > 0.8:  # High Risk
            penalty = new_score * HIGH_RISK_TX_PENALTY_FACTOR
            new_score -= penalty
            # print(f"  - Applied HIGH risk penalty: -{penalty:.4f} points")
            
        elif tx_risk_probability > 0.5: # Medium Risk
            penalty = new_score * MEDIUM_RISK_TX_PENALTY_FACTOR
            new_score -= penalty
            # print(f"  - Applied MEDIUM risk penalty: -{penalty:.4f} points")

        # 2. Apply Bonuses for low-risk activity
        else:
            # Add flat bonus for low-risk tx
            new_score += LOW_RISK_TX_BONUS
            # print(f"  + Applied LOW risk bonus: +{LOW_RISK_TX_BONUS:.4f} points")
            
            # Add small bonus for general activity
            new_score += ACTIVITY_BONUS
            # print(f"  + Applied ACTIVITY bonus: +{ACTIVITY_BONUS:.4f} points")

            # Add value-based bonus (normalized to prevent huge bonuses)
            normalized_value = min(tx_value_usd / 10000.0, 1.0)  # Cap at $10,000 equivalent
            value_bonus = normalized_value * VALUE_BONUS_FACTOR * 100  # Scale appropriately
            new_score += value_bonus
            # print(f"  + Applied VALUE bonus: +{value_bonus:.4f} points")

        # 3. Clamp the score between MIN and MAX bounds
        if new_score > MAX_SCORE:
            new_score = MAX_SCORE
            # print(f"  ! Score clamped to MAX: {MAX_SCORE}")
        elif new_score < MIN_SCORE:
            new_score = MIN_SCORE
            # print(f"  ! Score clamped to MIN: {MIN_SCORE}")
            
        return new_score

    def get_initial_score(self):
        """Returns the score for a wallet with 0 transactions."""
        return INITIAL_SCORE
