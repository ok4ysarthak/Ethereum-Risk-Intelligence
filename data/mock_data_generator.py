import random
import time
from backend.wallet_updater import WalletScoreUpdater

# We import the class from the file we just created
# This might show a linting error, but it will run
# from the root directory.

def simulate_wallet_history():
    """
    Simulates a series of transactions for a single wallet
    to test the WalletScoreUpdater logic.
    """
    updater = WalletScoreUpdater()
    
    # Wallet starts with 0 transactions
    current_score = updater.get_initial_score()
    
    print("--- Simulating Wallet History ---")
    print(f"Initial Score: {current_score}\n")
    
    # --- Simulation 1: A new, good user ---
    print("--- Scenario 1: Good User Activity ---")
    for i in range(10):
        print(f"Transaction {i+1}:")
        # Simulating a low-risk transaction
        mock_risk = random.uniform(0.0, 0.2) 
        mock_value = random.uniform(100, 1000) # $100 - $1000
        
        current_score = updater.calculate_new_score(
            current_score, 
            mock_risk, 
            mock_value
        )
        print(f"New Score: {current_score:.2f}\n")
        time.sleep(0.1)

    # --- Simulation 2: User interacts with a risky protocol ---
    print("--- Scenario 2: High Risk Transaction ---")
    print("Wallet interacts with a known-bad contract...")
    mock_risk = 0.95 # 95% risky
    mock_value = 50.0
    
    current_score = updater.calculate_new_score(
        current_score,
        mock_risk,
        mock_value
    )
    print(f"New Score after high-risk: {current_score:.2f}\n")
    time.sleep(0.1)

    # --- Simulation 3: User rehabs their score ---
    print("--- Scenario 3: Score Rehabilitation ---")
    for i in range(5):
        print(f"Rehab Transaction {i+1}:")
        mock_risk = random.uniform(0.0, 0.1) 
        mock_value = random.uniform(50, 200)
        
        current_score = updater.calculate_new_score(
            current_score, 
            mock_risk, 
            mock_value
        )
        print(f"New Score: {current_score:.2f}\n")
        time.sleep(0.1)
        
    print("--- Simulation Complete ---")
    print(f"Final Wallet Score: {current_score:.2f}")

if __name__ == "__main__":
    simulate_wallet_history()