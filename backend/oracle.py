# backend/oracle.py
import os
import json
import time
import random
import requests
from web3 import Web3
from dotenv import load_dotenv
import logging

# Import our Wallet Score Updater logic
from wallet_updater import WalletScoreUpdater

# --- 1. Load Configuration ---
load_dotenv(dotenv_path='../.env')

CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY")
API_URL = "http://127.0.0.1:5000/predict/transaction"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not all([CONTRACT_ADDRESS, SEPOLIA_RPC_URL, ORACLE_PRIVATE_KEY]):
    raise EnvironmentError("Missing one or more environment variables.")

# --- 2. Initialize Web3 and Contract ---
logger.info("Connecting to Web3...")
w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)
w3.eth.default_account = oracle_account.address

logger.info(f"Oracle wallet address: {oracle_account.address}")

# Load Contract ABI (Application Binary Interface)
# We get this from the Hardhat compilation
artifact_path = '../artifacts/contracts/TrustScore.sol/FraudDetection.json'
try:
    with open(artifact_path) as f:
        contract_json = json.load(f)
        contract_abi = contract_json['abi']
except FileNotFoundError:
    logger.error(f"❌ Error: Contract artifact not found at {artifact_path}")
    logger.error("Please make sure you have compiled your Hardhat project.")
    exit(1)

# Create contract instance
fraud_detection_contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)
logger.info(f"Connected to FraudDetection contract at {CONTRACT_ADDRESS}")

# Initialize our score calculator
score_updater = WalletScoreUpdater()

# --- 3. Oracle Main Loop ---

def generate_mock_ethereum_transaction():
    """Generates a mock Ethereum transaction and a random wallet address."""
    # A random "user" wallet to score
    mock_wallet = w3.eth.account.create().address 
    
    # Mock features for our ML model (new Ethereum features)
    mock_features = {
        "Transaction_Value": round(random.uniform(0.1, 10.0), 4),  # ETH value
        "Transaction_Fees": round(random.uniform(0.001, 0.01), 6),  # ETH fees
        "Number_of_Inputs": random.randint(1, 5),
        "Number_of_Outputs": random.randint(1, 10),
        "Gas_Price": random.randint(10, 100),  # Gwei
        "Wallet_Age_Days": random.randint(1, 365 * 5),  # Up to 5 years
        "Wallet_Balance": round(random.uniform(0.1, 100.0), 4),  # ETH balance
        "Transaction_Velocity": round(random.uniform(0.1, 10.0), 2),  # Transactions per day
        "Exchange_Rate": random.randint(1500, 3000)  # ETH/USD price
    }
    
    # Mock a "high risk" transaction 10% of the time
    if random.random() < 0.1:
        logger.info("...Generating a HIGH RISK mock transaction...")
        mock_features["Transaction_Value"] = round(random.uniform(50.0, 1000.0), 4)
        mock_features["Number_of_Outputs"] = random.randint(20, 100)
        mock_features["Gas_Price"] = random.randint(200, 500)

    return mock_wallet, mock_features

def update_transaction_risk_on_chain(tx_hash, from_address, to_address, value, risk_score):
    """Update transaction risk score on blockchain"""
    try:
        # Convert risk score to integer (0-100 scale)
        risk_score_int = int(risk_score * 100)
        
        # Build the transaction
        nonce = w3.eth.get_transaction_count(oracle_account.address)
        tx_data = fraud_detection_contract.functions.updateTransactionRisk(
            tx_hash,  # bytes32 txHash
            from_address,  # address from
            to_address,    # address to
            int(value * 10**18),  # uint256 value (wei)
            risk_score_int  # uint256 riskScore
        ).build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 300000,
            'gasPrice': w3.eth.gas_price
        })

        # Sign the transaction
        signed_tx = w3.eth.account.sign_transaction(tx_data, ORACLE_PRIVATE_KEY)

        # Send the transaction
        tx_hash_result = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        logger.info(f"    Transaction risk update sent, hash: {tx_hash_result.hex()}")
        
        # Wait for it to be mined
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash_result, timeout=120)
        
        logger.info(f"    ✅ Transaction risk update mined! Block: {tx_receipt.blockNumber}")
        return tx_hash_result.hex()
        
    except Exception as e:
        logger.error(f"❌ Error updating transaction risk on chain: {e}")
        return None

def update_wallet_score_on_chain(wallet_address, trust_score):
    """Update wallet trust score on blockchain"""
    try:
        # Convert trust score to integer (0-100 scale)
        trust_score_int = int(trust_score * 100)
        
        # Build the transaction
        nonce = w3.eth.get_transaction_count(oracle_account.address)
        tx_data = fraud_detection_contract.functions.updateWalletScore(
            wallet_address,    # address wallet
            trust_score_int    # uint256 trustScore
        ).build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price
        })

        # Sign the transaction
        signed_tx = w3.eth.account.sign_transaction(tx_data, ORACLE_PRIVATE_KEY)

        # Send the transaction
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        logger.info(f"    Wallet score update sent, hash: {tx_hash.hex()}")
        
        # Wait for it to be mined
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        
        logger.info(f"    ✅ Wallet score update mined! Block: {tx_receipt.blockNumber}")
        return tx_hash.hex()
        
    except Exception as e:
        logger.error(f"❌ Error updating wallet score on chain: {e}")
        return None

def main_loop():
    logger.info("\n--- Oracle Service Started: Listening for 'transactions' ---")
    while True:
        try:
            # --- A. Simulate a new transaction ---
            wallet_to_score, tx_features = generate_mock_ethereum_transaction()
            logger.info(f"\n[1] New 'transaction' from wallet: {wallet_to_score}")
            logger.info(f"    Features: {tx_features}")

            # --- B. Get ML Risk Score from API ---
            response = requests.post(API_URL, json={"features": tx_features})
            if response.status_code != 200:
                logger.error(f"    ❌ Error from API: {response.text}")
                time.sleep(10)
                continue
            
            risk_data = response.json()
            risk_probability = risk_data['risk_probability']
            risk_level = risk_data['risk_level']
            
            logger.info(f"[2] Got ML prediction. Risk probability: {risk_probability:.4f} ({risk_level})")

            # --- C. Get Current Wallet Score & Calculate New Score ---
            
            # Fetch current wallet score from the blockchain (a read-only call)
            try:
                current_score_int = fraud_detection_contract.functions.getWalletTrustScore(wallet_to_score).call()
                current_score = float(current_score_int) / 100.0  # Convert from integer scale
            except:
                current_score = 0.5  # Default neutral score
            
            if current_score == 0:
                current_score = score_updater.get_initial_score()
                logger.info(f"[3] Wallet is new. Assigning initial score: {current_score}")
            else:
                logger.info(f"[3] Got current score from chain: {current_score}")

            # Use our logic to calculate the new score
            mock_tx_value_usd = tx_features["Transaction_Value"] * tx_features["Exchange_Rate"]
            new_score = score_updater.calculate_new_score(
                current_score,
                risk_probability,
                mock_tx_value_usd
            )
            new_score_int = int(new_score * 100)  # Convert to integer scale for Solidity
            
            logger.info(f"[4] Calculated new score: {new_score_int/100:.2f}")

            # --- D. Update Transaction Risk on Blockchain ---
            # Generate a mock transaction hash
            mock_tx_hash = Web3.keccak(text=f"{wallet_to_score}_{time.time()}")[:32]
            
            tx_update_result = update_transaction_risk_on_chain(
                mock_tx_hash,
                wallet_to_score,
                w3.eth.account.create().address,  # Random to address
                tx_features["Transaction_Value"],
                risk_probability
            )

            # --- E. Update Wallet Score on Blockchain ---
            if new_score_int != current_score_int:
                logger.info("[5] Score changed. Updating wallet score on chain...")
                wallet_update_result = update_wallet_score_on_chain(wallet_to_score, new_score)
            else:
                logger.info("[5] Score unchanged. No wallet update needed.")

            # Wait before the next loop
            logger.info("\n--- Waiting for next 'transaction' (60s) ---")
            time.sleep(60)

        except Exception as e:
            logger.error(f"❌ An error occurred in the main loop: {e}")
            logger.exception(e)  # This will print the full traceback
            time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    main_loop()
