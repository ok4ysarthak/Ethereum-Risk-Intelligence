# backend/oracle.py
import os
import json
import time
import logging
import requests
from web3 import Web3
from dotenv import load_dotenv

# --- LOCAL IMPORTS ---
from utils.wallet_updater import WalletScoreUpdater
from utils.feature_calculator import RealTimeFeatureCalculator

# --- CONFIGURATION ---
load_dotenv(dotenv_path='../.env', override=True)  # Add override=True
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ORACLE_PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY")
API_URL = os.getenv("ML_API_URL", "http://127.0.0.1:5000")

# Settings
POLL_INTERVAL_SECONDS = 15
BATCH_SIZE = 10
API_PREDICT_URL = f"{API_URL.rstrip('/')}/predict/transaction"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Oracle")

# --- INITIALIZATION ---
if not all([CONTRACT_ADDRESS, SEPOLIA_RPC_URL, ORACLE_PRIVATE_KEY, ETHERSCAN_API_KEY]):
    raise EnvironmentError("Missing env variables (CONTRACT, RPC, ORACLE_KEY, or ETHERSCAN_KEY)")

w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
if not w3.is_connected():
    raise ConnectionError("Failed to connect to Sepolia RPC")

oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)
w3.eth.default_account = oracle_account.address
logger.info(f"✅ Oracle Signer: {oracle_account.address}")

# Load Contract
artifact_path = '../blockchain/artifacts/contracts/TrustScore.sol/FraudDetection.json'
try:
    with open(artifact_path, 'r') as f:
        cj = json.load(f)
        contract_abi = cj.get('abi', [])
    fraud_detection_contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=contract_abi)
except Exception as e:
    logger.error(f"Failed to load contract ABI: {e}")
    raise

# Helpers
score_updater = WalletScoreUpdater()
feature_calc = RealTimeFeatureCalculator(
    rpc_url=SEPOLIA_RPC_URL,
    etherscan_api_key=ETHERSCAN_API_KEY,
    network='sepolia'
)

processed_tx_hashes = set()
wallet_score_cache = {} 

# --- CORE LOGIC ---

def process_transaction(tx_hash):
    if tx_hash in processed_tx_hashes:
        return
    
    logger.info(f"Processing: {tx_hash}")

    # 1. Feature Calc
    time.sleep(0.25) 
    features = feature_calc.get_features_for_tx(tx_hash)
    
    if not features:
        logger.warning(f"Features missing for {tx_hash}. Skipping.")
        return

    logger.info(f"Calculated Features:\n{json.dumps(features, indent=2)}")

    try:
        tx_data = feature_calc.w3.eth.get_transaction(tx_hash)
        from_addr = tx_data['from']
        to_addr = tx_data['to']
        value_eth = float(feature_calc.w3.from_wei(tx_data['value'], 'ether'))
    except Exception as e:
        logger.error(f"Error reading tx data: {e}")
        return

    # 2. ML Prediction
    ml_response = call_ml_api(features)
    if not ml_response:
        return

    risk_prob = ml_response.get('risk_probability', 0.0)
    risk_score_int = ml_response.get('risk_score', 1)
    
    # 3. Trust Score
    current_score = get_onchain_wallet_score(from_addr)
    tx_value_usd = features['Transaction_Value'] * features['Exchange_Rate']
    new_score = score_updater.calculate_new_score(current_score, risk_prob, tx_value_usd)
    
    logger.info(f"Risk: {risk_score_int}/10 ({risk_prob:.4f}) | Trust: {current_score:.2f} -> {new_score:.2f}")

    # 4. Write to Blockchain
    logger.info("Writing results to Blockchain...")
    
    # Updated Function Call (Only 3 arguments)
    tx_update_hash = send_update_tx(tx_hash, value_eth, risk_prob)
    
    wallet_update_hash = send_update_wallet_score(from_addr, new_score)
    
    if wallet_update_hash:
        wallet_score_cache[from_addr] = int(new_score * 100)

    # 5. Sync DB
    payload = {
        "transaction_hash": tx_hash,
        "from_address": from_addr,
        "to_address": to_addr,
        **features,
        "timestamp": int(time.time()),
        "risk_probability": risk_prob,
        "wallet_trust_before": current_score,
        "wallet_trust_score": new_score,
        "onchain_tx_update": tx_update_hash,
        "onchain_wallet_update": wallet_update_hash
    }
    post_enriched_tx_to_backend(payload)
    
    processed_tx_hashes.add(tx_hash)

# --- HELPER FUNCTIONS ---

def call_ml_api(features):
    try:
        payload = {"features": features}
        r = requests.post(API_PREDICT_URL, json=payload, timeout=5)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def get_onchain_wallet_score(address):
    if address in wallet_score_cache:
        return float(wallet_score_cache[address]) / 100.0
    try:
        score_int = fraud_detection_contract.functions.getWalletTrustScore(Web3.to_checksum_address(address)).call()
        return float(score_int) / 100.0
    except:
        return score_updater.get_initial_score()

# --- FIXED FUNCTION ---
def send_update_tx(tx_hash, value_eth, risk):
    """
    Sends updated Transaction Risk to Smart Contract.
    FIXED: Uses the new 3-argument signature.
    """
    try:
        tx_hash_bytes = Web3.keccak(text=tx_hash)[:32]
        nonce = w3.eth.get_transaction_count(oracle_account.address)
        
        # New Signature: updateTransactionRisk(bytes32, uint256, uint8)
        func = fraud_detection_contract.functions.updateTransactionRisk(
            tx_hash_bytes,
            int(value_eth * 1e18),  # Value in Wei
            int(risk * 100)         # Risk Score (0-100)
        )
        
        tx = func.build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 200000,
            'gasPrice': w3.eth.gas_price
        })
        
        signed = w3.eth.account.sign_transaction(tx, ORACLE_PRIVATE_KEY)
        tx_hash_sent = w3.eth.send_raw_transaction(signed.raw_transaction)
        
        logger.info(f"🔗 TX Risk Written: {tx_hash_sent.hex()}")
        w3.eth.wait_for_transaction_receipt(tx_hash_sent)
        return tx_hash_sent.hex()
        
    except ValueError as e:
        if "insufficient funds" in str(e):
            logger.critical("🚨 INSUFFICIENT FUNDS! Fund your Oracle: " + oracle_account.address)
        else:
            logger.error(f"Blockchain Write Error (Tx): {e}")
        return None
    except Exception as e:
        logger.error(f"Write Error (Tx): {e}")
        return None

def send_update_wallet_score(address, score):
    try:
        nonce = w3.eth.get_transaction_count(oracle_account.address)
        func = fraud_detection_contract.functions.updateWalletScore(
            Web3.to_checksum_address(address),
            int(score * 100)
        )
        tx = func.build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': 150000,
            'gasPrice': w3.eth.gas_price
        })
        signed = w3.eth.account.sign_transaction(tx, ORACLE_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info(f"🔗 Wallet Score Written: {tx_hash.hex()}")
        w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex()
    except Exception as e:
        logger.error(f"Write Error (Wallet): {e}")
        return None

def post_enriched_tx_to_backend(payload):
    try: requests.post(f"{API_URL.rstrip('/')}/transactions", json=payload, timeout=2)
    except: pass

def main():
    logger.info("Oracle Started. Processing LIVE Sepolia Batches...")
    while True:
        try:
            block = w3.eth.get_block('latest', full_transactions=True)
            logger.info(f"📦 Fetching Batch from Block {block.number}")
            batch = block.transactions[:BATCH_SIZE]
            
            for tx in batch:
                process_transaction(tx['hash'].hex())
            
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()