# backend/oracle.py
import os
import json
import time
import logging
import requests
from web3 import Web3
from dotenv import load_dotenv
from typing import Any, Dict, Optional

# --- LOCAL IMPORTS ---
from utils.wallet_updater import WalletScoreUpdater
from utils.feature_calculator import RealTimeFeatureCalculator

import threading  # <--- NEW IMPORT
import concurrent.futures # Add this import at top
# Global State
current_nonce = -1
nonce_lock = threading.Lock()  # <--- NEW LOCK
processed_lock = threading.Lock()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# --- CONFIGURATION ---
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, '.env'), override=True)
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
logger.info(f"Oracle Signer: {oracle_account.address}")

# Load Contract
artifact_path = os.path.join(PROJECT_ROOT, 'blockchain', 'artifacts', 'contracts', 'TrustScore.sol', 'FraudDetection.json')
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

def process_transaction(tx_hash: str, defer_wallet_write: bool = False) -> Optional[Dict[str, Any]]:
    with processed_lock:
        if tx_hash in processed_tx_hashes:
            return None
    
    logger.info(f"Processing: {tx_hash}")
    
    features = feature_calc.get_features_for_tx(tx_hash)
    
    if not features:
        logger.warning(f"Features missing for {tx_hash}. Skipping.")
        return None

    logger.info(f"Calculated Features:\n{json.dumps(features, indent=2)}")

    try:
        tx_data = feature_calc.w3.eth.get_transaction(tx_hash)
        from_addr = tx_data['from']
        to_addr = tx_data['to']
        value_eth = float(feature_calc.w3.from_wei(tx_data['value'], 'ether'))
    except Exception as e:
        logger.error(f"Error reading tx data: {e}")
        return None

    # 2. ML Prediction + Temporal Intelligence
    tx_timestamp = int(time.time())
    ml_response = call_ml_api(features, wallet_address=from_addr, tx_timestamp=tx_timestamp)
    if not ml_response:
        return None

    risk_prob = ml_response.get('fraud_probability', ml_response.get('risk_probability', 0.0))
    risk_prob = max(0.0, min(1.0, float(risk_prob or 0.0)))
    risk_score_int = int(ml_response.get('risk_score', 1) or 1)

    # 3. Trust Score (primary source: temporal model response from backend)
    current_score = ml_response.get('temporal_previous_score')
    if current_score is None:
        current_score = get_onchain_wallet_score(from_addr)
    current_score = max(0.0, min(1.0, float(current_score or 0.0)))

    temporal_score = ml_response.get('temporal_score_normalized')
    if temporal_score is None:
        tx_value_usd = features['Transaction_Value'] * features['Exchange_Rate']
        temporal_score = score_updater.calculate_new_score(current_score, risk_prob, tx_value_usd)

    new_score = max(0.0, min(1.0, float(temporal_score)))
    
    logger.info(f"Risk: {risk_score_int}/10 ({risk_prob:.4f}) | Trust: {current_score:.2f} -> {new_score:.2f}")

    # 4. Write to Blockchain
    logger.info("Writing results to Blockchain...")
    
    # Updated Function Call (Only 3 arguments)
    tx_update_hash = send_update_tx(tx_hash, value_eth, risk_prob)

    wallet_update_hash = None
    if not defer_wallet_write:
        wallet_update_hash = send_update_wallet_score(from_addr, new_score)
        if wallet_update_hash:
            wallet_score_cache[from_addr] = int(new_score * 100)

    # 5. Sync DB
    payload = {
        "transaction_hash": tx_hash,
        "from_address": from_addr,
        "to_address": to_addr,
        **features,
        "timestamp": tx_timestamp,
        "risk_probability": risk_prob,
        "fraud_probability": risk_prob,
        "risk_score": risk_score_int,
        "wallet_trust_before": current_score,
        "wallet_trust_score": new_score,
        "temporal_score": ml_response.get("temporal_score", round(new_score * 10.0, 3)),
        "temporal_score_normalized": new_score,
        "temporal_state": ml_response.get("temporal_state"),
        "shap_explanation": ml_response.get("shap_explanation"),
        "rule_explanation": ml_response.get("explanation"),
        "onchain_tx_update": tx_update_hash,
        "onchain_wallet_update": wallet_update_hash
    }
    if defer_wallet_write:
        with processed_lock:
            processed_tx_hashes.add(tx_hash)
        return {
            "wallet_address": from_addr,
            "wallet_score": new_score,
            "payload": payload,
        }

    post_enriched_tx_to_backend(payload)

    with processed_lock:
        processed_tx_hashes.add(tx_hash)
    return {
        "wallet_address": from_addr,
        "wallet_score": new_score,
        "payload": payload,
    }

# --- HELPER FUNCTIONS ---

def call_ml_api(features, wallet_address=None, tx_timestamp=None):
    try:
        payload = {
            "features": features,
            "wallet_address": wallet_address,
            "timestamp": tx_timestamp,
            "update_state": bool(wallet_address),
        }
        r = requests.post(API_PREDICT_URL, json=payload, timeout=5)
        if r.status_code == 200:
            return r.json()
        logger.warning("Prediction API returned %s for %s", r.status_code, wallet_address)
    except Exception as e:
        logger.warning("Prediction API request failed: %s", e)
    return None

def get_onchain_wallet_score(address):
    if address in wallet_score_cache:
        return float(wallet_score_cache[address]) / 100.0
    try:
        score_int = fraud_detection_contract.functions.getWalletTrustScore(Web3.to_checksum_address(address)).call()
        return float(score_int) / 100.0
    except:
        return score_updater.get_initial_score()

def get_nonce():
    global current_nonce
    with nonce_lock:  # <--- LOCKS THIS BLOCK
        if current_nonce == -1:
            current_nonce = w3.eth.get_transaction_count(oracle_account.address, 'pending')
        else:
            current_nonce += 1
        return current_nonce

# --- FIXED FUNCTION ---
def send_update_tx(tx_hash, value_eth, risk):
    """
    Sends updated Transaction Risk to Smart Contract.
    FIXED: Uses the new 3-argument signature.
    """
    try:
        # tx_hash_bytes = Web3.keccak(text=tx_hash)[:32]
        # tx_hash_bytes = bytes.fromhex(tx_hash[2:])

        if tx_hash.startswith("0x"):
            clean_hex = tx_hash[2:]
        else:
            clean_hex = tx_hash
            
        tx_hash_bytes = bytes.fromhex(clean_hex)

        nonce = get_nonce()
                
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
        # w3.eth.wait_for_transaction_receipt(tx_hash_sent)
        return tx_hash_sent.hex()
        
    except ValueError as e:
        if "insufficient funds" in str(e):
            logger.critical("🚨 INSUFFICIENT FUNDS! Fund your Oracle: " + oracle_account.address)
        else:
            logger.error(f"Blockchain Write Error (Tx): {e}")
        return None
    except Exception as e:
        logger.error(f"Write Error (Tx): {e}")
        global current_nonce
        current_nonce = -1 
        return None

def send_update_wallet_score(address, score):
    try:
        nonce = get_nonce()
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
        # w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex()
    except Exception as e:
        logger.error(f"Write Error (Wallet): {e}")
        return None


def send_batch_wallet_scores(score_map: Dict[str, float]) -> Optional[str]:
    if not score_map:
        return None

    try:
        wallet_addrs = []
        wallet_scores = []
        for address, score in score_map.items():
            if not address:
                continue
            try:
                wallet_addrs.append(Web3.to_checksum_address(address))
                normalized = max(0.0, min(1.0, float(score or 0.0)))
                wallet_scores.append(int(normalized * 100))
            except Exception:
                continue

        if not wallet_addrs:
            return None

        nonce = get_nonce()
        func = fraud_detection_contract.functions.batchUpdateWallets(wallet_addrs, wallet_scores)
        est_gas = min(2_500_000, 120000 + (60000 * len(wallet_addrs)))
        tx = func.build_transaction({
            'from': oracle_account.address,
            'nonce': nonce,
            'gas': est_gas,
            'gasPrice': w3.eth.gas_price,
        })

        signed = w3.eth.account.sign_transaction(tx, ORACLE_PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("🔗 Wallet Scores Batch Written (%d wallets): %s", len(wallet_addrs), tx_hash.hex())
        return tx_hash.hex()

    except Exception as e:
        logger.error("Write Error (Wallet Batch): %s", e)
        global current_nonce
        current_nonce = -1
        return None

def post_enriched_tx_to_backend(payload):
    try: requests.post(f"{API_URL.rstrip('/')}/transactions", json=payload, timeout=2)
    except: pass

def main():
    logger.info("Oracle Started. Processing LIVE Sepolia Batches...")
    
    # Initialize nonce correctly once at startup
    global current_nonce
    # Subtract 1 so the first get_nonce() call increments it to the correct value
    current_nonce = w3.eth.get_transaction_count(oracle_account.address, 'pending') - 1

    while True:
        try:
            block = w3.eth.get_block('latest', full_transactions=True)
            batch = block.transactions[:BATCH_SIZE]
            
            if not batch:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
                
            logger.info(f"⚡ Processing Batch of {len(batch)} txs with SAFE PARALLELISM")

            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                tx_hashes = [tx['hash'].hex() for tx in batch]
                futures = [executor.submit(process_transaction, txh, True) for txh in tx_hashes]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        result = fut.result()
                        if result:
                            batch_results.append(result)
                    except Exception as worker_exc:
                        logger.error("Worker processing failed: %s", worker_exc)

            wallet_scores = {}
            for result in batch_results:
                addr = result.get("wallet_address")
                score = result.get("wallet_score")
                if addr and score is not None:
                    wallet_scores[addr] = float(score)

            wallet_tx_hash_map = {}
            if wallet_scores:
                batch_wallet_hash = send_batch_wallet_scores(wallet_scores)
                if batch_wallet_hash:
                    for addr, score in wallet_scores.items():
                        wallet_tx_hash_map[addr] = batch_wallet_hash
                        wallet_score_cache[addr] = int(max(0.0, min(1.0, float(score))) * 100)
                else:
                    # Fallback to per-wallet writes if the batch transaction fails.
                    for addr, score in wallet_scores.items():
                        single_hash = send_update_wallet_score(addr, score)
                        if single_hash:
                            wallet_tx_hash_map[addr] = single_hash
                            wallet_score_cache[addr] = int(max(0.0, min(1.0, float(score))) * 100)

            for result in batch_results:
                payload = result.get("payload")
                if not payload:
                    continue
                source_wallet = payload.get("from_address")
                payload["onchain_wallet_update"] = wallet_tx_hash_map.get(source_wallet)
                post_enriched_tx_to_backend(payload)
            
            time.sleep(POLL_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(5)
if __name__ == "__main__":
    main()