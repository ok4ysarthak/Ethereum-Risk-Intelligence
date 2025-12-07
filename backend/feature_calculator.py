import time
import requests
import logging
from web3 import Web3
from typing import Dict, Any, Tuple

# Chainlink Price Feed ABI
CHAINLINK_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"internalType": "uint80", "name": "roundId", "type": "uint80"},
            {"internalType": "int256", "name": "answer", "type": "int256"},
            {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
            {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
            {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealTimeFeatures")

class RealTimeFeatureCalculator:
    def __init__(self, rpc_url=None, etherscan_api_key=None, network='sepolia'):
        """
        network: 'sepolia' or 'mainnet'
        """
        self.network = network.lower()
        
        # 1. Configure Chain ID & RPC
        if self.network == 'mainnet':
            self.chain_id = 1  # Mainnet Chain ID
            default_rpc = "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
        else:
            self.chain_id = 11155111  # Sepolia Chain ID
            default_rpc = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"

        if rpc_url is None:
            rpc_url = default_rpc

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.etherscan_api_key = etherscan_api_key
        
        # 2. FIX: Use Unified V2 Endpoint for ALL networks
        self.etherscan_base = "https://api.etherscan.io/v2/api"
        
        if not self.w3.is_connected():
            raise Exception(f"Failed to connect to Ethereum network: {rpc_url}")
            

    def get_eth_price(self):
        try:
            if self.network == 'mainnet':
                addr = "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419"
            else:
                addr = "0x694AA1769357215DE4FAC081bf1f309aDC325306"
                
            price_feed_address = Web3.to_checksum_address(addr)
            contract = self.w3.eth.contract(address=price_feed_address, abi=CHAINLINK_ABI)
            round_id, answer, start, update, answered_in = contract.functions.latestRoundData().call()
            return float(answer) / 1e8 
        except Exception as e:
            logger.warning(f"Price feed failed: {e}. Defaulting to 3000.")
            return 3000.0

    def _fetch_etherscan_history(self, address: str):
        if not self.etherscan_api_key:
            logger.error("Missing Etherscan API Key")
            return []
            
        params = {
            "chainid": self.chain_id,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "asc",
            "apikey": self.etherscan_api_key
        }
        try:
            # logger.info(f"🔍 Querying Etherscan V2 for {address}...")
            resp = requests.get(self.etherscan_base, params=params, timeout=10).json()
            
            if resp['status'] == '0' and resp['message'] == 'No transactions found':
                logger.warning(f"Etherscan: No transactions found (New Wallet).")
                return []

            if resp['status'] == '0':
                logger.error(f"Etherscan API Error: {resp['result']}")
                return []
                
            if resp['status'] == '1':
                # logger.info(f"Etherscan found {len(resp['result'])} transactions.")
                return resp['result']
            
            return []
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            return []

    def calculate_volatility_and_age(self, address: str, current_balance_eth: float, tx_timestamp: int):
        history = self._fetch_etherscan_history(address)
        
        # FIX: Return 6 values (volatility, age, velocity, b_max, b_min, count)
        if not history:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0

        try:
            first_tx_ts = int(history[0]['timeStamp'])
            age_seconds = tx_timestamp - first_tx_ts
            age_days = max(age_seconds / 86400, 0.001) 
        except Exception as e:
            age_days = 0.001

        # Calculate Velocity
        tx_count = len(history)
        velocity = tx_count / age_days

        # Calculate Volatility & BMax/BMin
        history.sort(key=lambda x: int(x['timeStamp']), reverse=True)
        running_balance = current_balance_eth
        balances = [running_balance]
        address_lower = address.lower()
        
        for tx in history:
            val = float(tx['value']) / 1e18
            if tx['to'].lower() == address_lower:
                running_balance -= val
            elif tx['from'].lower() == address_lower:
                gas_cost = (float(tx.get('gasPrice', 0)) * float(tx.get('gasUsed', 0))) / 1e18
                running_balance += (val + gas_cost)
            
            if running_balance < 0: running_balance = 0
            balances.append(running_balance)

        b_max = max(balances)
        b_min = min(balances)
        
        volatility = (b_max - b_min) / max(tx_count, 1)

        # Return extended stats
        return volatility, age_days, velocity, b_max, b_min, tx_count

    def get_features_for_tx(self, tx_hash: str) -> Dict[str, float]:
        try:
            tx = self.w3.eth.get_transaction(tx_hash)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            block = self.w3.eth.get_block(tx['blockNumber'])
            tx_timestamp = block['timestamp']
        except Exception as e:
            logger.error(f"RPC fetch failed: {e}")
            return None

        eth_value = float(self.w3.from_wei(tx['value'], 'ether'))
        gas_price_gwei = float(self.w3.from_wei(tx['gasPrice'], 'gwei'))
        gas_used = receipt['gasUsed']
        tx_fee_eth = float(self.w3.from_wei(tx['gasPrice'] * gas_used, 'ether'))
        
        num_inputs = 1.0
        num_outputs = float(1 + len(receipt.get('logs', [])))

        sender = tx['from']
        current_balance_wei = self.w3.eth.get_balance(sender)
        current_balance_eth = float(self.w3.from_wei(current_balance_wei, 'ether'))
        
        # logger.info(f"🚀 Calculating features for sender: {sender}")
        
        # Unpack all 6 values
        volatility, age_days, velocity, b_max, b_min, nt = self.calculate_volatility_and_age(sender, current_balance_eth, tx_timestamp)
        
        exchange_rate = self.get_eth_price()

        return {
            # Standard ML Features
            "Transaction_Value": eth_value,
            "Transaction_Fees": tx_fee_eth,
            "Number_of_Inputs": num_inputs,
            "Number_of_Outputs": num_outputs,
            "Gas_Price": gas_price_gwei,
            "Wallet_Age_Days": age_days,
            "Wallet_Balance": current_balance_eth,
            "Transaction_Velocity": velocity,
            "Exchange_Rate": exchange_rate,
            "Final_Balance": current_balance_eth,
            "BMax_BMin_per_NT": volatility,
            
            # DEBUGGING METRICS (Extra)
            "B_max": b_max,
            "B_min": b_min,
            "NT": nt
        }