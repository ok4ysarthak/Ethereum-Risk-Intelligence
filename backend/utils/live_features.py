import requests
import time
from typing import Dict, Tuple

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