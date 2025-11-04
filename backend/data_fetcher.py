# backend/data_fetcher.py
from web3 import Web3
import pandas as pd
import time
from datetime import datetime
import json
import logging
from config import Config

class EthereumDataFetcher:
    def __init__(self, provider_url=None):
        # Use provided URL or get from config
        if provider_url is None:
            provider_url = Config.get_rpc_url()
            
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.logger = logging.getLogger(__name__)
        
        if not self.w3.is_connected():
            raise Exception(f"Failed to connect to Ethereum network: {provider_url}")
        
        self.logger.info(f"✅ Connected to Ethereum network: {provider_url}")
    
    def get_transaction_details(self, tx_hash):
        """Get detailed transaction information"""
        try:
            # Handle both string and HexBytes tx_hash
            if hasattr(tx_hash, 'hex'):
                tx_hash_hex = tx_hash.hex()
            else:
                tx_hash_hex = tx_hash
            
            tx = self.w3.eth.get_transaction(tx_hash_hex)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash_hex)
            
            # Get block information for timestamp
            block = self.w3.eth.get_block(tx['blockNumber'])
            
            # Calculate transaction fees
            gas_price = tx['gasPrice']
            gas_used = receipt['gasUsed']
            transaction_fee = self.w3.from_wei(gas_price * gas_used, 'ether')
            
            # Get wallet age (approximate)
            from_address = tx['from']
            wallet_age_days = self.estimate_wallet_age(from_address, block['timestamp'])
            
            # Get wallet balance
            wallet_balance = self.w3.from_wei(self.w3.eth.get_balance(from_address), 'ether')
            
            # Estimate transaction velocity (simplified)
            transaction_velocity = self.estimate_transaction_velocity(from_address)
            
            # Get current ETH price (simplified - you might want to use a real API)
            exchange_rate = self.get_ethereum_price()
            
            return {
                'Transaction_Value': float(self.w3.from_wei(tx['value'], 'ether')),
                'Transaction_Fees': float(transaction_fee),
                'Number_of_Inputs': 1,  # Simplified - can be improved
                'Number_of_Outputs': len(receipt['logs']) if receipt['logs'] else 1,  # Approximate
                'Gas_Price': float(self.w3.from_wei(gas_price, 'gwei')),
                'Wallet_Age_Days': wallet_age_days,
                'Wallet_Balance': float(wallet_balance),
                'Transaction_Velocity': transaction_velocity,
                'Exchange_Rate': exchange_rate,
                'timestamp': block['timestamp'],
                'from_address': from_address,
                'to_address': tx['to']
            }
        except Exception as e:
            self.logger.error(f"Error fetching transaction {tx_hash}: {e}")
            return None
    
    def estimate_wallet_age(self, address, current_timestamp):
        """Estimate wallet age in days"""
        try:
            # Get the earliest transaction for this wallet (more accurate approach)
            nonce = self.w3.eth.get_transaction_count(address)
            if nonce > 0:
                # Estimate based on average transactions per day
                # This is still an approximation - for better accuracy, you'd need to scan all transactions
                return min(nonce * 30, 365 * 10)  # Cap at 10 years
            return 1  # New wallet
        except Exception as e:
            self.logger.warning(f"Error estimating wallet age for {address}: {e}")
            return 1
    
    def estimate_transaction_velocity(self, address):
        """Estimate transaction velocity"""
        try:
            nonce = self.w3.eth.get_transaction_count(address)
            # Simple estimation based on nonce and wallet age
            wallet_age_days = self.estimate_wallet_age(address, time.time())
            return nonce / max(wallet_age_days, 1)
        except Exception as e:
            self.logger.warning(f"Error estimating transaction velocity for {address}: {e}")
            return 0
    
    def get_ethereum_price(self):
        """Get current Ethereum price (simplified)"""
        try:
            # You can integrate with a real price API here
            # For now, returning a default value
            # Consider integrating with CoinGecko, CoinMarketCap, or similar APIs
            return 2000.0  # Default ETH price in USD
        except:
            return 2000.0
    
    def get_latest_transactions(self, num_transactions=100):
        """Get latest transactions from the network"""
        try:
            latest_block = self.w3.eth.get_block('latest')
            transactions = []
            
            block_number = latest_block['number']
            blocks_checked = 0
            max_blocks_to_check = 50  # Limit to prevent long running operations
            
            self.logger.info(f"Fetching {num_transactions} latest transactions...")
            
            while len(transactions) < num_transactions and block_number > 0 and blocks_checked < max_blocks_to_check:
                try:
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    self.logger.debug(f"Processing block {block_number} with {len(block['transactions'])} transactions")
                    
                    for tx in block['transactions']:
                        tx_details = self.get_transaction_details(tx['hash'])
                        if tx_details:
                            transactions.append(tx_details)
                            if len(transactions) >= num_transactions:
                                break
                    
                    block_number -= 1
                    blocks_checked += 1
                    
                    # Small delay to prevent rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing block {block_number}: {e}")
                    block_number -= 1
                    blocks_checked += 1
                    continue
                    
            self.logger.info(f"Successfully fetched {len(transactions)} transactions")
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error fetching latest transactions: {e}")
            return []
    
    def get_wallet_transaction_history(self, address, limit=100):
        """Get transaction history for a specific wallet"""
        try:
            # This is a simplified approach - for production, you'd want to use a more comprehensive method
            # like scanning all blocks or using a specialized API
            
            transactions = []
            nonce = self.w3.eth.get_transaction_count(address)
            
            # Get recent transactions by scanning recent blocks
            latest_block = self.w3.eth.get_block('latest')
            block_number = latest_block['number']
            blocks_checked = 0
            max_blocks_to_check = 1000
            
            while len(transactions) < limit and block_number > 0 and blocks_checked < max_blocks_to_check:
                try:
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    
                    for tx in block['transactions']:
                        if tx['from'] == address or tx['to'] == address:
                            tx_details = self.get_transaction_details(tx['hash'])
                            if tx_details:
                                transactions.append(tx_details)
                                if len(transactions) >= limit:
                                    break
                    
                    block_number -= 1
                    blocks_checked += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing block {block_number} for wallet {address}: {e}")
                    block_number -= 1
                    blocks_checked += 1
                    continue
            
            return transactions
            
        except Exception as e:
            self.logger.error(f"Error fetching wallet transaction history for {address}: {e}")
            return []

# Example usage (for testing)
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        fetcher = EthereumDataFetcher()
        latest_transactions = fetcher.get_latest_transactions(10)
        print(f"Fetched {len(latest_transactions)} transactions")
        
        if latest_transactions:
            print("Sample transaction:")
            print(json.dumps(latest_transactions[0], indent=2, default=str))
            
    except Exception as e:
        print(f"Error: {e}")
