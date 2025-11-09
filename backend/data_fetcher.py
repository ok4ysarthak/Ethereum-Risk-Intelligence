# backend/data_fetcher.py
from web3 import Web3
import pandas as pd
import time
from datetime import datetime
import json
import logging
from config import Config
import requests
import copy

class EthereumDataFetcher:
    def __init__(self, provider_url=None):
        # Use provided URL or get from config
        if provider_url is None:
            provider_url = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"

        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.logger = logging.getLogger(__name__)
        self.wallet_age_cache = {}
        self.velocity_cache = {}

        if not self.w3.is_connected():
            raise Exception(f"Failed to connect to Ethereum network: {provider_url}")

        self.logger.info(f"✅ Connected to Ethereum network: {provider_url}")

    # ----------------- Helper utilities -----------------
    def _parse_alchemy_tx_timestamp(self, tx):
        """
        Given an Alchemy transfer item (may contain metadata, blockNum, etc),
        return a unix timestamp (int seconds) or None if unavailable.
        Handles:
          - metadata.blockTimestamp (iso string or hex)
          - blockNum (hex) -> fetch block timestamp from chain
        """
        try:
            # 1) metadata.blockTimestamp if present
            metadata = tx.get("metadata")
            if metadata and metadata.get("blockTimestamp"):
                ts = metadata["blockTimestamp"]
                # If it's hex like "0x..." -> parse as hex int
                if isinstance(ts, str) and ts.startswith("0x"):
                    try:
                        return int(ts, 16)
                    except Exception:
                        pass
                # If it's ISO datetime string (Alchemy sometimes returns ISO)
                try:
                    # remove timezone Z if present
                    iso = ts.rstrip("Z")
                    dt = datetime.fromisoformat(iso)
                    return int(dt.timestamp())
                except Exception:
                    pass

            # 2) if no metadata timestamp, use blockNum (hex) and fetch block
            block_num_hex = tx.get("blockNum") or tx.get("blockNumber")
            if block_num_hex:
                # block_num_hex may be like "0x827e8b" or decimal string
                if isinstance(block_num_hex, str) and block_num_hex.startswith("0x"):
                    block_number = int(block_num_hex, 16)
                else:
                    block_number = int(block_num_hex)
                try:
                    block = self.w3.eth.get_block(block_number)
                    return int(block["timestamp"])
                except Exception as e:
                    self.logger.debug(f"Could not fetch block {block_number} for timestamp: {e}")
                    return None

        except Exception as e:
            self.logger.debug(f"Error parsing timestamp from Alchemy tx object: {e}")
            return None

        return None

    # ----------------- Main fetchers -----------------
    def get_transaction_details(self, tx_hash):
        """Get detailed transaction information"""
        try:
            # normalize hash to hex string with 0x prefix
            if hasattr(tx_hash, "hex"):
                tx_hash_hex = tx_hash.hex()
            else:
                # if user passed a hex without 0x ensure it has 0x
                tx_hash_hex = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash

            tx = self.w3.eth.get_transaction(tx_hash_hex)
            receipt = self.w3.eth.get_transaction_receipt(tx_hash_hex)
            block = self.w3.eth.get_block(tx["blockNumber"])

            gas_price = tx.get("gasPrice", 0)
            gas_used = receipt.get("gasUsed", 0)
            transaction_fee = self.w3.from_wei(gas_price * gas_used, "ether")

            from_address = tx.get("from")
            # wallet age: provide block timestamp (int) to estimate_wallet_age
            wallet_age_days = self.estimate_wallet_age(from_address, block["timestamp"])
            wallet_balance = self.w3.from_wei(self.w3.eth.get_balance(from_address), "ether")

            velocity_features = self.estimate_transaction_velocity(from_address)
            exchange_rate = self.get_ethereum_price()

            tx_data = {
                "Transaction_Value": float(self.w3.from_wei(tx.get("value", 0), "ether")),
                "Transaction_Fees": float(transaction_fee),
                "Number_of_Inputs": 1,
                "Number_of_Outputs": len(receipt.get("logs", [])) if receipt.get("logs") else 1,
                "Gas_Price": float(self.w3.from_wei(gas_price, "gwei")) if gas_price else 0.0,
                "Wallet_Age_Days": wallet_age_days,
                "Wallet_Balance": float(wallet_balance),
                "Exchange_Rate": exchange_rate,
                "timestamp": int(block["timestamp"]),
                "from_address": from_address,
                "to_address": tx.get("to"),
            }

            tx_data.update(velocity_features)
            tx_data["transaction_hash"] = Web3.to_hex(tx.hash)

            return tx_data

        except Exception as e:
            self.logger.error(f"Error fetching transaction {tx_hash}: {e}", exc_info=True)
            return None

    def get_contract_transactions(self, contract_address=None, from_block=None, to_block="latest", limit=50):
        """
        Fast: try eth_getLogs to find transactions touching the contract address (events).
        Fallback: scan blocks and inspect transactions if eth_getLogs fails.
        """
        if contract_address is None:
            self.logger.warning("get_contract_transactions called without a contract_address.")
            return []

        try:
            latest = self.w3.eth.block_number
            if to_block == "latest":
                to_block = latest
            if from_block is None:
                lookback_blocks = min(100, latest)
                from_block = max(0, latest - lookback_blocks)

            self.logger.info(f"DEBUG [get_contract_transactions]: Calling eth_getLogs, from={from_block}, to={to_block}")

            try:
                logs = self.w3.eth.get_logs(
                    {"fromBlock": from_block, "toBlock": to_block, "address": Web3.to_checksum_address(contract_address)}
                )

                # collect unique tx hashes from logs
                tx_hashes = []
                seen = set()
                for ev in reversed(logs):
                    th = ev["transactionHash"]
                    if hasattr(th, "hex"):
                        th = th.hex()
                    if th not in seen:
                        seen.add(th)
                        tx_hashes.append(th)
                    if len(tx_hashes) >= limit:
                        break

            except Exception as e:
                # fallback when eth_getLogs fails (400 or other)
                self.logger.warning(f"eth_getLogs failed or returned error: {e}. Falling back to scanning blocks.")
                tx_hashes = []
                seen = set()
                # scan blocks backwards; keep it limited
                for bn in range(to_block, max(from_block, to_block - 200) - 1, -1):
                    try:
                        block = self.w3.eth.get_block(bn, full_transactions=True)
                    except Exception as be:
                        self.logger.debug(f"Could not fetch block {bn}: {be}")
                        continue
                    for tx in block["transactions"]:
                        # tx.to may be None for contract creation; also checksum compare
                        to_addr = tx.to
                        if to_addr and Web3.to_checksum_address(to_addr) == Web3.to_checksum_address(contract_address):
                            th = Web3.to_hex(tx.hash)
                            if th not in seen:
                                seen.add(th)
                                tx_hashes.append(th)
                                if len(tx_hashes) >= limit:
                                    break
                    if len(tx_hashes) >= limit:
                        break

            self.logger.info(f"Collected {len(tx_hashes)} tx hashes to inspect")
            transactions = []
            for th in tx_hashes:
                details = self.get_transaction_details(th)
                if details:
                    transactions.append(details)
                    if len(transactions) >= limit:
                        break

            return transactions

        except Exception as e:
            self.logger.error(f"Error get_contract_transactions: {e}", exc_info=True)
            return []

    def _find_first_tx_block(self, address, max_lookback_blocks=500_000, coarse_step=1000):
    
        try:
            latest = self.w3.eth.block_number
            start_block = max(0, latest - max_lookback_blocks)
            # coarse scan backwards
            found_block = None
            scan_top = latest
            scan_bottom = max(start_block, latest - coarse_step)
            while scan_top >= start_block:
                try:
                    # fetch block with full txs at scan_top
                    block = self.w3.eth.get_block(scan_top, full_transactions=True)
                except Exception as e:
                    # if provider fails at an isolated block, step back
                    self.logger.debug(f"Block fetch failed at {scan_top}: {e}")
                    scan_top -= coarse_step
                    scan_bottom = max(start_block, scan_top - coarse_step)
                    continue

                # check if any tx in this block touches the address
                addr_checksum = Web3.to_checksum_address(address)
                hit = False
                for tx in block.get("transactions", []):
                    if tx and ((tx.to and Web3.to_checksum_address(tx.to) == addr_checksum) or (tx.get("from") and Web3.to_checksum_address(tx.get("from")) == addr_checksum)):
                        hit = True
                        break
                if hit:
                    found_block = scan_top
                    break

                # move window backwards
                scan_top = scan_bottom - 1
                scan_bottom = max(start_block, scan_top - coarse_step)

            if found_block is None:
                # nothing found in coarse scan
                return (None, None)

            # binary search between scan_bottom..found_block to find earliest block with a hit
            low = max(start_block, found_block - coarse_step)
            high = found_block
            first_hit = None
            while low <= high:
                mid = (low + high) // 2
                try:
                    block = self.w3.eth.get_block(mid, full_transactions=True)
                except Exception as e:
                    self.logger.debug(f"Block fetch failed during binary at {mid}: {e}")
                    # nudge boundaries safely
                    low = mid + 1
                    continue

                addr_checksum = Web3.to_checksum_address(address)
                hit = False
                for tx in block.get("transactions", []):
                    if tx and ((tx.to and Web3.to_checksum_address(tx.to) == addr_checksum) or (tx.get("from") and Web3.to_checksum_address(tx.get("from")) == addr_checksum)):
                        hit = True
                        break

                if hit:
                    first_hit = mid
                    high = mid - 1
                else:
                    low = mid + 1

            if first_hit is None:
                return (None, None)

            first_block = self.w3.eth.get_block(first_hit)
            return (first_hit, int(first_block["timestamp"]))

        except Exception as e:
            self.logger.warning(f"_find_first_tx_block fallback failed: {e}", exc_info=True)
            return (None, None)


    def estimate_wallet_age(self, address, current_timestamp):
        """
        Estimate wallet age in days by fetching its first transaction from Alchemy.
        Uses a cache to avoid repeated API calls for the same address.
        `current_timestamp` should be an int unix timestamp (seconds).
        """
        if address in self.wallet_age_cache:
            return self.wallet_age_cache[address]

        # Ensure timestamp is int seconds
        try:
            current_ts = int(current_timestamp)
        except Exception:
            current_ts = int(time.time())

        # Use the dynamic URL from the class constructor
        alchemy_url = self.w3.provider.endpoint_uri

        payload_template = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "category": ["external", "internal", "erc20", "erc721", "erc1155"],
                    "order": "asc",
                    # return only the first transfer (hex)
                    "maxCount": "0x1",
                }
            ],
        }

        payload_from = copy.deepcopy(payload_template)
        payload_from["params"][0]["fromAddress"] = address

        payload_to = copy.deepcopy(payload_template)
        payload_to["params"][0]["toAddress"] = address

        try:
            self.logger.info(f"DEBUG [estimate_wallet_age]: Sending 'from' payload: {json.dumps(payload_from)}")
            response_from = requests.post(alchemy_url, json=payload_from, timeout=10)
            response_from.raise_for_status()
            data_from = response_from.json()
            self.logger.debug(f"DEBUG [estimate_wallet_age]: Got 'from' response: {json.dumps(data_from)}")

            self.logger.info(f"DEBUG [estimate_wallet_age]: Sending 'to' payload: {json.dumps(payload_to)}")
            response_to = requests.post(alchemy_url, json=payload_to, timeout=10)
            response_to.raise_for_status()
            data_to = response_to.json()
            self.logger.debug(f"DEBUG [estimate_wallet_age]: Got 'to' response: {json.dumps(data_to)}")

            from_txs = data_from.get("result", {}).get("transfers", []) or []
            to_txs = data_to.get("result", {}).get("transfers", []) or []

            all_txs = from_txs + to_txs
            timestamps = []
            for tx in all_txs:
                ts = self._parse_alchemy_tx_timestamp(tx)
                if ts:
                    timestamps.append(ts)

            if timestamps:
                first_tx_timestamp = min(timestamps)
                age_in_seconds = current_ts - int(first_tx_timestamp)
                age_in_days = max(1, int(age_in_seconds / (60 * 60 * 24)))
                self.wallet_age_cache[address] = age_in_days
                return age_in_days

            # --- Fallback A: try get_wallet_transaction_history (maybe different paging) ---
            try:
                hist = self.get_wallet_transaction_history(address, limit=50)
                if hist:
                    # hist entries include 'timestamp' (block timestamp)
                    first_ts = min(int(h["timestamp"]) for h in hist if h.get("timestamp"))
                    age_in_seconds = current_ts - int(first_ts)
                    age_in_days = max(1, int(age_in_seconds / (60 * 60 * 24)))
                    self.wallet_age_cache[address] = age_in_days
                    return age_in_days
            except Exception as e:
                self.logger.debug(f"get_wallet_transaction_history fallback failed: {e}")

            # --- Fallback B: do an on-chain block scan to find earliest tx (reliable but slow) ---
            self.logger.info(f"No transfers from Alchemy for {address}. Falling back to block-scan (slower).")
            first_block, first_block_ts = self._find_first_tx_block(address)
            if first_block is not None and first_block_ts is not None:
                age_in_seconds = current_ts - int(first_block_ts)
                age_in_days = max(1, int(age_in_seconds / (60 * 60 * 24)))
                # cache the value (and optionally cache the first_block as well)
                self.wallet_age_cache[address] = age_in_days
                # optionally cache the block number separately if you want
                return age_in_days

            # final fallback: try using nonce to produce a coarse estimate (avoid division by 1 producing huge rates)
            try:
                nonce = self.w3.eth.get_transaction_count(address)
                # assume a conservative activity rate (e.g., 1 tx per day) to avoid extreme tx_per_day values
                assumed_days = max(7, int(nonce / 1) if nonce > 0 else 7)
                age_in_days = max(1, assumed_days)
                self.wallet_age_cache[address] = age_in_days
                return age_in_days
            except Exception as e:
                self.logger.warning(f"Could not use nonce fallback for {address}: {e}")

            # ultimate fallback
            self.wallet_age_cache[address] = 1
            return 1

        except Exception as e:
            self.logger.error(f"Error fetching wallet age from Alchemy for {address}: {e}. Defaulting to 1 day.", exc_info=True)
            self.wallet_age_cache[address] = 1
            return 1

    def estimate_transaction_velocity(self, address):
        """
        Estimate transaction velocity features using Alchemy.
        Uses a 60-second cache to avoid rate-limiting.
        """
        current_time = time.time()
        if address in self.velocity_cache:
            cache_time, cached_data = self.velocity_cache[address]
            if (current_time - cache_time) < 60:
                return cached_data

        try:
            try:
                nonce = self.w3.eth.get_transaction_count(address)
                # ensure we pass a reasonable timestamp (int seconds) to estimate_wallet_age
                wallet_age_days = self.estimate_wallet_age(address, int(current_time))
                lifetime_avg_tx_per_day = nonce / max(wallet_age_days, 1)
            except Exception as e:
                self.logger.warning(f"Error getting base velocity data for {address}: {e}")
                lifetime_avg_tx_per_day = 0

            alchemy_url = self.w3.provider.endpoint_uri

            payload_template = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getAssetTransfers",
                "params": [
                    {
                        "category": ["external", "internal", "erc20", "erc721", "erc1155"],
                        "order": "desc",
                        # request up to 100 records (hex)
                        "maxCount": "0x64",
                    }
                ],
            }

            payload_from = json.loads(json.dumps(payload_template))
            payload_from["params"][0]["fromAddress"] = address

            payload_to = json.loads(json.dumps(payload_template))
            payload_to["params"][0]["toAddress"] = address

            tx_count_24h = 0
            tx_count_7d = 0
            tx_count_30d = 0

            try:
                response_from = requests.post(alchemy_url, json=payload_from, timeout=10)
                response_from.raise_for_status()
                data_from = response_from.json()
                from_txs = data_from.get("result", {}).get("transfers", []) or []

                response_to = requests.post(alchemy_url, json=payload_to, timeout=10)
                response_to.raise_for_status()
                data_to = response_to.json()
                to_txs = data_to.get("result", {}).get("transfers", []) or []

                # dedupe by hash
                all_txs_map = {}
                for tx in from_txs + to_txs:
                    if "hash" in tx:
                        all_txs_map[tx["hash"]] = tx

                if all_txs_map:
                    for tx in all_txs_map.values():
                        ts = self._parse_alchemy_tx_timestamp(tx)
                        if ts is None:
                            continue
                        age_seconds = current_time - ts
                        if age_seconds <= (24 * 60 * 60):
                            tx_count_24h += 1
                        if age_seconds <= (7 * 24 * 60 * 60):
                            tx_count_7d += 1
                        if age_seconds <= (30 * 24 * 60 * 60):
                            tx_count_30d += 1

            except Exception as e:
                self.logger.warning(f"Could not fetch recent tx for velocity from Alchemy: {e}", exc_info=True)

            velocity_data = {
                "tx_per_day_lifetime": lifetime_avg_tx_per_day,
                "tx_count_24h": tx_count_24h,
                "tx_count_7d": tx_count_7d,
                "tx_count_30d": tx_count_30d,
            }
            self.velocity_cache[address] = (current_time, velocity_data)

            return velocity_data

        except Exception as e:
            self.logger.error(f"Failed to estimate transaction velocity for {address}: {e}", exc_info=True)
            return {"tx_per_day_lifetime": 0, "tx_count_24h": 0, "tx_count_7d": 0, "tx_count_30d": 0}

    def get_ethereum_price(self):
        """Get current Ethereum price from CoinMarketCap API"""
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        params = {"symbol": "ETH", "convert": "USD"}
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": "2d8fb3cc874645cab0912b2399a7f654",
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            price = data["data"]["ETH"]["quote"]["USD"]["price"]
            return float(price)
        except Exception as e:
            self.logger.warning(f"Could not fetch ETH price from CoinMarketCap: {e}. Returning default 2000.0")
            return 2000.0

    def get_latest_transactions(self, num_transactions=100):
        """Get latest transactions from the network"""
        try:
            latest_block = self.w3.eth.get_block("latest")
            transactions = []
            block_number = latest_block["number"]
            blocks_checked = 0
            max_blocks_to_check = 50

            self.logger.info(f"Fetching {num_transactions} latest transactions...")

            while len(transactions) < num_transactions and block_number > 0 and blocks_checked < max_blocks_to_check:
                try:
                    block = self.w3.eth.get_block(block_number, full_transactions=True)
                    self.logger.debug(f"Processing block {block_number} with {len(block['transactions'])} transactions")

                    for tx in reversed(block["transactions"]):
                        tx_details = self.get_transaction_details(tx.hash)
                        if tx_details:
                            transactions.append(tx_details)
                            if len(transactions) >= num_transactions:
                                break
                    if len(transactions) >= num_transactions:
                        break
                    block_number -= 1
                    blocks_checked += 1
                    time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error processing block {block_number}: {e}", exc_info=True)
                    block_number -= 1
                    blocks_checked += 1
                    continue

            self.logger.info(f"Successfully fetched {len(transactions)} transactions")
            return transactions
        except Exception as e:
            self.logger.error(f"Error fetching latest transactions: {e}", exc_info=True)
            return []

    def get_wallet_transaction_history(self, address, limit=100):
        """
        Get transaction history for a specific wallet using Alchemy.
        """
        alchemy_url = self.w3.provider.endpoint_uri

        payload_template = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "category": ["external", "internal", "erc20", "erc721", "erc1155"],
                    "order": "desc",
                    # maxCount MUST be hex string; use hex(limit)
                    "maxCount": hex(limit),
                }
            ],
        }

        payload_from = json.loads(json.dumps(payload_template))
        payload_from["params"][0]["fromAddress"] = address

        payload_to = json.loads(json.dumps(payload_template))
        payload_to["params"][0]["toAddress"] = address

        try:
            response_from = requests.post(alchemy_url, json=payload_from, timeout=10)
            response_from.raise_for_status()
            data_from = response_from.json()
            from_txs = data_from.get("result", {}).get("transfers", []) or []

            response_to = requests.post(alchemy_url, json=payload_to, timeout=10)
            response_to.raise_for_status()
            data_to = response_to.json()
            to_txs = data_to.get("result", {}).get("transfers", []) or []

            all_txs = {}
            for tx in from_txs + to_txs:
                if "hash" in tx:
                    all_txs[tx["hash"]] = tx

            # sort by the best available timestamp: metadata.blockTimestamp or blockNum -> use helper
            sorted_txs = sorted(
                all_txs.values(),
                key=lambda x: self._parse_alchemy_tx_timestamp(x) or 0,
                reverse=True,
            )

            enriched_transactions = []
            for tx in sorted_txs[:limit]:
                tx_hash = tx.get("hash")
                if not tx_hash:
                    continue
                tx_hash_prefix = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
                tx_details = self.get_transaction_details(tx_hash_prefix)
                if tx_details:
                    enriched_transactions.append(tx_details)

            return enriched_transactions

        except Exception as e:
            self.logger.error(f"Error fetching wallet transaction history from Alchemy for {address}: {e}", exc_info=True)
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider_url = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"

    try:
        fetcher = EthereumDataFetcher(provider_url=provider_url)

        latest_transactions = fetcher.get_latest_transactions(10)

        print(f"Fetched {len(latest_transactions)} transactions")

        if latest_transactions:
            print("Sample transaction:")
            print(json.dumps(latest_transactions[0], indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}")
