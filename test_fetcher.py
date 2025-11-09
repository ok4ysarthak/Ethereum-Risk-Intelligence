# test_fetcher.py
import logging
import json
import time
import sys

# Import your fetcher
from backend.data_fetcher import EthereumDataFetcher

PROVIDER_URL = "https://eth-sepolia.g.alchemy.com/v2/5bURjldvKPu4glB_tFxWt"

def pretty_print(title, data):
    print(title)
    try:
        print(json.dumps(data, indent=2, default=str))
    except Exception as e:
        print(f"Could not pretty-print: {e}\nData: {data}")
    print("-" * 40)

def main():
    logging.basicConfig(level=logging.INFO)
    print("--- 1. Init fetcher & get ETH price ---")
    try:
        fetcher = EthereumDataFetcher(provider_url=PROVIDER_URL)
        price = fetcher.get_ethereum_price()
        print(f"✅ Price: ${price}")
        print("-" * 40)
    except Exception as e:
        print(f"❌ FAILED: __init__ or get_ethereum_price: {e}")
        sys.exit(1)

    # --- 2. Get latest transactions and pick a sample transaction dynamically ---
    print("--- 2. Fetch latest transactions and pick a sample ---")
    latest_txs = fetcher.get_latest_transactions(5)
    if not latest_txs:
        print("❌ FAILED: get_latest_transactions returned no data.")
        sys.exit(1)

    print(f"✅ Fetched {len(latest_txs)} transactions.")
    sample_tx = latest_txs[0]
    pretty_print("Sample latest tx (raw):", sample_tx)

    # Extract wallet (from) and contract/counterparty (to) from the sample transaction
    sample_from = sample_tx.get("from_address") or sample_tx.get("from") or sample_tx.get("fromAddress")
    sample_to = sample_tx.get("to_address") or sample_tx.get("to") or sample_tx.get("toAddress")

    if not sample_from:
        print("❌ Could not determine a 'from' address from the sample transaction. Aborting.")
        sys.exit(1)

    print(f"Using wallet for tests: {sample_from}")
    if sample_to:
        print(f"Using 'to' address for contract tests (if applicable): {sample_to}")
    else:
        print("No 'to' address found in the sample tx; contract tests will be skipped or attempt fallback later.")
    print("-" * 40)

    # --- 3. Test get_transaction_details for the sample tx hash (sanity check) ---
    print("--- 3. Test get_transaction_details (sanity) ---")
    test_tx_hash = sample_tx.get("transaction_hash") or sample_tx.get("hash")
    if not test_tx_hash:
        print("❌ Sample tx has no transaction_hash field. Skipping get_transaction_details test.")
    else:
        single_tx = fetcher.get_transaction_details(test_tx_hash)
        if not single_tx:
            print("❌ FAILED: get_transaction_details returned no data.")
        else:
            print(f"✅ Fetched single tx: {single_tx.get('transaction_hash')}")
            pretty_print("Single tx details:", single_tx)

    # --- 4. Test get_wallet_transaction_history using the dynamically obtained wallet ---
    print("--- 4. Test get_wallet_transaction_history for sample wallet ---")
    try:
        history = fetcher.get_wallet_transaction_history(sample_from, limit=5)
        print(f"✅ Fetched {len(history)} history txs for {sample_from}.")
        if history:
            pretty_print("Sample history tx:", history[0])
    except Exception as e:
        print(f"❌ FAILED: get_wallet_transaction_history: {e}")

    # --- 5. Test get_contract_transactions using the sample_to if present ---
    print("--- 5. Test get_contract_transactions (only if 'to' address present) ---")
    if sample_to:
        try:
            contract_txs = fetcher.get_contract_transactions(sample_to, limit=5)
            print(f"✅ Fetched {len(contract_txs)} contract txs for {sample_to}.")
            if contract_txs:
                pretty_print("Sample contract tx:", contract_txs[0])
        except Exception as e:
            print(f"❌ FAILED: get_contract_transactions: {e}")
    else:
        print("Skipping contract tx test because sample tx had no 'to' address.")
    print("-" * 40)

    # --- 6. Test estimate_wallet_age for the sample wallet ---
    print("--- 6. Test estimate_wallet_age (for sample wallet) ---")
    try:
        age = fetcher.estimate_wallet_age(sample_from, int(time.time()))
        print(f"✅ Wallet age for {sample_from}: {age} days")
    except Exception as e:
        print(f"❌ FAILED: estimate_wallet_age: {e}")

    # --- 7. Test estimate_transaction_velocity (for sample wallet) ---
    print("--- 7. Test estimate_transaction_velocity (for sample wallet) ---")
    try:
        velocity = fetcher.estimate_transaction_velocity(sample_from)
        print("✅ Wallet velocity:")
        pretty_print("Velocity features:", velocity)
    except Exception as e:
        print(f"❌ FAILED: estimate_transaction_velocity: {e}")

    print("\n✅✅✅ Dynamic tests complete ✅✅✅")

if __name__ == "__main__":
    main()
