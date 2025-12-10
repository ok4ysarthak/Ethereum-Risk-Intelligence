# backend/verify_data.py
from Database.db import SessionLocal
from Database.models_db import Transaction, Wallet
import sys

def verify():
    print("🔌 Connecting to Database...")
    try:
        session = SessionLocal()
        
        # 1. Check Wallets
        wallet_count = session.query(Wallet).count()
        print(f"\nWALLETS FOUND: {wallet_count}")
        
        if wallet_count > 0:
            print("   Latest 3 Wallets:")
            for w in session.query(Wallet).order_by(Wallet.last_seen.desc()).limit(3):
                print(f"   - {w.address} (Score: {w.trust_score})")

        # 2. Check Transactions
        tx_count = session.query(Transaction).count()
        print(f"\nTRANSACTIONS FOUND: {tx_count}")
        
        if tx_count > 0:
            print("   Latest 3 Transactions:")
            for t in session.query(Transaction).order_by(Transaction.timestamp.desc()).limit(3):
                print(f"   - {t.tx_hash[:15]}... | Risk: {t.risk_score}")
        
        session.close()
        
    except Exception as e:
        print(f"DATABASE ERROR: {e}")
        print("Tip: Ensure your database is running and credentials in .env are correct.")

if __name__ == "__main__":
    verify()