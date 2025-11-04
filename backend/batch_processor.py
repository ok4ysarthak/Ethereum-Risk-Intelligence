# backend/batch_processor.py
# import schedule
import time
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, data_fetcher, fraud_detector, wallet_scorer):
        self.data_fetcher = data_fetcher
        self.fraud_detector = fraud_detector
        self.wallet_scorer = wallet_scorer
        self.processed_transactions = set()
        self.setup_logging()
    
    def setup_logging(self):
        # Logging is already configured in main.py
        pass
    
    def process_new_transactions(self):
        """Process new transactions and update wallet scores"""
        logger.info("Starting batch processing...")
        
        try:
            # Fetch latest transactions
            transactions = self.data_fetcher.get_latest_transactions(50)
            logger.info(f"Fetched {len(transactions)} transactions")
            
            results = []
            for tx in transactions:
                tx_identifier = f"{tx.get('from_address', '')}_{tx.get('timestamp', '')}"
                
                if tx_identifier in self.processed_transactions:
                    continue
                
                try:
                    # Predict fraud risk
                    fraud_risk = self.fraud_detector.predict_fraud_risk(tx)
                    
                    # Update wallet history
                    from_address = tx.get('from_address')
                    if from_address:
                        self.wallet_scorer.update_wallet_history(from_address, tx)
                        
                        # Calculate wallet score
                        wallet_score = self.wallet_scorer.calculate_wallet_score(
                            from_address, 
                            self.wallet_scorer.wallet_history[from_address]
                        )
                        
                        # Store wallet score
                        self.wallet_scorer.wallet_scores[from_address] = wallet_score
                    
                    results.append({
                        'transaction': tx_identifier,
                        'fraud_risk': fraud_risk,
                        'wallet_score': wallet_score if from_address else None
                    })
                    
                    self.processed_transactions.add(tx_identifier)
                    
                except Exception as e:
                    logger.error(f"Error processing transaction: {e}")
                    continue
            
            logger.info(f"Processed {len(results)} transactions in batch")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
    
    def start_scheduler(self):
        """Start the batch processing scheduler"""
        # This method can be used if you want schedule to run within this class
        pass
