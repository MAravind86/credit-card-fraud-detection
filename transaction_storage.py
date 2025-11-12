"""
Transaction Storage Module
Handles storing and retrieving transaction history
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import threading

class TransactionStorage:
    """Thread-safe storage for transaction history"""
    
    def __init__(self, storage_file='transactions.json'):
        self.storage_file = storage_file
        self.lock = threading.Lock()
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create storage file if it doesn't exist"""
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, 'w') as f:
                json.dump([], f)
    
    def save_transaction(self, transaction_data: Dict) -> bool:
        """Save a transaction to storage"""
        try:
            with self.lock:
                # Read existing transactions
                if os.path.exists(self.storage_file) and os.path.getsize(self.storage_file) > 0:
                    with open(self.storage_file, 'r') as f:
                        transactions = json.load(f)
                else:
                    transactions = []
                
                # Add metadata
                transaction_data['stored_at'] = datetime.now().isoformat()
                transaction_data['id'] = len(transactions) + 1
                
                # Append new transaction
                transactions.append(transaction_data)
                
                # Write back
                with open(self.storage_file, 'w') as f:
                    json.dump(transactions, f, indent=2)
                
                return True
        except Exception as e:
            print(f"Error saving transaction: {e}")
            return False
    
    def get_all_transactions(self, limit: Optional[int] = None, 
                            offset: int = 0) -> List[Dict]:
        """Get all transactions, optionally paginated"""
        try:
            with self.lock:
                if not os.path.exists(self.storage_file) or os.path.getsize(self.storage_file) == 0:
                    return []
                
                with open(self.storage_file, 'r') as f:
                    transactions = json.load(f)
                
                # Reverse to show newest first
                transactions = list(reversed(transactions))
                
                # Pagination
                if limit:
                    transactions = transactions[offset:offset+limit]
                
                return transactions
        except Exception as e:
            print(f"Error reading transactions: {e}")
            return []
    
    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict]:
        """Get a specific transaction by transaction_id"""
        try:
            with self.lock:
                if not os.path.exists(self.storage_file) or os.path.getsize(self.storage_file) == 0:
                    return None
                
                with open(self.storage_file, 'r') as f:
                    transactions = json.load(f)
                
                for tx in reversed(transactions):  # Check newest first
                    if tx.get('transaction_id') == transaction_id:
                        return tx
                
                return None
        except Exception as e:
            print(f"Error finding transaction: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Calculate statistics from stored transactions"""
        try:
            with self.lock:
                if not os.path.exists(self.storage_file) or os.path.getsize(self.storage_file) == 0:
                    return {
                        'total_transactions': 0,
                        'fraud_count': 0,
                        'legitimate_count': 0,
                        'high_risk_count': 0,
                        'medium_risk_count': 0,
                        'low_risk_count': 0,
                        'total_amount': 0,
                        'fraud_amount': 0,
                        'fraud_rate': 0
                    }
                
                with open(self.storage_file, 'r') as f:
                    transactions = json.load(f)
                
                stats = {
                    'total_transactions': len(transactions),
                    'fraud_count': 0,
                    'legitimate_count': 0,
                    'high_risk_count': 0,
                    'medium_risk_count': 0,
                    'low_risk_count': 0,
                    'total_amount': 0,
                    'fraud_amount': 0,
                    'fraud_rate': 0
                }
                
                for tx in transactions:
                    if 'amount' in tx:
                        stats['total_amount'] += float(tx['amount'])
                    
                    prediction = tx.get('fraud_prediction', '').upper()
                    if prediction == 'FRAUD':
                        stats['fraud_count'] += 1
                        if 'amount' in tx:
                            stats['fraud_amount'] += float(tx['amount'])
                    elif prediction == 'LEGITIMATE':
                        stats['legitimate_count'] += 1
                    
                    risk = tx.get('risk_level', '').upper()
                    if risk == 'HIGH':
                        stats['high_risk_count'] += 1
                    elif risk == 'MEDIUM':
                        stats['medium_risk_count'] += 1
                    elif risk == 'LOW':
                        stats['low_risk_count'] += 1
                
                if stats['total_transactions'] > 0:
                    stats['fraud_rate'] = (stats['fraud_count'] / stats['total_transactions']) * 100
                
                return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return {}
    
    def clear_history(self) -> bool:
        """Clear all transaction history"""
        try:
            with self.lock:
                with open(self.storage_file, 'w') as f:
                    json.dump([], f)
                return True
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False

