"""
Credit Card Fraud Detection System
Simple interface for predicting fraud based on Transaction ID, Time, and Amount
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import random

class SimpleFraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def load_and_prepare_data(self, csv_file='creditcard.csv'):
        """Load and prepare the credit card dataset"""
        print("Loading credit card dataset...")
        self.df = pd.read_csv(csv_file)
        
        print(f"Dataset loaded: {self.df.shape[0]} transactions")
        print(f"Fraud cases: {self.df['Class'].sum()} ({self.df['Class'].mean()*100:.2f}%)")
        
        # Feature engineering
        self.df['hour'] = (self.df['Time'] / 3600) % 24
        self.df['amount_log'] = np.log1p(self.df['Amount'])
        
        # Select important features
        self.feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                               'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                               'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
                               'hour', 'amount_log']
        
        return self.df
    
    def balance_data(self):
        """Balance the dataset to handle class imbalance"""
        print("Balancing dataset...")
        
        # Separate classes
        fraud = self.df[self.df['Class'] == 1]
        normal = self.df[self.df['Class'] == 0]
        
        # Undersample normal transactions (2:1 ratio)
        normal_downsampled = resample(normal, 
                                   replace=False, 
                                   n_samples=len(fraud) * 2, 
                                   random_state=42)
        
        # Combine datasets
        self.df_balanced = pd.concat([normal_downsampled, fraud])
        
        print(f"Balanced dataset: {self.df_balanced.shape[0]} transactions")
        print(f"Class distribution: {self.df_balanced['Class'].value_counts().to_dict()}")
        
        return self.df_balanced
    
    def train_model(self):
        """Train the fraud detection model"""
        print("Training fraud detection model...")
        
        # Prepare data
        X = self.df_balanced[self.feature_columns]
        y = self.df_balanced['Class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("Model Performance:")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return self.model
    
    def save_model(self):
        """Save the trained model"""
        if self.is_trained:
            joblib.dump(self.model, 'fraud_model.pkl')
            joblib.dump(self.scaler, 'fraud_scaler.pkl')
            print("Model saved successfully!")
        else:
            print("Model not trained yet!")
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = joblib.load('fraud_model.pkl')
            self.scaler = joblib.load('fraud_scaler.pkl')
            self.is_trained = True
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Please train the model first.")
    
    def predict_fraud(self, transaction_id, time, amount):
        """
        Predict if a transaction is fraudulent using improved feature engineering
        
        Args:
            transaction_id (str): Unique transaction identifier
            time (float): Transaction time in seconds
            amount (float): Transaction amount
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            return {
                'error': 'Model not trained. Please train the model first.',
                'transaction_id': transaction_id
            }
        
        # Calculate time-based features
        hour = (time / 3600) % 24
        day_of_week = datetime.fromtimestamp(time).weekday()
        
        # Enhanced feature engineering based on fraud patterns
        # High amounts are more likely to be fraud
        amount_risk = 1 if amount > 1000 else (0.5 if amount > 500 else 0)
        
        # Night time transactions (2-6 AM) are more suspicious
        night_risk = 1 if 2 <= hour <= 6 else 0
        
        # Weekend transactions have different risk patterns
        weekend_risk = 1 if day_of_week >= 5 else 0
        
        # Create more realistic V1-V28 features based on amount and time patterns
        # These simulate PCA features that would correlate with fraud
        base_features = self._generate_realistic_features(amount, hour, day_of_week)
        
        transaction_features = {
            'Time': time,
            'V1': base_features['V1'],
            'V2': base_features['V2'],
            'V3': base_features['V3'],
            'V4': base_features['V4'],
            'V5': base_features['V5'],
            'V6': base_features['V6'],
            'V7': base_features['V7'],
            'V8': base_features['V8'],
            'V9': base_features['V9'],
            'V10': base_features['V10'],
            'V11': base_features['V11'],
            'V12': base_features['V12'],
            'V13': base_features['V13'],
            'V14': base_features['V14'],
            'V15': base_features['V15'],
            'V16': base_features['V16'],
            'V17': base_features['V17'],
            'V18': base_features['V18'],
            'V19': base_features['V19'],
            'V20': base_features['V20'],
            'V21': base_features['V21'],
            'V22': base_features['V22'],
            'V23': base_features['V23'],
            'V24': base_features['V24'],
            'V25': base_features['V25'],
            'V26': base_features['V26'],
            'V27': base_features['V27'],
            'V28': base_features['V28'],
            'Amount': amount,
            'hour': hour,
            'amount_log': np.log1p(amount)
        }
        
        # Convert to DataFrame
        transaction_df = pd.DataFrame([transaction_features])
        
        # Select features and scale
        X = transaction_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        fraud_probability = self.model.predict_proba(X_scaled)[0, 1]
        fraud_prediction = self.model.predict(X_scaled)[0]
        
        # Apply business rules to enhance prediction
        enhanced_probability = self._apply_business_rules(fraud_probability, amount, hour, day_of_week)
        
        # Determine risk level based on fraud score (0-100 scale)
        # LOW: 0-40, MEDIUM: 41-70, HIGH: 71-100
        fraud_score = enhanced_probability * 100  # Convert to 0-100 scale
        if fraud_score >= 71:
            risk_level = "HIGH"
        elif fraud_score >= 41:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            'transaction_id': transaction_id,
            'timestamp': timestamp,
            'amount': amount,
            'fraud_prediction': 'FRAUD' if enhanced_probability >= 0.5 else 'LEGITIMATE',
            'fraud_probability': round(enhanced_probability, 4),
            'fraud_score': round(fraud_score, 2),  # Fraud score on 0-100 scale
            'risk_level': risk_level,
            'confidence': 'High' if enhanced_probability > 0.8 or enhanced_probability < 0.2 else 'Medium',
            'risk_factors': self._get_risk_factors(amount, hour, day_of_week)
        }
    
    def _generate_realistic_features(self, amount, hour, day_of_week):
        """Generate realistic V1-V28 features based on transaction patterns"""
        # Seed random with transaction characteristics for consistency
        np.random.seed(int(amount * 100 + hour * 10 + day_of_week))
        
        features = {}
        
        # V1-V5: Amount-related features (higher amounts = more extreme values)
        amount_factor = min(amount / 1000, 3)  # Cap at 3x normal
        for i in range(1, 6):
            features[f'V{i}'] = np.random.normal(0, 1 + amount_factor)
        
        # V6-V10: Time-related features (night time = more extreme values)
        time_factor = 1.5 if 2 <= hour <= 6 else 1.0
        for i in range(6, 11):
            features[f'V{i}'] = np.random.normal(0, 1 * time_factor)
        
        # V11-V15: Weekend patterns
        weekend_factor = 1.3 if day_of_week >= 5 else 1.0
        for i in range(11, 16):
            features[f'V{i}'] = np.random.normal(0, 1 * weekend_factor)
        
        # V16-V20: Interaction features (amount * time risk)
        interaction_factor = amount_factor * time_factor
        for i in range(16, 21):
            features[f'V{i}'] = np.random.normal(0, 1 + interaction_factor * 0.5)
        
        # V21-V28: Random but consistent features
        for i in range(21, 29):
            features[f'V{i}'] = np.random.normal(0, 1)
        
        return features
    
    def _apply_business_rules(self, base_probability, amount, hour, day_of_week):
        """Apply business rules to enhance fraud probability"""
        enhanced_prob = base_probability
        
        # Hard rule: high-value night transactions are almost certainly fraud
        # 1.5 Lakhs (₹150,000) during night hours (00:00–06:00)
        if amount >= 150000 and 0 <= hour <= 6:
            return 0.95

        # High amount transactions are more suspicious
        if amount > 2000:
            enhanced_prob += 0.2
        elif amount > 1000:
            enhanced_prob += 0.1
        elif amount > 500:
            enhanced_prob += 0.05
        
        # Night time transactions (2-6 AM) are more suspicious
        if 2 <= hour <= 6:
            enhanced_prob += 0.15
        
        # Very early morning (12-2 AM) also suspicious
        elif 0 <= hour <= 2:
            enhanced_prob += 0.1
        
        # Weekend high-value transactions
        if day_of_week >= 5 and amount > 1000:
            enhanced_prob += 0.1
        
        # Cap probability at 0.95
        return min(enhanced_prob, 0.95)
    
    def _get_risk_factors(self, amount, hour, day_of_week):
        """Get human-readable risk factors"""
        factors = []
        
        if amount >= 150000 and 0 <= hour <= 6:
            factors.append("High-value night transaction (\u2265 \u20B91.5 Lakhs, 00:00–06:00)")
        elif amount > 20000:
            factors.append("High transaction amount")
        elif amount > 10000:
            factors.append("Above-average transaction amount")
        
        if 2 <= hour <= 6:
            factors.append("Night time transaction")
        elif 0 <= hour <= 2:
            factors.append("Late night transaction")
        
        if day_of_week >= 5 and amount > 1000:
            factors.append("Weekend high-value transaction")
        
        return factors if factors else ["Normal transaction patterns"]

def main():
    """Main function to demonstrate the system"""
    print("=== Credit Card Fraud Detection System ===\n")
    
    # Initialize detector
    detector = SimpleFraudDetector()
    
    # Load and prepare data
    detector.load_and_prepare_data()
    
    # Balance data
    detector.balance_data()
    
    # Train model
    detector.train_model()
    
    # Save model
    detector.save_model()
    
    print("\n" + "="*60)
    print("FRAUD DETECTION SYSTEM READY")
    print("="*60)
    
    # Interactive prediction
    while True:
        print("\nEnter transaction details:")
        try:
            txn_id = input("Transaction ID: ").strip()
            if txn_id.lower() == 'quit':
                break
                
            print("Time options:")
            print("1. Enter 'now' for current time")
            print("2. Enter time in format 'HH:MM' (e.g., '14:30')")
            print("3. Enter seconds from epoch (e.g., '1640995200')")
            time_input = input("Time: ").strip()
            
            if time_input.lower() == 'now':
                time = datetime.now().timestamp()
            elif ':' in time_input:
                # Handle HH:MM format
                try:
                    hour, minute = map(int, time_input.split(':'))
                    # Create a datetime for today with the specified time
                    today = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
                    time = today.timestamp()
                except ValueError:
                    print("Invalid time format. Please use HH:MM format (e.g., '14:30')")
                    continue
            else:
                try:
                    time = float(time_input)
                except ValueError:
                    print("Invalid time format. Please enter 'now', 'HH:MM', or seconds from epoch.")
                    continue
            
            amount_input = input("Amount: $").strip()
            try:
                amount = float(amount_input)
            except ValueError:
                print("Invalid amount. Please enter a numeric value.")
                continue
            
            # Make prediction
            result = detector.predict_fraud(txn_id, time, amount)
            
            print("\n" + "="*50)
            print("FRAUD DETECTION RESULT")
            print("="*50)
            print(f"Transaction ID: {result['transaction_id']}")
            print(f"Timestamp: {result['timestamp']}")
            print(f"Amount: ${result['amount']}")
            print(f"Prediction: {result['fraud_prediction']}")
            print(f"Fraud Probability: {result['fraud_probability']}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Confidence: {result['confidence']}")
            print("="*50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
