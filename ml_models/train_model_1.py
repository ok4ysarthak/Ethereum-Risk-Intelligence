# ml_models/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

# Add backend to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fraud_detector import EthereumFraudDetector

def train_model_from_csv(csv_path):
    """Train the fraud detection model from CSV"""
    print(f"Loading dataset from: {csv_path}")
    
    # Load your dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded with {len(df)} rows")
    
    # Preprocess data
    df = df.fillna(0)
    
    # Feature engineering
    df['Value_to_Fee_Ratio'] = df['Transaction_Value'] / (df['Transaction_Fees'] + 1e-8)
    df['Input_Output_Ratio'] = df['Number_of_Inputs'] / (df['Number_of_Outputs'] + 1e-8)
    df['Gas_Efficiency'] = df['Transaction_Value'] / (df['Gas_Price'] + 1e-8)
    df['Balance_Turnover'] = df['Transaction_Value'] / (df['Wallet_Balance'] + 1e-8)
    
    # Create target variable
    df['is_fraud'] = df.get('Is_Scam', 0).astype(int)
    
    feature_columns = [
        'Transaction_Value', 'Transaction_Fees', 'Number_of_Inputs', 
        'Number_of_Outputs', 'Gas_Price', 'Wallet_Age_Days', 
        'Wallet_Balance', 'Transaction_Velocity', 'Exchange_Rate',
        'Value_to_Fee_Ratio', 'Input_Output_Ratio', 
        'Gas_Efficiency', 'Balance_Turnover'
    ]
    
    X = df[feature_columns]
    y = df['is_fraud']
    
    print(f"Features: {len(feature_columns)} columns")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train detector
    detector = EthereumFraudDetector()
    
    # Scale features
    X_train_scaled = detector.scaler.fit_transform(X_train)
    
    # Train model
    print("Training model...")
    detector.model.fit(X_train_scaled, y_train)
    print("Model training completed!")
    
    # Save the model
    save_path = 'saved_models'
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, 'fraud_detector_model.pkl')
    joblib.dump(detector, model_file)
    print(f"✅ Model trained and saved successfully to {model_file}!")
    
    return detector

# Train the model
if __name__ == "__main__":
    # Update this path to your actual dataset location
    dataset_path = '../data/ethereum_fraud_dataset.csv'
    try:
        detector = train_model_from_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset not found at {dataset_path}")
        print("Please make sure your dataset is in the correct location")
