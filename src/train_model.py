import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_fraud_detector():
    print("Loading data from data/transactions.csv...")
    if not os.path.exists('data/transactions.csv'):
        raise FileNotFoundError("Dataset not found! Please run 'python data/generate_data.py' first.")
        
    df = pd.DataFrame(pd.read_csv('data/transactions.csv'))
    
    # Separate features (Sensors) and target (Fraud label)
    X = df.drop(columns=['is_fraud', 'processing_latency_ms']) # Latency is an output metric, not an input feature
    y = df['is_fraud']
    
    # Split data into 80% Training and 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]} | Testing set size: {X_test.shape[0]}")
    
    # Initialize and train the Random Forest Agent
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    print("\n" + "="*40)
    print("          MODEL EVALUATION METRICS          ")
    print("="*40)
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate False Positive Rate (FPR) specifically for your PEAS assessment criteria
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (tn + fp)
    print(f"False Positive Rate (FPR): {fpr * 100:.2f}% (Target: Minimize to protect customer experience)")
    print(f"Fraud Detection Rate (Recall): {tp / (tp + fn) * 100:.2f}% (Target: Maximize to stop fraud)")
    print("="*40)
    
    # Ensure models/ directory exists and save model weights
    os.makedirs('models', exist_ok=True)
    with open('models/fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("\nModel saved successfully at 'models/fraud_model.pkl'!")

if __name__ == "__main__":
    train_fraud_detector()