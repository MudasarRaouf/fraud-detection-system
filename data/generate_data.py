import pandas as pd
import numpy as np
import os

def generate_fraud_dataset(num_samples=5000):
    np.random.seed(42)
    
    # 1. Generate Transaction  Metadata (Sensors)
    amount = np.random.exponential(scale=50, size=num_samples) + 1  # Most transactions are small
    # Add a few massive transactions (often a fraud indicator)
    high_value_indices = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
    amount[high_value_indices] = np.random.uniform(500, 5000, size=len(high_value_indices))
    
    # Simulate processing time/latency (in milliseconds)
    processing_latency = np.random.normal(loc=15, scale=5, size=num_samples)
    processing_latency = np.clip(processing_latency, 2, 50)
    
    # Hour of day (0 to 23)
    hour = np.random.randint(0, 24, size=num_samples)
    
    # 2. User Telemetry & Location Risk (Sensors)
    # 0 = Domestic/Trusted, 1 = High-risk international IP/Location mismatch
    location_mismatch = np.random.choice([0, 1], size=num_samples, p=[0.93, 0.07])
    
    # Device fingerprint trust score (0.0 to 1.0, lower means suspicious/new device)
    device_trust_score = np.random.uniform(0.3, 1.0, size=num_samples)
    
    # Historical behavioral anomaly score (0.0 to 1.0, higher means deviating from past habits)
    behavior_anomaly_score = np.random.uniform(0.0, 0.6, size=num_samples)
    
    # 3. Inject Rules to Label Fraud (Target Variable: is_fraud)
    is_fraud = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # High risk conditions
        score = 0
        if amount[i] > 1000: score += 3
        if location_mismatch[i] == 1: score += 2
        if device_trust_score[i] < 0.5: score += 2
        if hour[i] >= 2 and hour[i] <= 5: score += 1  # Late night transactions
        if behavior_anomaly_score[i] > 0.5: score += 2
        
        # Ground truth rule assignment with a touch of stochastic randomness
        if score >= 5 or (score >= 3 and np.random.rand() > 0.3):
            is_fraud[i] = 1
            # Adjust feature distributions for fraud cases to help the AI learn patterns
            device_trust_score[i] = np.random.uniform(0.0, 0.4)
            behavior_anomaly_score[i] = np.random.uniform(0.6, 1.0)

    # 4. Create  DataFrame
    df = pd.DataFrame({
        'amount': np.round(amount, 2),
        'hour_of_day': hour,
        'location_mismatch': location_mismatch,
        'device_trust_score': np.round(device_trust_score, 2),
        'behavior_anomaly_score': np.round(behavior_anomaly_score, 2),
        'processing_latency_ms': np.round(processing_latency, 2),
        'is_fraud': is_fraud
    })
    
    # Ensure data directory exists and save
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Dataset generated successfully! Saved 5,000 records to 'data/transactions.csv'")
    print(f"Total Fraud Cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

if __name__ == "__main__":
    generate_fraud_dataset()
