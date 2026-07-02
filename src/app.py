import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os

# Set up page configurations
st.set_page_config(page_title="AI Fraud Detection System", layout="wide")

# Title and Description
st.title("🛡️ AI Fraud Detection System Dashboard")
st.subheader("Lab Demonstration - Introduction to Artificial Intelligence (PEAS Framework)")
st.markdown("---")

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'models/fraud_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

if model is None:
    st.error("❌ Model file not found! Please run 'python src/train_model.py' first.")
else:
    # Sidebar layout for Sensors inputs
    st.sidebar.header("🎛️ Agent Sensors (Inputs)")
    
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=45.0, step=10.0)
    hour_of_day = st.sidebar.slider("Hour of Day (Timestamp)", 0, 23, 14)
    location_mismatch = st.sidebar.selectbox("Location/IP Mismatch Status", options=[0, 1], format_func=lambda x: "1 - High-Risk / Foreign IP" if x == 1 else "0 - Domestic / Matched")
    device_trust_score = st.sidebar.slider("Device Trust Score (Fingerprint Token)", 0.0, 1.0, 0.85, step=0.05)
    behavior_anomaly_score = st.sidebar.slider("Behavioral Anomaly Score (Historical Profile)", 0.0, 1.0, 0.15, step=0.05)

    # Main Dashboard layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🔄 Transaction Real-Time Inference Pipeline")
        st.write("Click below to pass the input telemetry data through the trained Random Forest Classifier.")
        
        if st.button("Evaluate Transaction", type="primary"):
            with st.spinner("Processing transaction telemetry..."):
                # Simulate sensor reading delays & computational processing latency
                start_time = time.time()
                time.sleep(np.random.uniform(0.01, 0.03)) 
                
                # Construct feature dataframe matching training columns
                input_data = pd.DataFrame([{
                    'amount': amount,
                    'hour_of_day': hour_of_day,
                    'location_mismatch': location_mismatch,
                    'device_trust_score': device_trust_score,
                    'behavior_anomaly_score': behavior_anomaly_score
                }])
                
                # Model predicts probability and class
                prediction_prob = model.predict_proba(input_data)[0][1]
                prediction = model.predict(input_data)[0]
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms

            st.success("Analysis Complete!")
            
            # --- DISPLAY EVALUATION DETAILS ---
            st.markdown("### 📊 Inference Diagnostic Readout")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Calculated Fraud Risk", f"{prediction_prob * 100:.1f}%")
            metric_col2.metric("Processing Latency", f"{latency:.2f} ms")
            
            # Formulate utility function value for demonstration purposes
            # U = 100 * (1 - risk) - (amount if fraud occurs)
            utility_score = 100 * (1 - prediction_prob) if prediction == 0 else -amount
            metric_col3.metric("Agent Utility State", f"{utility_score:.2f}")

            st.markdown("---")
            st.markdown("### ⚙️ Executed Agent Actuators (Actions)")
            
            if prediction == 1 or prediction_prob > 0.5:
                st.error("🚨 **ACTION EXECUTED: TRANSACTION BLOCKED & ACCOUNT FROZEN**")
                st.markdown("""
                * **Transaction Status Flag:** `DECLINE` sent back to gateway API.
                * **Notification Trigger:** Real-time verification SMS and high-priority push notification pushed to user's device.
                * **Account Access Modifier:** Card state temporarily shifted to `LOCKED` pending customer authentication.
                """)
            else:
                st.success("✅ **ACTION EXECUTED: TRANSACTION APPROVED**")
                st.markdown("""
                * **Transaction Status Flag:** `APPROVE` token returned to network gateway.
                * **Notification Trigger:** standard transactional receipt emailed to user.
                * **Account Access Modifier:** None. Balance updated successfully.
                """)

    with col2:
        st.header("📋 PEAS Checklist Validation")
        st.markdown("""
        **Performance Measures Mapping:**
        * ✅ **FPR Tracking:** Controls false positives through threshold adjustments.
        * ✅ **Latency:** Real-time evaluation logged in under 50ms constraint.
        
        **Environment Characteristics:**
        * 🔍 *Partially Observable* (Model acts on statistical risk probabilities, not direct internal state intent).
        * ⚡ *Dynamic & Stochastic* (Input patterns fluctuate dynamically over continuous intervals).
        """)