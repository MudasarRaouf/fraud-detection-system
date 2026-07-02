# 🛡️ AI-Powered Fraud Detection System for Digital Payments

An interactive, real-time Machine Learning application designed to evaluate financial transaction telemetry, assess risk profiles, and execute agent-based automated actions to mitigate fraud. Developed as part of the **Introduction to Artificial Intelligence** curriculum (Lab Track Option B).

---

## 👥 Team Members & Project Contributors

* **👑 Project Leader:** Mudasar Ali — *Roll No: 2k23/CSM/80*
* **💻 Team Member:** Devi Khatri — *Roll No: 2k23/CSM/33*
* **💻 Team Member:** Klapna — *Roll No: 2k23/CSM/55*

---

## 🏗️ System Architecture & Workflow

The system acts as an intelligent agent processing raw sensor inputs to determine optimal risk classifications and execute security compliance actuators.



1. **Telemetry Capture:** Real-time collection of transaction metadata and user device sensor values.
2. **Inference Pipeline:** Data features are parsed by a trained Random Forest Classifier pipeline.
3. **Actuator Execution:** Based on risk boundaries, the agent dynamically triggers automated system overrides (Approve, Deny, Freeze Account).

---

## 🧠 PEAS Framework Specification

### 📈 Performance Measures
* **False Positive Rate (FPR):** Minimizing operational friction for legitimate cardholders.
* **Fraud Detection Recall:** Maximizing the rate of accurately intercepted unauthorized transactions.
* **Processing Latency:** Keeping per-transaction evaluation times under a strict **50ms** constraint.
* **Total Financial Loss Prevented:** Maximizing the preservation of core transactional capital.

### 🌐 Environment
* Global digital payment gateways, web application interfaces, core banking ledgers, and adversarial fraudulent threat actors.

### ⚙️ Actuators
* **Transaction Status Flags:** `APPROVE`, `DECLINE`, or `HOLD` response status maps.
* **Notification Triggers:** Direct multi-channel API hooks for out-of-band verification alerts (SMS/Push notifications).
* **Access Modifiers:** Real-time temporary card suspensions and system-enforced password expirations.

### 🎛️ Sensors
* **Transaction Metadata:** Value amounts ($), transaction times (hour of day), and merchant codes.
* **User Telemetry:** Location flags, client IP space, and device fingerprint tokens.
* **Historical Profiles:** Baselines tracking standard consumer behavioral variances.

---

## 📊 Environment Classification

| Dimension | Type | Justification |
| :--- | :--- | :--- |
| **Observability** | **Partially Observable** | The agent cannot explicitly observe a user's intent or hidden credentials. |
| **Agents** | **Multi-Agent** | Operates against malicious fraudsters and alongside compliance operators. |
| **Determinism** | **Stochastic** | The environment state transitions contain highly unpredictable human patterns. |
| **Episodic/Sequential** | **Sequential** | Current choices influence subsequent user telemetry paths and profile history logs. |
| **Static/Dynamic** | **Dynamic** | Independent background transactions run concurrently during model inference. |
| **Discrete/Continuous** | **Continuous** | Amounts, processing delays, and risk metric spaces scale on a continuous plane. |

---

## 🛠️ Local Setup and Execution Instructions

To run this project on a demonstration computer during evaluation, follow these quick terminal steps:

### 1. Replicate Dependencies
Ensure Python is installed on your device. Open your terminal inside the root project directory and install the necessary libraries:
```bash
pip install -r requirements.txt
