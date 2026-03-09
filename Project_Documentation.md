# ⚙️ NASA CMAPSS Predictive Maintenance: Project Documentation

This document provides a comprehensive guide to the **NASA CMAPSS FD001 Predictive Maintenance** project, covering all theoretical concepts, technical architectures, and implementation details.

---

## 🏗️ 1. Project Overview
The objective of this project is to build a robust system for **Predictive Maintenance (PdM)** using the NASA Turbofan Engine Degradation dataset. The goal is to predict whether an engine will fail within the next **30 operational cycles**.

### Key System Components
1.  **Orchestrated User Interface**: A unified Streamlit application for training, analysis, and live monitoring.
2.  **Custom Tree-Based Engine**: Manual implementations of Decision Tree, Random Forest, and CatBoost (Gradient Boosting).
3.  **Real-Time Monitor & Drift Detection**: ADWIN-based conceptual drift monitoring for live data streams.
4.  **Auto-Retrain Orchestrator**: An automated loop that detects performance drops or data shifts and triggers model retraining on augmented data.
5.  **XAI (Explainable AI)**: Using SHAP and LIME to interpret individual engine failures.

---

## 📊 2. Data Engineering & Preprocessing
**Dataset**: NASA CMAPSS FD001 (Turbofan Engine Degradation).
- **Engines**: 100 train / 100 test units.
- **Sensors**: 21 sensor signals + 3 operational settings.

### Technical Workflow:
1.  **Remaining Useful Life (RUL) Calculation**:
    - The ground truth is established by finding the maximum cycle recorded for each engine ID.
    - $RUL = Max\_Cycle - Current\_Cycle$.
2.  **Binary Labeling**:
    - A binary target is created: `1` if $RUL \leq 30$, else `0`.
3.  **Constant Feature Removal**: Sensors with zero variance (e.g., `sensor_1`, `sensor_5`, `sensor_6`, etc.) are dropped to remove noise.
4.  **Min-Max Scaling**: Features are scaled to $[0, 1]$ to ensure convergence in the model training engine.

---

## 🧠 3. Custom Model Implementations (Core Mechanics)
The project uses **Custom Hardcoded Trees** instead of standard libraries to demonstrate the underlying math/logic:

### A. Decision Tree (Custom)
- **Algorithm**: CART (Classification and Regression Trees).
- **Splitting Criterion**: 
    - **Gini Impurity** for classification: $Gi = 1 - \sum p_i^2$
    - **Mean Squared Error (MSE)** for regression residuals (used in CatBoost).
- **Mechanism**: Iteratively finds the feature and threshold that maximizes "Information Gain" or "Reduction in Variance."

### B. Random Forest (Custom)
- **Mechanism**: Bootstrap Aggregation (Bagging).
- **Implementation**:
    1.  Create $N$ trees (e.g., 20-50).
    2.  For each tree, use **Bootstrapping** (sampling rows with replacement) and **Feature Bagging** (randomly selecting features for each split).
    3.  **Voting**: The final classification is a majority vote from all trees.

### C. CatBoost (Custom Gradient Boosting)
- **Concept**: Sequential boosting where each tree corrects the errors of the previous ones.
- **Mechanism**:
    1.  Starts with an initial log-odds of zero (probability 0.5).
    2.  Calculates **residuals** (gradient of Log-Loss function): $Residual = y - current\_prob$.
    3.  Fits a **Regression Tree** to these residuals.
    4.  Updates probabilities using a **Sigmoid Activation Function**: $P = 1 / (1 + e^{-f})$.
    5.  Repeats for $N$ iterations with a specified **Learning Rate**.

---

## 🔄 4. Real-World Streaming Orchestrator
The streaming module simulates a continuous feed of engine sensor data, mimicking a real aircraft's telemetry system.

### Technical Efficiency (Solving the Flicker)
- **Problem**: Standard Streamlit apps flicker when using `st.rerun()` in a loop because the entire page layout is destroyed and recreated every frame.
- **Solution**: I implemented a **Persistent Buffer Execution** model. By using a controlled `while` loop within a single session and targeted `st.empty()` containers, the system updates only the specific metric and chart pixels that change. This provides a smooth, "real-world" industrial dashboard experience.

### Human-in-the-Loop (HITL) Logic
- **ADWIN Sensitivity ($\delta$)**: In the real world, "drift" is subjective. A sensor might have noise that isn't true degradation.
- **Control**: The system provides a live **Sensitivity Slider**. 
    - A **Lower Delta** (e.g., 0.002) makes the system alert on tiny statistical shifts (high false alarms).
    - A **Higher Delta** (e.g., 0.05) ensures the system only retrains when there is a massive engine anomaly (avoids noise).
- **Orchestration Workflow**: When drift triggers cross the user-defined Feature Threshold, the system **Auto-Orchestrates** a retrain:
    1. Pauses the telemetry stream.
    2. Identifies the "Drifted" sensor subset.
    3. Retrains the Custom Tree Engine (Decision Tree, RF, CatBoost) on augmented and historical data.
    4. Resumes monitoring with the "adapted" model.

---

## 👁️ 5. Explainable AI (XAI)
To make the predictions actionable, we use XAI techniques:

### SHAP (SHapley Additive exPlanations)
- **Theory**: Based on Cooperative Game Theory.
- **Goal**: Explain the contribution of each sensor toward the final failure prediction.
- **Logic**: It calculates the difference between the average prediction and the actual prediction for an engine, and distributes that difference among the sensors fairly.

### LIME (Local Interpretable Model-agnostic Explanations)
- **Theory**: Sparse linear models as local proxies.
- **Goal**: Explain why *this specific* engine is failing.
- **Logic**: It slightly perturbs (changes) the sensor values for an engine and sees how the model's prediction changes. It then fits a simple linear model to these changes to show which sensors are pulling the prediction toward "Failure."

---

## 🛠️ 6. Technical Stack
- **Dashboard**: Streamlit (Python)
- **Data Science**: Pandas, NumPy, Scikit-learn
- **Drift Logic**: River (ADWIN)
- **Explanation**: SHAP, LIME
- **Architecture**: Modular Python (`core/` package and `pages/` directory)

---
*End of Documentation — Prepared for Presentation.*
