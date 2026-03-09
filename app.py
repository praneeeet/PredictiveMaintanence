import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Orchestrator",
    page_icon="⚙️",
    layout="wide",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
        color: #1F2937;
    }
    .card h3 {
        margin-top: 0;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚙️ Predictive Maintenance Orchestrator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NASA CMAPSS FD001 Engine Degradation Analysis</div>', unsafe_allow_html=True)

st.info("""
This application orchestrates the end-to-end predictive maintenance workflow using **custom hardcoded models**:
- **Decision Tree** (Custom Manual Implementation)
- **Random Forest** (Bootstrap Aggregated Trees)
- **CatBoost** (Iterative Gradient Boosting with sigmoid-based classification)
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>📊 1. Model Training & Analysis</h3>
        <p>Explore the NASA CMAPSS dataset, visualize sensor degradation, analyze outliers, and train the custom hardcoded models.</p>
        <p><b>Features:</b> Hold-out & K-Fold validation, SHAP & LIME Interpretability, Feature Importance.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>🔁 2. Streaming & Drift Detection</h3>
        <p>Simulate real-time data streaming and monitor for <b>Concept Drift</b> using ADWIN (ADaptive WINdowing).</p>
        <p><b>Orchestration:</b> Orchestrates automatic retraining when performance drops below threshold or significant drift is detected in sensors.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.warning("**Getting Started:** Use the sidebar on the left to navigate between the Training and Streaming modules.")

with st.expander("System Architecture Overview"):
    st.markdown("""
    - **Data Layer:** NASA CMAPSS FD001 raw signals (21 sensors + settings).
    - **Preprocessing:** Constant feature removal, RUL labeling, and MinMax scaling.
    - **Custom Models:** Manual implementations of Decision Tree, Random Forest, and CatBoost.
    - **Drift Detection:** ADWIN algorithm monitoring sensor signals in real-time.
    - **Orchestration:** Automatic retraining loop triggered by drift flags or performance decay.
    """)

st.markdown("---")
st.markdown("### 📚 Project Resources")
st.success("A comprehensive **Technical & Theoretical Guide** has been generated for your presentation.")
st.markdown("[📄 Open Project Documentation](https://github.com/praneeeet/PredictiveMaintanence/blob/main/Project_Documentation.md)")

st.caption("Developed for Predictive Maintenance Excellence.")