import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from core.data_loader import DataLoader
from core.custom_trees import DecisionTree, RandomForest, CatBoost
from core.evaluation import Evaluator
from core.drift_detector import DriftDetector

st.set_page_config(page_title="Streaming & Drift", layout="wide")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===============================
# Helpers: data loading & preprocessing
# ===============================
@st.cache_data(show_spinner="Loading and preprocessing FD001 data...")
def load_fd001():
    return DataLoader.load_local_fd001()

def plot_importance(importances, feat_names, title="Top Features"):
    names, vals = Evaluator.get_top_features(importances, feat_names, k=min(12, len(feat_names)))
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(names[::-1], vals[::-1], color="teal")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ===============================
# Load data
# ===============================
st.title("🔁 Streaming & Drift Detection")
st.caption("Hardcoded Trees • Concept Drift Detection (ADWIN) • Auto-Retraining")

st.header("📦 1. Load NASA CMAPSS (FD001)")
try:
    data = load_fd001()
except Exception as e:
    st.error(f"Failed to load dataset locally from `CMaps/`: {e}")
    st.stop()
    
train = data["train"]
feat_cols = data["feat_cols"]

c1, c2, c3 = st.columns(3)
c1.metric("Train rows", f"{len(train):,}")
c2.metric("Features", f"{len(feat_cols)}")
c3.metric("Dropped constants", f"{len(data['const_cols'])}")
st.caption("Labels: 1 → will fail within 30 cycles; 0 → healthy.")

# ===============================
# Split: 1/4 initial training, remaining as stream
# ===============================
st.header("🧪 2. Initialization & Hyperparameters")
train_fraction = st.slider("Initial training fraction", 0.1, 0.5, 0.25, 0.05)
batch_size = st.slider("Stream batch size", 100, 2000, 500, 50)
retrain_trigger = st.slider("Drifted features to trigger retrain", 1, 10, 5, 1)
acc_trigger = st.slider("Min batch accuracy (retrain if below)", 0.50, 0.95, 0.80, 0.01)
adwin_delta = st.selectbox("ADWIN delta (smaller = stricter)", [0.001, 0.002, 0.005, 0.01], index=1)

train_shuf = train.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
n0 = int(train_fraction * len(train_shuf))
init_df = train_shuf.iloc[:n0].copy()
stream_df = train_shuf.iloc[n0:].copy()

X0 = init_df[feat_cols].values
y0 = init_df["label"].astype(int).values
Xs = stream_df[feat_cols].values
ys = stream_df["label"].astype(int).values

st.write(f"Using **{train_fraction:.0%}** ({len(X0):,} rows) for initial training and streaming the remaining **{len(Xs):,}** rows.")

# ===============================
# Train initial models
# ===============================
if st.button("Train Initial Models", type="primary"):
    with st.spinner("Training hardcoded custom trees on initial slice..."):
        dt = DecisionTree(max_depth=4, min_samples_split=5, criterion='gini').fit(X0, y0)
        rf = RandomForest(n_trees=25, max_depth=6, min_samples_split=5).fit(X0, y0)
        cb = CatBoost(n_estimators=60, learning_rate=0.1, max_depth=3).fit(X0, y0)

        st.session_state.update({
            "models_ready": True,
            "dt": dt,
            "rf": rf,
            "cb": cb,
            "X_train_aug": X0.copy(),
            "y_train_aug": y0.copy(),
            "drift_det": DriftDetector(delta=adwin_delta, num_features=len(feat_cols)),
            "logs": [],
            "pos": 0,
            "retrain_count": 0
        })

if "models_ready" in st.session_state and st.session_state.models_ready:
    st.subheader("📊 Initial Model Performance (on held-out test cycles)")
    init_models = {
        "Decision Tree": st.session_state.dt, 
        "Random Forest": st.session_state.rf, 
        "CatBoost (Hardcoded)": st.session_state.cb
    }
    init_eval = Evaluator.evaluate_all(init_models, data["X_test_last_scaled"], data["y_test_last"])
    st.dataframe(init_eval, width="stretch")

    # ===============================
    # Drift detection + streaming simulation (Real-World Optimized)
    # ===============================
    st.header("⚙️ Integrated Maintenance Orchestrator")

    with st.sidebar:
        st.subheader("🛠️ Simulation Controls")
        sim_speed = st.slider("Simulation Speed (ms)", 0, 500, 100, 50)
        drift_sensitivity = st.slider("ADWIN Sensitivity (Delta)", 0.001, 0.05, 0.01, 0.005, 
                                      help="Lower delta = More sensitive to drift.")
        retrain_threshold = st.slider("Auto-Retrain Feature Threshold", 1, 15, 5)

    c_a, c_b, c_c = st.columns([1,1,2])
    
    if c_a.button("▶️ Start / Resume", type="primary"):
        st.session_state.is_running = True
    
    if c_b.button("⏸️ Pause", type="secondary"):
        st.session_state.is_running = False
        st.rerun()

    if c_c.button("🔄 Reset Environment"):
        st.session_state.is_running = False
        st.session_state.drift_det = DriftDetector(delta=drift_sensitivity, num_features=len(feat_cols))
        st.session_state.X_train_aug = X0.copy()
        st.session_state.y_train_aug = y0.copy()
        st.session_state.pos = 0
        st.session_state.retrain_count = 0
        st.session_state.drift_history = []
        st.success("Simulation environment reset.")
        st.rerun()

    st.info("The system is currently scanning **all 21 sensors** in the background using parallel ADWIN instances.")

    # Global Sensor Matrix (Proofs all features are tracked)
    with st.expander("🌐 Global Sensor Status Matrix", expanded=True):
        st.caption("Live status of all operational monitoring points (Green=Safe, Red=Drift):")
        matrix_cols = st.columns(7)
        sensor_slots = []
        for i in range(21):
            slot = matrix_cols[i % 7].empty()
            sensor_slots.append(slot)

    # Persistent UI Slots
    status_slot = st.empty()
    metric_cols = st.columns(4)
    m_acc = metric_cols[0].empty()
    m_drift = metric_cols[1].empty()
    m_retrain = metric_cols[2].empty()
    m_stream = metric_cols[3].empty()

    chart_slot = st.empty()
    heatmap_slot = st.empty()

    # Simulation Logic
    if st.session_state.get("is_running", False):
        if "drift_history" not in st.session_state:
            st.session_state.drift_history = []

        while st.session_state.is_running and st.session_state.pos < len(Xs):
            start_idx = st.session_state.pos
            CHUNK_SIZE = 50 
            end_idx = min(start_idx + CHUNK_SIZE, len(Xs))
            
            X_sim = Xs[start_idx:end_idx]
            y_sim = ys[start_idx:end_idx]

            # 1. Update ADWIN Detectors Row-by-Row
            current_chunk_drifts = []
            drifted_sensors_total = set([h["sensor"] for h in st.session_state.drift_history if h["cycle"] >= start_idx - 500])

            for i in range(len(X_sim)):
                row = X_sim[i]
                for j, adw in enumerate(st.session_state.drift_det.adwins):
                    adw.delta = drift_sensitivity
                    if adw.update(float(row[j])):
                        st.session_state.drift_history.append({"cycle": start_idx + i, "sensor": j})
                        current_chunk_drifts.append(j)
                        drifted_sensors_total.add(j)

            # Update Matrix UI
            for idx, slot in enumerate(sensor_slots):
                color = "#DC2626" if idx in drifted_sensors_total else "#16A34A"
                label = f"S{idx+1}"
                slot.markdown(f"""
                    <div style='background-color:{color}; color:white; padding:5px; border-radius:5px; text-align:center; font-size:10px; font-weight:bold;'>
                        {label}
                    </div>
                """, unsafe_allow_html=True)

            # 2. Update Metrics & Overall Anomaly Score
            y_pred_sim = st.session_state.rf.predict(X_sim)
            from sklearn.metrics import accuracy_score
            acc_sim = accuracy_score(y_sim, y_pred_sim)
            
            # Overall Score: Higher drift count = Lower Health
            total_unique_drifts = len(set([h["sensor"] for h in st.session_state.drift_history if h["cycle"] >= start_idx - 1000]))
            health_index = max(0, 100 - (total_unique_drifts * 5)) # Each drifted sensor drops health by 5%
            
            m_acc.metric("Batch Accuracy", f"{acc_sim:.3f}")
            m_stream.metric("Cycles Processed", f"{end_idx:,}")
            m_retrain.metric("Overall Engine Health", f"{health_index}%", delta=f"{-total_unique_drifts*5 if total_unique_drifts > 0 else 0}%", delta_color="inverse")
            m_drift.metric("Detected Alarms", len(st.session_state.drift_history))

            # 3. Dynamic Plotting (Overall System View)
            with chart_slot.container():
                # Focus on a window of 500 cycles
                window_lookback = 500
                w_start = max(0, end_idx - window_lookback)
                
                # Dynamic Selection: Top 4 sensors with most RECENT activity
                if st.session_state.drift_history:
                    # Filter for recent history to make the selection 'overall' and 'changing'
                    recent_h = [h["sensor"] for h in st.session_state.drift_history if h["cycle"] >= end_idx - 1000]
                    if recent_h:
                        top_s = pd.Series(recent_h).value_counts().index[:4].tolist()
                    else:
                        top_s = [0, 1, 2, 3] # Fallback
                else:
                    top_s = [0, 1, 2, 3]

                # Ensure we have precisely 4 to keep layout stable
                while len(top_s) < 4: top_s.append(np.random.randint(0, len(feat_cols)))

                fig, axs = plt.subplots(2, 2, figsize=(10, 6))
                axs = axs.flatten()
                
                for idx, s_idx in enumerate(top_s):
                    # Optimized plotting for speed
                    axs[idx].plot(range(w_start, end_idx), Xs[w_start:end_idx, s_idx], color='#1E3A8A', lw=1.2)
                    axs[idx].set_title(f"Monitor: {feat_cols[s_idx]}", fontsize=8)
                    
                    # Highlight drift triggers
                    s_drifts = [h["cycle"] for h in st.session_state.drift_history if h["sensor"] == s_idx and h["cycle"] >= w_start]
                    for d_cyc in s_drifts:
                        axs[idx].axvline(d_cyc, color='red', alpha=0.3, linestyle='--')
                    axs[idx].grid(True, alpha=0.1)
                
                plt.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.image(buf)
                plt.close(fig)

            # 4. Check for Orchestration Trigger
            drifted_in_chunk = len(set(current_chunk_drifts))
            if drifted_in_chunk >= retrain_threshold or acc_sim < acc_trigger:
                st.session_state.is_running = False # Pause for retraining
                status_slot.warning(f"⚠️ **ORCHESTRATION TRIGGERED**: {drifted_in_chunk} sensors drifted. High variance detected.")
                
                with st.spinner("Augmenting data pool and re-optimizing custom trees..."):
                    MAX_MEMORY = 15000 
                    st.session_state.X_train_aug = np.vstack([st.session_state.X_train_aug, X_sim])[-MAX_MEMORY:]
                    st.session_state.y_train_aug = np.concatenate([st.session_state.y_train_aug, y_sim])[-MAX_MEMORY:]
                    Xsub, ysub = st.session_state.X_train_aug, st.session_state.y_train_aug
                    
                    st.session_state.dt.fit(Xsub, ysub)
                    st.session_state.rf.fit(Xsub, ysub)
                    st.session_state.cb.fit(Xsub, ysub)
                    
                    st.session_state.retrain_count += 1
                    st.session_state.drift_det.reset_adwins()
                    status_slot.success(f"✅ Maintenance Model Re-Orchestrated (Rank: {st.session_state.retrain_count})")
                    time.sleep(1.5)
                
                # Auto-resume if needed or stay paused
                st.session_state.is_running = True 

            st.session_state.pos = end_idx
            time.sleep(sim_speed / 1000)

    # Explainability Section (Shown only when simulation is stopped)
    if not st.session_state.get("is_running", False):
        st.divider()
        st.subheader("💡 Why is this 'Real-World' Optimized?")
        st.info("""
        1. **Human-in-the-Loop Simulation**: Real aircraft engines don't drift instantly. You can now tune the **ADWIN Delta** live to distinguish between "Sensor Noise" and "True Degradation."
        2. **Stable Monitoring**: We've removed frame-flicker by using high-efficiency buffer processing and localized UI updates.
        3. **Drift Intensity Heatmap**: Use this to identify which subsystems (groups of sensors) are failing in unison.
        """)

        if st.session_state.get("drift_history"):
            st.markdown("**🌡️ Live Drift Intensity Grid**")
            hist_df = pd.DataFrame(st.session_state.drift_history)
            hist_df["Sensor Name"] = hist_df["sensor"].apply(lambda x: feat_cols[x])
            pivot = pd.crosstab(hist_df["Sensor Name"], hist_df["cycle"])
            fig_h, ax_h = plt.subplots(figsize=(10, 3))
            import seaborn as sns
            sns.heatmap(pivot, cmap="YlOrRd", cbar=False, ax=ax_h, xticklabels=False)
            st.pyplot(fig_h)

    # ===============================
    # Static Analytics (Show when stopped or after simulation)
    # ===============================
    if not st.session_state.get("is_running", False) and len(st.session_state.get("logs", [])):
        st.header("📊 Final Stream Analytics")
        # (Rest of the previous logs and feature importance plots...)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Decision Tree**")
            plot_importance(st.session_state.dt.feature_importances_, feat_cols, "DT Importances")
        with col2:
            st.markdown("**Random Forest**")
            plot_importance(st.session_state.rf.feature_importances_, feat_cols, "RF Importances")
        with col3:
            st.markdown("**CatBoost**")
            plot_importance(st.session_state.cb.feature_importances_, feat_cols, "CB Importances")
