import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
plt.style.use("default")

from core.data_loader import DataLoader
from core.models import get_models
from core.evaluation import Evaluator

# XAI
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

st.set_page_config(page_title="Model Training | Predictive Maintenance", layout="wide")
st.title("Model Training & Validation")
st.caption("Decision Tree • Random Forest • CatBoost (Hardcoded) | SHAP + LIME | K-Fold + Hold-out")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------
# Preprocessing wrapper
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Preprocessing data...")
def load_and_preprocess(train_file, test_file, rul_file):
    return DataLoader.load_from_files(train_file, test_file, rul_file)

# ----------------------------------------------------------------------
# UI: Upload
# ----------------------------------------------------------------------
st.header("1. Upload Files")
c1, c2, c3 = st.columns(3)
train_file = c1.file_uploader("`train_FD001.txt`", type="txt", key="up_train")
test_file  = c2.file_uploader("`test_FD001.txt`",  type="txt", key="up_test")
rul_file   = c3.file_uploader("`RUL_FD001.txt`",   type="txt", key="up_rul")

if train_file and test_file and rul_file:
    data = load_and_preprocess(train_file, test_file, rul_file)
    train_df = data['train']
    test_df = data['test_scaled']
    feat_cols = data['feat_cols']
    const_cols = data['const_cols']
    X_test_last = data['X_test_last_scaled']
    y_test_last = data['y_test_last']
    scaler = data['scaler']

    # Safety alignment: Ensures the app doesn't crash if lengths are mismatched due to stale cache
    if len(X_test_last) != len(y_test_last):
        min_len = min(len(X_test_last), len(y_test_last))
        X_test_last = X_test_last[:min_len]
        y_test_last = y_test_last[:min_len]

    # ------------------------------------------------------------------
    # Dataset Overview + VISUALIZATIONS
    # ------------------------------------------------------------------
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Rows", f"{len(train_df):,}")
    c2.metric("Test Rows", f"{len(test_df):,}")
    c3.metric("Features", len(feat_cols))
    c4.metric("Dropped Constants", len(const_cols))

    with st.expander("Class Balance & RUL Distribution", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            train_df['label'].value_counts().plot(kind='bar', ax=ax, color=['green','red'])
            ax.set_title("Label"); ax.set_xticklabels(['Healthy','Fail'], rotation=0)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            train_df['RUL'].hist(bins=50, ax=ax, color='skyblue', alpha=0.7)
            ax.axvline(30, color='red', ls='--', label='Threshold')
            ax.set_title("RUL"); ax.legend()
            st.pyplot(fig)

    with st.expander("Sensor Degradation (Sample Engine)"):
        unit = train_df['unit'].iloc[0]
        sample_data = train_df[train_df['unit']==unit]
        sensors = [c for c in feat_cols if c.startswith('sensor_')][:6]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs = axs.flatten()
        for i, s in enumerate(sensors):
            axs[i].plot(sample_data['cycle'], sample_data[s], color='teal')
            axs[i].set_title(s)
            axs[i].axvline(sample_data[sample_data['RUL']<=30]['cycle'].min(), color='red', ls='--')
        plt.tight_layout()
        st.pyplot(fig)

    with st.expander("Feature Correlations (Heatmap)"):
        st.markdown("Inter-sensor correlations reveal redundancy and co-integration in engine degradation.")
        corr = train_df[feat_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, annot=False, cbar=True)
        ax.set_title("Sensor Correlation Matrix")
        st.pyplot(fig)

    with st.expander("Boxplot & Outlier Analysis (Top 6 Sensors)", expanded=False):
        top_sensors = [c for c in feat_cols if c.startswith('sensor_')][:6]
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = axs.flatten()
        outlier_counts = {}

        for i, s in enumerate(top_sensors):
            sns.boxplot(data=train_df, x='label', y=s, ax=axs[i], palette=['lightblue','orange'])
            axs[i].set_title(f"{s} – Boxplot")
            axs[i].set_xticklabels(['Healthy','Fail'])

            Q1 = train_df[s].quantile(0.25)
            Q3 = train_df[s].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = train_df[(train_df[s] < lower) | (train_df[s] > upper)]
            outlier_counts[s] = len(outliers)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("**Outlier Count per Sensor (IQR Method)**")
        outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=["Sensor", "Outliers"]).sort_values("Outliers", ascending=False)
        st.dataframe(outlier_df.style.bar(subset=["Outliers"], color="salmon"), width="stretch")

    # ------------------------------------------------------------------
    # Evaluation Method
    # ------------------------------------------------------------------
    st.header("2. Evaluation Method")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    eval_method = st.radio("Choose", ["Hold-out", "K-Fold CV"], horizontal=True, key="eval_method")

    X = train_df[feat_cols].values
    y = train_df["label"].values
    models = get_models()

    if eval_method == "Hold-out":
        test_size = st.slider("Validation %", 10, 40, 20, 5, key="holdout_split") / 100
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

        if st.button("Train & Evaluate (Hold-out)", type="primary"):
            for name, model in models.items():
                model.fit(Xtr, ytr)
            
            results_df = Evaluator.evaluate_all(models, Xval, yval)
            best_name = results_df.iloc[0]["Model"]
            best_model = models[best_name]

            y_test_pred = best_model.predict(X_test_last)
            test_acc = accuracy_score(y_test_last, y_test_pred)
            test_f1 = f1_score(y_test_last, y_test_pred)

            st.session_state.update({
                "results_df": results_df,
                "best_model": best_model,
                "best_name": best_name,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "train_df": train_df,
                "feat_cols": feat_cols
            })
            st.success("Hold-out evaluation complete!")

    else:  # K-Fold
        k = st.slider("K (folds)", 3, 10, 5, key="kfold_k")
        if st.button("Run K-Fold CV", type="primary"):
            with st.spinner("Running K-Fold..."):
                df_k = Evaluator.evaluate_kfold(models, X, y, k=k)
                best_name = df_k.iloc[0]["Model"]
                best_model = models[best_name]
                best_model.fit(X, y)

                y_test_pred = best_model.predict(X_test_last)
                test_acc = accuracy_score(y_test_last, y_test_pred)
                test_f1 = f1_score(y_test_last, y_test_pred)

                st.session_state.update({
                    "kfold_df": df_k,
                    "best_model": best_model,
                    "best_name": best_name,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "train_df": train_df,
                    "feat_cols": feat_cols
                })
                st.success("K-Fold complete!")

    # ------------------------------------------------------------------
    # Results & XAI
    # ------------------------------------------------------------------
    if "best_model" in st.session_state:
        st.header("3. Results")
        if "results_df" in st.session_state:
            st.dataframe(st.session_state.results_df.style.highlight_max(axis=0), width="stretch")
        else:
            st.dataframe(st.session_state.kfold_df, width="stretch")

        st.metric("Best Model", st.session_state.best_name)
        st.metric("Test Acc (last cycle)", f"{st.session_state.test_acc:.3f}")
        st.metric("Test F1 (last cycle)", f"{st.session_state.test_f1:.3f}")

        st.header("4. Predict & Explain")
        src = st.radio("Input", ["Test Last Cycle", "Custom Input"], horizontal=True, key="pred_src")

        if src == "Custom Input":
            st.info("Enter scaled values [0–1]:")
            vals = {}
            l, r = st.columns(2)
            for i, f in enumerate(st.session_state.feat_cols):
                col = l if i % 2 == 0 else r
                vals[f] = col.number_input(f"`{f}`", 0.0, 1.0, 0.5, 0.01, format="%.4f", key=f"inp_{f}")
            instance = np.array([list(vals.values())])
        else:
            last_df = test_df.groupby("unit").last().reset_index()
            unit = st.selectbox("Unit", sorted(last_df["unit"].unique()), key="sel_unit")
            instance = last_df[last_df["unit"]==unit][st.session_state.feat_cols].values

        if st.button("Predict", type="secondary"):
            model = st.session_state.best_model
            pred = model.predict(instance)[0]
            prob = model.predict_proba(instance)[0,1]
            verdict = "FAILURE" if pred else "HEALTHY"
            st.markdown(f"### {verdict}")
            st.progress(prob)
            st.caption(f"Probability: **{prob:.3f}**")

            # SHAP
            if HAS_SHAP:
                with st.expander("SHAP Explanation", expanded=True):
                    try:
                        bg_data = st.session_state.train_df[st.session_state.feat_cols].values
                        bg_summary = shap.kmeans(bg_data, 50)
                        explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:,1], bg_summary)
                        sv = explainer.shap_values(instance, nsamples=100)
                        sv = sv[0] if isinstance(sv, list) else sv

                        df_shap = pd.DataFrame({"Feature": st.session_state.feat_cols, "SHAP": sv.flatten()})
                        df_shap["Abs"] = df_shap["SHAP"].abs()
                        top = df_shap.sort_values("Abs", ascending=False).head(10)

                        fig, ax = plt.subplots(figsize=(8,5))
                        colors = ['red' if v<0 else 'green' for v in top["SHAP"]]
                        ax.barh(top["Feature"], top["SHAP"], color=colors)
                        ax.axvline(0, color='k', lw=0.5)
                        ax.set_title("SHAP Contributions")
                        st.pyplot(fig)

                        st.markdown("**🔍 Explaining the SHAP Insights:**")
                        for _, r in top.iterrows():
                            direction = "increases" if r["SHAP"] > 0 else "decreases"
                            severity = "Critical" if abs(r["SHAP"]) > 0.1 else "Moderate"
                            impact = f"**{direction}** the risk of **FAILURE**"
                            st.markdown(f"- `{r['Feature']}`: This sensor reading {impact} (Impact: {abs(r['SHAP']):.3f}, {severity}).")

                    except Exception as e:
                        st.error(f"SHAP: {e}")

            # LIME
            if HAS_LIME:
                with st.expander("LIME Explanation", expanded=True):
                    try:
                        bg_data = st.session_state.train_df[st.session_state.feat_cols].values
                        explainer = LimeTabularExplainer(bg_data, feature_names=st.session_state.feat_cols,
                                                         class_names=["Healthy","Fail"], mode="classification")
                        exp = explainer.explain_instance(instance.flatten(), model.predict_proba, num_features=10)
                        st.pyplot(exp.as_pyplot_figure())

                        st.markdown("**🔍 Explaining the LIME Insights:**")
                        for f, w in exp.as_list():
                            direction = "failure" if w > 0 else "health"
                            st.markdown(f"- When `{f}`, the model's confidence in **{direction}** increases by **{abs(w*100):.1f}%**.")

                    except Exception as e:
                        st.error(f"LIME: {e}")
