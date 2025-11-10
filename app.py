# app.py — CMAPSS FD001 Predictive Maintenance
# Full Preprocessing | Rich Visualizations | Hold-out + K-Fold | SHAP + LIME | ADWIN
# --------------------------------------------------------------

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
plt.style.use("default")

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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

# Drift
try:
    from river.drift import ADWIN
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    class ADWIN:
        def __init__(self, delta=0.002): pass
        def update(self, x): return False

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
st.set_page_config(page_title="CMAPSS FD001 Predictive Maintenance", layout="wide")
st.title("CMAPSS FD001 – Predictive Maintenance")
st.caption("Logistic • SVM • KNN • Naive Bayes | SHAP + LIME | ADWIN | K-Fold + Hold-out")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------
# Dataset Info
# ----------------------------------------------------------------------
with st.expander("About CMAPSS FD001 Dataset", expanded=True):
    st.markdown("""
    **NASA Turbofan Engine Degradation (FD001)**  
    - 100 engines (train + test)  
    - Single operating condition, single fault mode  
    - **Label**: `1` if **RUL ≤ 30**, else `0`  
    - **Features**: 3 settings + 21 sensors → **drop constants** → scale [0,1]
    """)

# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Preprocessing data...")
def load_and_preprocess(train_file, test_file, rul_file):
    cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

    def read_file(f):
        if isinstance(f, (str, os.PathLike)):
            return pd.read_csv(f, sep=r"\s+", header=None)
        return pd.read_csv(io.BytesIO(f.read()), sep=r"\s+", header=None)

    train = read_file(train_file); train.columns = cols
    test  = read_file(test_file);  test.columns  = cols
    rul   = read_file(rul_file);   rul.columns   = ['RUL']

    # RUL & label
    max_cycle = train.groupby('unit')['cycle'].max().reset_index(name='max_cycle')
    train = train.merge(max_cycle, on='unit')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train['label'] = (train['RUL'] <= 30).astype(int)

    # Drop constants
    feat_cols = [c for c in train.columns if c not in ['unit','cycle','RUL','label','max_cycle']]
    const_cols = [c for c in feat_cols if train[c].std() == 0]
    train = train.drop(columns=const_cols)
    test  = test.drop(columns=const_cols)
    feat_cols = [c for c in feat_cols if c not in const_cols]

    # Scale
    scaler = MinMaxScaler()
    train[feat_cols] = scaler.fit_transform(train[feat_cols])
    test[feat_cols]  = scaler.transform(test[feat_cols])

    # Test last cycle
    test_last = test.groupby('unit').last().reset_index()
    X_test_last = test_last[feat_cols].values
    y_test_last = (rul['RUL'].values <= 30).astype(int)

    return train, test, feat_cols, scaler, X_test_last, y_test_last, const_cols

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM":                 SVC(probability=True, random_state=RANDOM_STATE),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB()
    }

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_holdout(models, Xtr, ytr, Xval, yval):
    results = []
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xval)
        prob = model.predict_proba(Xval)[:,1] if hasattr(model, 'predict_proba') else pred
        results.append({
            "Model": name,
            "Acc": accuracy_score(yval, pred),
            "Prec": precision_score(yval, pred, zero_division=0),
            "Rec": recall_score(yval, pred, zero_division=0),
            "F1": f1_score(yval, pred, zero_division=0),
            "AUC": roc_auc_score(yval, prob) if len(np.unique(prob)) > 1 else np.nan
        })
    return pd.DataFrame(results)

def evaluate_kfold(X, y, k=5):
    models = get_models()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
        results[name] = scores.mean()
    return results

# ----------------------------------------------------------------------
# Permutation Importance
# ----------------------------------------------------------------------
def permutation_importance(model, X, y, n_repeats=5):
    base = f1_score(y, model.predict(X), zero_division=0)
    rng = np.random.default_rng(RANDOM_STATE)
    imps = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(max(0, base - f1_score(y, model.predict(Xp), zero_division=0)))
        imps[j] = np.mean(drops)
    return imps

# ----------------------------------------------------------------------
# UI: Upload
# ----------------------------------------------------------------------
st.header("1. Upload Files")
c1, c2, c3 = st.columns(3)
train_file = c1.file_uploader("`train_FD001.txt`", type="txt", key="up_train")
test_file  = c2.file_uploader("`test_FD001.txt`",  type="txt", key="up_test")
rul_file   = c3.file_uploader("`RUL_FD001.txt`",   type="txt", key="up_rul")

if train_file and test_file and rul_file:
    train_df, test_df, feat_cols, scaler, X_test_last, y_test_last, const_cols = \
        load_and_preprocess(train_file, test_file, rul_file)

    # ------------------------------------------------------------------
    # Dataset Overview + RICH VISUALIZATIONS
    # ------------------------------------------------------------------
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train Rows", f"{len(train_df):,}")
    c2.metric("Test Rows", f"{len(test_df):,}")
    c3.metric("Features", len(feat_cols))
    c4.metric("Dropped Constants", len(const_cols))

    # 1. Class Balance & RUL
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



    # 3. Sensor Degradation (Top 6)
    with st.expander("Sensor Degradation (Sample Engine)"):
        unit = train_df['unit'].iloc[0]
        data = train_df[train_df['unit']==unit]
        sensors = [c for c in feat_cols if c.startswith('sensor_')][:6]
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs = axs.flatten()
        for i, s in enumerate(sensors):
            axs[i].plot(data['cycle'], data[s], color='teal')
            axs[i].set_title(s)
            axs[i].axvline(data[data['RUL']<=30]['cycle'].min(), color='red', ls='--')
        plt.tight_layout()
        st.pyplot(fig)

    # 4. Correlation Heatmap
    with st.expander("Feature Correlation"):
        corr = train_df[feat_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={'shrink': 0.7})
        ax.set_title("Feature Correlation")
        st.pyplot(fig)



    # ------------------------------------------------------------------
    # NEW: Boxplot + Outliers
    # ------------------------------------------------------------------
    with st.expander("Boxplot & Outlier Analysis (Top 6 Sensors)", expanded=False):
        top_sensors = [c for c in feat_cols if c.startswith('sensor_')][:6]
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = axs.flatten()
        outlier_counts = {}

        for i, s in enumerate(top_sensors):
            # Boxplot
            sns.boxplot(data=train_df, x='label', y=s, ax=axs[i], palette=['lightblue','orange'])
            axs[i].set_title(f"{s} – Boxplot")
            axs[i].set_xticklabels(['Healthy','Fail'])

            # Outlier detection
            Q1 = train_df[s].quantile(0.25)
            Q3 = train_df[s].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = train_df[(train_df[s] < lower) | (train_df[s] > upper)]
            outlier_counts[s] = len(outliers)

        plt.tight_layout()
        st.pyplot(fig)

        # Outlier Table
        st.markdown("**Outlier Count per Sensor (IQR Method)**")
        outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=["Sensor", "Outliers"])
        outlier_df = outlier_df.sort_values("Outliers", ascending=False)
        st.dataframe(outlier_df.style.bar(subset=["Outliers"], color="salmon"), use_container_width=True)

    # ------------------------------------------------------------------
    # 2. Evaluation Method
    # ------------------------------------------------------------------
    st.header("2. Evaluation Method")
    eval_method = st.radio("Choose", ["Hold-out", "K-Fold CV"], horizontal=True, key="eval_method")

    X = train_df[feat_cols].values
    y = train_df["label"].values

    if eval_method == "Hold-out":
        test_size = st.slider("Validation %", 10, 40, 20, 5, key="holdout_split") / 100
        Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

        if st.button("Train & Evaluate (Hold-out)", type="primary", key="train_holdout"):
            models = get_models()
            results_df = evaluate_holdout(models, Xtr, ytr, Xval, yval)
            best_name = results_df.sort_values("F1", ascending=False).iloc[0]["Model"]
            best_model = models[best_name]
            best_model.fit(Xtr, ytr)

            y_test_pred = best_model.predict(X_test_last)
            test_acc = accuracy_score(y_test_last, y_test_pred)
            test_f1 = f1_score(y_test_last, y_test_pred)

            st.session_state.update({
                "results_df": results_df,
                "best_model": best_model,
                "best_name": best_name,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "Xval": Xval, "yval": yval,
                "feat_cols": feat_cols, "scaler": scaler,
                "X_test_last": X_test_last, "y_test_last": y_test_last,
                "train_df": train_df, "models": models
            })
            st.success("Hold-out evaluation complete!")

    else:  # K-Fold
        k = st.slider("K (folds)", 3, 10, 5, key="kfold_k")
        if st.button("Run K-Fold CV", type="primary", key="run_kfold"):
            with st.spinner("Running K-Fold..."):
                kfold_results = evaluate_kfold(X, y, k)
                df_k = pd.DataFrame(list(kfold_results.items()), columns=["Model", "F1 (mean)"])
                best_name = df_k.sort_values("F1 (mean)", ascending=False).iloc[0]["Model"]
                best_model = get_models()[best_name]
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
                    "feat_cols": feat_cols, "scaler": scaler,
                    "X_test_last": X_test_last, "y_test_last": y_test_last,
                    "train_df": train_df
                })
                st.success("K-Fold complete!")

# ----------------------------------------------------------------------
# 3. Results
# ----------------------------------------------------------------------
if "results_df" in st.session_state or "kfold_df" in st.session_state:
    st.header("3. Results")
    if "results_df" in st.session_state:
        df = st.session_state.results_df
        st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
    else:
        df = st.session_state.kfold_df
        st.dataframe(df, use_container_width=True)

    st.metric("Best Model", st.session_state.best_name)
    st.metric("Test Acc (last cycle)", f"{st.session_state.test_acc:.3f}")
    st.metric("Test F1 (last cycle)", f"{st.session_state.test_f1:.3f}")

    st.subheader("Permutation Importance")
    X_imp = st.session_state.train_df[st.session_state.feat_cols].values[-2000:]
    y_imp = st.session_state.train_df["label"].values[-2000:]
    imp = permutation_importance(st.session_state.best_model, X_imp, y_imp)
    top = np.argsort(imp)[-10:]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(np.array(st.session_state.feat_cols)[top], imp[top], color="teal")
    ax.set_title("Top 10 Features")
    st.pyplot(fig)

    # ------------------------------------------------------------------
    # 4. Predict + XAI
    # ------------------------------------------------------------------
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

    if st.button("Predict", type="secondary", key="btn_predict"):
        model = st.session_state.best_model
        pred = model.predict(instance)[0]
        prob = model.predict_proba(instance)[0,1] if hasattr(model, 'predict_proba') else pred
        verdict = "FAILURE" if pred else "HEALTHY"
        st.markdown(f"### {verdict}")
        st.progress(prob)
        st.caption(f"Probability: **{prob:.3f}**")

        # SHAP
        if HAS_SHAP:
            with st.expander("SHAP Explanation", expanded=True):
                try:
                    bg = st.session_state.train_df[st.session_state.feat_cols].values
                    bg_summary = shap.kmeans(bg, 50)
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

                    st.markdown("**Text:**")
                    for _, r in top.iterrows():
                        dir_ = "increases" if r["SHAP"]>0 else "decreases"
                        impact = "**FAILURE**" if r["SHAP"]>0 else "**HEALTHY**"
                        st.markdown(f"- `{r['Feature']}` **{dir_}** risk by **{abs(r['SHAP']):.3f}** → {impact}")
                except Exception as e:
                    st.error(f"SHAP: {e}")

        # LIME
        if HAS_LIME:
            with st.expander("LIME Explanation", expanded=True):
                try:
                    bg = st.session_state.train_df[st.session_state.feat_cols].values
                    explainer = LimeTabularExplainer(bg, feature_names=st.session_state.feat_cols,
                                                     class_names=["Healthy","Fail"], mode="classification")
                    exp = explainer.explain_instance(instance.flatten(), model.predict_proba, num_features=10)
                    st.pyplot(exp.as_pyplot_figure())

                    st.markdown("**Text:**")
                    for f, w in exp.as_list():
                        dir_ = "increases" if w>0 else "decreases"
                        impact = "**FAILURE**" if w>0 else "**HEALTHY**"
                        st.markdown(f"- `{f}` **{dir_}** prob by **{abs(w):.3f}** → {impact}")
                except Exception as e:
                    st.error(f"LIME: {e}")

    # ------------------------------------------------------------------
    # 5. Drift
    # ------------------------------------------------------------------
    st.header("5. Drift Detection (ADWIN)")
    if not RIVER_AVAILABLE:
        st.warning("`pip install river`")
    else:
        bs = st.slider("Batch", 50, 500, 100, key="drift_bs")
        delta = st.selectbox("Delta", [0.001,0.002,0.005], key="drift_delta")
        thr = st.slider("Trigger", 1, 8, 3, key="drift_thr")

        if st.button("Scan", key="scan_drift"):
            adwins = [ADWIN(delta=delta) for _ in st.session_state.feat_cols]
            logs, retrain = [], 0
            for i in range(0, len(X_test_last), bs):
                batch = X_test_last[i:i+bs]
                drifted = []
                for j, adw in enumerate(adwins):
                    for val in batch[:, j]:
                        if adw.update(val):
                            drifted.append(st.session_state.feat_cols[j])
                            break
                drifted = list(set(drifted))
                logs.append({"Batch": i//bs+1, "Drifted": len(drifted)})
                if len(drifted) >= thr:
                    retrain += 1
            st.dataframe(pd.DataFrame(logs))
            st.write(f"**Retraining needed: {retrain} time(s)**" if retrain else "No drift")