import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from models import DecisionTree, RandomForest, HardcodedGradientBoosting

# ---------------- Drift Detection (ADWIN) ----------------
try:
    from river.drift import ADWIN
    HAS_RIVER = True
except Exception:
    HAS_RIVER = False
    class ADWIN:
        def __init__(self, delta=0.002): pass
        def update(self, x): return False

def set_seed(seed=42):
    np.random.seed(seed)

# ---------------- CMAPSS Data Loading ----------------
def _read_space_txt(file):
    return pd.read_csv(file, sep=r"\s+", header=None)

def load_cmaps_from_uploads(train_file, test_file, rul_file):
    cols = ['unit','cycle'] + [f'op_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]

    train = _read_space_txt(train_file);  train.columns = cols
    test  = _read_space_txt(test_file);   test.columns  = cols
    rul   = _read_space_txt(rul_file);    rul.columns   = ['RUL']

    max_cycles = train.groupby('unit')['cycle'].max().reset_index(name='max_cycle')
    train = train.merge(max_cycles, on='unit', how='left')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train['label'] = (train['RUL'] <= 30).astype(int)

    feat_cols = [c for c in train.columns if c not in ['unit','cycle','RUL','label','max_cycle']]
    constant_cols = [c for c in feat_cols if train[c].std() == 0.0]
    if constant_cols:
        train = train.drop(columns=constant_cols)
        test  = test.drop(columns=constant_cols)
        feat_cols = [c for c in feat_cols if c not in constant_cols]

    scaler = MinMaxScaler()
    train[feat_cols] = scaler.fit_transform(train[feat_cols])
    test_scaled = test.copy()
    test_scaled[feat_cols] = scaler.transform(test_scaled[feat_cols])

    test_last = test.groupby('unit').last().reset_index()
    X_test_last_scaled = scaler.transform(test_last[feat_cols])
    y_test_last = (rul['RUL'] <= 30).astype(int).values

    return train, test, rul, feat_cols, scaler, constant_cols, X_test_last_scaled, y_test_last

# ---------------- Describe DF ----------------
def describe_dataframe(df: pd.DataFrame):
    st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    sensors = [c for c in df.columns if c.startswith("sensor_")]
    if sensors:
        fig, ax = plt.subplots(figsize=(8, 3))
        df[sensors].mean().sort_values().plot(kind="bar", ax=ax)
        ax.set_title("Mean per Sensor")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------- Model Training Helpers ----------------
def _rf_conf(rf, X):
    try:
        preds = np.array([t.predict(X[:, feats]) for t, feats in zip(rf.trees, rf.feat_indices)])
        return (preds == 1).mean(axis=0)
    except Exception:
        return np.zeros(X.shape[0])

def _gb_conf(gb, X):
    f = np.zeros(X.shape[0], dtype=float)
    for t in gb.models:
        f += gb.learning_rate * t.predict(X)
    f = np.clip(f, -20, 20)
    return 1/(1+np.exp(-f))

def train_three_models(Xtr, ytr):
    models = {
        "DecisionTreeManual": DecisionTree(max_depth=5, min_samples_split=5).fit(Xtr, ytr),
        "RandomForestManual": RandomForest(n_trees=20, max_depth=6, sample_size=0.8).fit(Xtr, ytr),
        "GradientBoostingManual": HardcodedGradientBoosting(n_estimators=60, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)
    }
    return models

def eval_models(models, Xte, yte):
    rows, rocs = [], {}
    for name, m in models.items():
        y_pred = m.predict(Xte).astype(int)
        if name == "RandomForestManual":
            probs = _rf_conf(m, Xte)
        elif name == "GradientBoostingManual":
            probs = _gb_conf(m, Xte)
        else:
            probs = y_pred.astype(float)

        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1 = f1_score(yte, y_pred, zero_division=0)
        auc = roc_auc_score(yte, probs) if np.unique(probs).size > 1 else np.nan
        rows.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc})

        try:
            fpr, tpr, _ = roc_curve(yte, probs)
            rocs[name] = (fpr, tpr, auc)
        except Exception:
            pass

    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    best = df.iloc[0]["Model"]
    cm = confusion_matrix(yte, (models[best].predict(Xte)).astype(int))
    return df, best, cm, rocs

def split_holdout_preprocess(train_df, feat_cols, test_size=0.2):
    X = train_df[feat_cols].values
    y = train_df["label"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    models = train_three_models(X_tr, y_tr)
    results_df, best_name, cm, rocs = eval_models(models, X_te, y_te)
    return results_df, best_name, cm, rocs, models, X_tr, X_te, y_tr, y_te

def kfold_train_evaluate(train_df, feat_cols, k=5):
    X = train_df[feat_cols].values
    y = train_df["label"].astype(int).values
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    agg_rows = {"DecisionTreeManual": [], "RandomForestManual": [], "GradientBoostingManual": []}
    cms, roc_curves = [], {"DecisionTreeManual": [], "RandomForestManual": [], "GradientBoostingManual": []}

    for tr, te in kf.split(X, y):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        models = train_three_models(X_tr, y_tr)
        df, _, cm, rocs = eval_models(models, X_te, y_te)
        cms.append(cm)
        for _, row in df.iterrows():
            agg_rows[row["Model"]].append([row["Accuracy"], row["Precision"], row["Recall"], row["F1"], row["ROC-AUC"]])
        for name, curve in rocs.items():
            roc_curves[name].append(curve)

    rows = []
    for name, vals in agg_rows.items():
        arr = np.array(vals)
        mean = np.nanmean(arr, axis=0)
        rows.append({"Model": name, "Accuracy": mean[0], "Precision": mean[1], "Recall": mean[2], "F1": mean[3], "ROC-AUC": mean[4]})
    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    best_name = results_df.iloc[0]["Model"]
    cm_avg = np.round(np.mean(np.stack(cms, axis=0), axis=0)).astype(int)
    return results_df, best_name, cm_avg, roc_curves

def select_best_model(results_df):
    return results_df.sort_values("F1", ascending=False).iloc[0]["Model"]

def fit_full_model_for_predict(train_df, feat_cols, best_name):
    X = train_df[feat_cols].values
    y = train_df["label"].astype(int).values
    if best_name == "DecisionTreeManual":
        model = DecisionTree(max_depth=5, min_samples_split=5).fit(X, y)
    elif best_name == "RandomForestManual":
        model = RandomForest(n_trees=20, max_depth=6, sample_size=0.8).fit(X, y)
    else:
        model = HardcodedGradientBoosting(n_estimators=60, learning_rate=0.1, max_depth=3).fit(X, y)
    return model, {}, None, feat_cols

# ---------------- Drift Simulation ----------------
def stream_with_adwin(model, model_name, X_stream, y_stream, batch_size, feat_names, adwin_delta=0.002, drift_trigger=5, acc_trigger=0.8):
    adwins = [ADWIN(delta=adwin_delta) for _ in range(X_stream.shape[1])]
    logs, pos, retrain_count = [], 0, 0
    while pos < len(X_stream):
        end = min(pos + batch_size, len(X_stream))
        Xb = X_stream[pos:end]
        drift_flags = []
        for j, adw in enumerate(adwins):
            drift_detected = False
            for v in Xb[:, j]:
                if adw.update(float(v)): drift_detected = True
            drift_flags.append(drift_detected)
        drifted_feats = [feat_names[i] for i, f in enumerate(drift_flags) if f]
        if y_stream is not None:
            yb = y_stream[pos:end]
            y_pred = model.predict(Xb).astype(int)
            acc = accuracy_score(yb, y_pred)
            f1 = f1_score(yb, y_pred, zero_division=0)
        else:
            acc, f1 = np.nan, np.nan
        logs.append({"from": pos, "to": end, "acc": acc, "f1": f1, "n_drifted": len(drifted_feats)})
        if len(drifted_feats) >= drift_trigger or (np.isfinite(acc) and acc < acc_trigger):
            retrain_count += 1
        pos = end
    return logs, retrain_count
