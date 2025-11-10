# app.py
# Streamlit Predictive Maintenance with ADWIN Drift Detection and Auto-Retraining
# Models: Hardcoded Decision Tree, Hardcoded Random Forest, Hardcoded Gradient Boosting (CatBoost-like)

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from river.drift import ADWIN

st.set_page_config(page_title="Predictive Maintenance + Drift Detection", layout="wide")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===============================
# Helpers: data loading & preprocessing
# ===============================
@st.cache_data(show_spinner=True)
def load_fd001(base="CMaps"):
    cols = ['unit','cycle'] + [f'op_{i}' for i in range(1,4)] + [f'sensor_{i}' for i in range(1,22)]
    train = pd.read_csv(os.path.join(base,'train_FD001.txt'), sep=r"\s+", header=None)
    test  = pd.read_csv(os.path.join(base,'test_FD001.txt'),  sep=r"\s+", header=None)
    rul   = pd.read_csv(os.path.join(base,'RUL_FD001.txt'),   sep=r"\s+", header=None)

    train.columns = cols
    test.columns  = cols
    rul.columns   = ['RUL']

    # compute RUL & binary label for TRAIN
    max_cycles = train.groupby('unit')['cycle'].max().reset_index(name='max_cycle')
    train = train.merge(max_cycles, on='unit', how='left')
    train['RUL']   = train['max_cycle'] - train['cycle']
    train['label'] = (train['RUL'] <= 30).astype(int)

    # Drop constants (across training features)
    feat_cols = [c for c in train.columns if c not in ['unit','cycle','RUL','label','max_cycle']]
    constant_cols = [c for c in feat_cols if train[c].std() == 0]
    if constant_cols:
        train = train.drop(columns=constant_cols)
        test  = test.drop(columns=constant_cols)

    # Scale features (fit on train)
    feat_cols = [c for c in train.columns if c not in ['unit','cycle','RUL','label','max_cycle']]
    scaler = MinMaxScaler()
    train[feat_cols] = scaler.fit_transform(train[feat_cols])

    # Prepare a scaled test time-series too (to stream later)
    test_scaled = test.copy()
    test_scaled[feat_cols] = scaler.transform(test_scaled[feat_cols])

    # Prepare last-cycle targets for official test evaluation (not used in stream)
    test_last = test.groupby('unit').last().reset_index()
    y_test_bin = (rul['RUL'] <= 30).astype(int).values
    X_test_last = test_last.drop(columns=['unit','cycle'])
    X_test_last_scaled = scaler.transform(X_test_last[feat_cols])

    return {
        "train": train,
        "test": test,
        "test_scaled": test_scaled,
        "rul": rul,
        "feat_cols": feat_cols,
        "scaler": scaler,
        "constant_cols": constant_cols,
        "X_test_last_scaled": X_test_last_scaled,
        "y_test_last": y_test_bin,
    }

# ===============================
# Hardcoded Models (DT, RF, GB)
# ===============================
class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion 
        self.tree = None
        self.feature_importances_ = None

    def _gini(self, y):
        y = y.astype(int)
        p = np.bincount(y, minlength=2) / len(y)
        return 1 - np.sum(p**2)

    def _mse(self, y):
        return np.var(y) * len(y)
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None, 0.0
        best_gain, best_f, best_thr = 0.0, None, None
        parent_imp = self._gini(y) if self.criterion == 'gini' else self._mse(y)
        # modest percentiles for speed and stability
        for f in range(n):
            thresholds = np.percentile(X[:, f], np.linspace(10, 90, 10))
            for thr in np.unique(thresholds):
                left = X[:, f] <= thr
                right = ~left
                if left.sum() < self.min_samples_split or right.sum() < self.min_samples_split:
                    continue
                if self.criterion == 'gini':
                    imp_left, imp_right = self._gini(y[left]), self._gini(y[right])
                else:
                    imp_left, imp_right = self._mse(y[left]), self._mse(y[right])
                weighted = (left.sum()*imp_left + right.sum()*imp_right)/m
                gain = parent_imp - weighted
                if gain > best_gain:
                    best_gain, best_f, best_thr = gain, f, thr
        return best_f, best_thr, best_gain

    def _fit(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            if self.criterion == 'gini':
                return {'leaf': True, 'value': int(np.bincount(y.astype(int)).argmax())}
            else:
                return {'leaf': True, 'value': float(np.mean(y))}
        f, thr, gain = self._best_split(X, y)
        if f is None:
            if self.criterion == 'gini':
                return {'leaf': True, 'value': int(np.bincount(y.astype(int)).argmax())}
            else:
                return {'leaf': True, 'value': float(np.mean(y))}
        left = X[:, f] <= thr
        right = ~left
        self.feature_importances_[f] += gain * len(y)
        return {
            'leaf': False, 'feature': f, 'threshold': thr,
            'left': self._fit(X[left], y[left], depth+1),
            'right': self._fit(X[right], y[right], depth+1)
        }

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.tree = self._fit(X, y, 0)
        s = self.feature_importances_.sum()
        if s > 0: self.feature_importances_ /= s
        return self

    def _predict_one(self, x, node):
        if node['leaf']: return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        out = np.array([self._predict_one(x, self.tree) for x in X])
        return out.astype(int) if self.criterion == 'gini' else out

    def score(self, X, y):  # for sklearn-like compatibility
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

class RandomForest:
    def __init__(self, n_trees=25, max_depth=6, min_samples_split=5, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.feat_indices = []
        self.feature_importances_ = None

    def fit(self, X, y):
        m, n = X.shape
        self.feature_importances_ = np.zeros(n, dtype=float)
        self.trees, self.feat_indices = [], []

        max_feat = int(np.sqrt(n)) if self.max_features == 'sqrt' else n
        for _ in range(self.n_trees):
            idx = np.random.choice(m, m, replace=True)
            feats = np.random.choice(n, max_feat, replace=False)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, criterion='gini')
            tree.fit(X[idx][:, feats], y[idx])
            self.trees.append(tree)
            self.feat_indices.append(feats)
            # accumulate importances back to original positions
            for j, f_orig in enumerate(feats):
                self.feature_importances_[f_orig] += tree.feature_importances_[j]

        s = self.feature_importances_.sum()
        if s > 0: self.feature_importances_ /= s
        return self

    def predict(self, X):
        votes = []
        for tree, feats in zip(self.trees, self.feat_indices):
            votes.append(tree.predict(X[:, feats]))
        votes = np.stack(votes, axis=1)  # (n, n_trees)
        return np.apply_along_axis(lambda row: np.bincount(row, minlength=2).argmax(), axis=1, arr=votes).astype(int)

    def predict_proba_simple(self, X):
        # average of class predictions (hard vote → probability proxy)
        votes = []
        for tree, feats in zip(self.trees, self.feat_indices):
            votes.append(tree.predict(X[:, feats]))
        votes = np.stack(votes, axis=1)  # (n, trees)
        p1 = np.mean(votes, axis=1)  # fraction voting class 1
        return np.vstack([1-p1, p1]).T

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

class HardcodedGradientBoosting:
    """CatBoost-like logistic gradient boosting using our DecisionTree as weak learner."""
    def __init__(self, n_estimators=60, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.feature_importances_ = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -20, 20)
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.trees = []
        self.feature_importances_ = np.zeros(n, dtype=float)
        f = np.zeros(m, dtype=float)  # initial log-odds = 0 (p=0.5)

        for _ in range(self.n_estimators):
            p = self._sigmoid(f)
            residuals = y - p  # negative gradient of logloss
            # Fit a regression tree to residuals (use 'mse' in DecisionTree)
            reg_tree = DecisionTree(max_depth=self.max_depth, min_samples_split=2, criterion='mse').fit(X, residuals)
            update = reg_tree.predict(X)
            f += self.learning_rate * update

            self.trees.append(reg_tree)
            self.feature_importances_ += reg_tree.feature_importances_

        s = self.feature_importances_.sum()
        if s > 0: self.feature_importances_ /= s
        return self

    def predict_proba(self, X):
        f = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            f += self.learning_rate * tree.predict(X)
        p = self._sigmoid(f)
        return np.vstack([1-p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

# ===============================
# Training / evaluation utils
# ===============================
def evaluate_all(models, X, y):
    rows = []
    for name, m in models.items():
        y_pred = m.predict(X)
        # probability when available
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X)[:,1]
        elif hasattr(m, "predict_proba_simple"):
            proba = m.predict_proba_simple(X)[:,1]
        else:
            proba = None
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1": f1_score(y, y_pred, zero_division=0)
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False)

def top_features(importances, names, k=10):
    idx = np.argsort(importances)[-k:][::-1]
    return [names[i] for i in idx], importances[idx]

def plot_importance(importances, feat_names, title="Top Features"):
    k = min(12, len(feat_names))
    names, vals = top_features(importances, feat_names, k=k)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(names[::-1], vals[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

# ===============================
# Load data
# ===============================
st.header("📦 Load & Preprocess NASA CMAPSS (FD001)")
data = load_fd001()
train = data["train"]
feat_cols = data["feat_cols"]

c1, c2, c3 = st.columns(3)
c1.metric("Train rows", f"{len(train):,}")
c2.metric("Features", f"{len(feat_cols)}")
c3.metric("Dropped constants", f"{len(data['constant_cols'])}")

st.caption("Labels: 1 → will fail within 30 cycles; 0 → healthy.")

# ===============================
# Split: 1/4 initial training, remaining as stream
# ===============================
st.header("🧪 Initial Training (¼ of data) + Streaming")
train_fraction = st.sidebar.slider("Initial training fraction", 0.1, 0.5, 0.25, 0.05)
batch_size = st.sidebar.slider("Stream batch size", 100, 2000, 500, 50)
retrain_trigger = st.sidebar.slider("Drifted features to trigger retrain", 1, 10, 5, 1)
acc_trigger = st.sidebar.slider("Min batch accuracy (retrain if below)", 0.50, 0.95, 0.80, 0.01)
adwin_delta = st.sidebar.selectbox("ADWIN delta (smaller = stricter)", [0.001, 0.002, 0.005, 0.01], index=1)

# Shuffle for better mixing across units
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
with st.spinner("Training initial models on ¼ data..."):
    dt = DecisionTree(max_depth=4, min_samples_split=5, criterion='gini').fit(X0, y0)
    rf = RandomForest(n_trees=25, max_depth=6, min_samples_split=5).fit(X0, y0)
    gb = HardcodedGradientBoosting(n_estimators=60, learning_rate=0.1, max_depth=3).fit(X0, y0)

models = {"DT (hardcoded)": dt, "RF (hardcoded)": rf, "GB (CatBoost-like, hardcoded)": gb}

st.subheader("📊 Initial Model Performance (on held-out official test last-cycles)")
init_eval = evaluate_all(models, data["X_test_last_scaled"], data["y_test_last"])
st.dataframe(init_eval, use_container_width=True)

# ===============================
# Drift detection + streaming loop
# ===============================
st.header("🔁 Real-time Screening with ADWIN Drift Detection")
if "adwins" not in st.session_state:
    st.session_state.adwins = [ADWIN(delta=adwin_delta) for _ in range(len(feat_cols))]
if "X_train_aug" not in st.session_state:
    st.session_state.X_train_aug = X0.copy()
    st.session_state.y_train_aug = y0.copy()
if "rf" not in st.session_state:
    st.session_state.rf = rf
    st.session_state.dt = dt
    st.session_state.gb = gb
if "logs" not in st.session_state:
    st.session_state.logs = []
if "pos" not in st.session_state:
    st.session_state.pos = 0
if "retrain_count" not in st.session_state:
    st.session_state.retrain_count = 0

c_a, c_b, c_c = st.columns([1,1,2])
if c_a.button("▶️ Stream next batch"):
    start = st.session_state.pos
    end = min(start + batch_size, len(Xs))
    if start >= end:
        st.info("✅ Streaming completed.")
    else:
        Xb, yb = Xs[start:end], ys[start:end]
        # Drift per feature (feed all values in batch)
        drift_flags = []
        for j, adw in enumerate(st.session_state.adwins):
            drift_happened = False
            for v in Xb[:, j]:
                if adw.update(float(v)):
                    drift_happened = True
            drift_flags.append(drift_happened)
        drifted_feats = [feat_cols[i] for i,f in enumerate(drift_flags) if f]

        # Evaluate current RF on this batch
        y_pred_b = st.session_state.rf.predict(Xb)
        acc_b = accuracy_score(yb, y_pred_b)
        prec_b = precision_score(yb, y_pred_b, zero_division=0)
        rec_b = recall_score(yb, y_pred_b, zero_division=0)
        f1_b = f1_score(yb, y_pred_b, zero_division=0)

        st.session_state.logs.append({
            "from": start, "to": end, "n": len(Xb),
            "drifted": len(drifted_feats),
            "drift_feats": drifted_feats[:12],
            "acc": acc_b, "prec": prec_b, "rec": rec_b, "f1": f1_b
        })

        # Retrain rule
        retrain = (len(drifted_feats) >= retrain_trigger) or (acc_b < acc_trigger and len(Xb) >= 20)
        if retrain:
            st.warning("⚠️ Drift detected and/or performance drop — retraining on augmented data...")
            # augment training buffer with this labeled batch
            st.session_state.X_train_aug = np.vstack([st.session_state.X_train_aug, Xb])
            st.session_state.y_train_aug = np.concatenate([st.session_state.y_train_aug, yb])

            # Refit all three models (subsample if huge for speed)
            sub_n = min(20000, len(st.session_state.X_train_aug))
            sub_idx = np.random.choice(len(st.session_state.X_train_aug), sub_n, replace=False)
            Xsub = st.session_state.X_train_aug[sub_idx]
            ysub = st.session_state.y_train_aug[sub_idx]

            st.session_state.dt = DecisionTree(max_depth=4, min_samples_split=5, criterion='gini').fit(Xsub, ysub)
            st.session_state.rf = RandomForest(n_trees=25, max_depth=6, min_samples_split=5).fit(Xsub, ysub)
            st.session_state.gb = HardcodedGradientBoosting(n_estimators=60, learning_rate=0.1, max_depth=3).fit(Xsub, ysub)
            st.session_state.retrain_count += 1

        st.session_state.pos = end

if c_b.button("🔄 Reset stream"):
    st.session_state.adwins = [ADWIN(delta=adwin_delta) for _ in range(len(feat_cols))]
    st.session_state.X_train_aug = X0.copy()
    st.session_state.y_train_aug = y0.copy()
    st.session_state.rf = rf
    st.session_state.dt = dt
    st.session_state.gb = gb
    st.session_state.logs = []
    st.session_state.pos = 0
    st.session_state.retrain_count = 0
    st.success("Stream and models reset.")

# Show logs table
if len(st.session_state.logs):
    st.subheader("📜 Streaming Batches Log")
    st.dataframe(pd.DataFrame(st.session_state.logs), use_container_width=True)

    # Plot batch F1 over time
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot([row["f1"] for row in st.session_state.logs])
    ax.set_title("Batch F1 over Stream")
    ax.set_xlabel("Batch #")
    ax.set_ylabel("F1")
    st.pyplot(fig)

    st.info(f"Total retrains so far: {st.session_state.retrain_count}")

# ===============================
# Explainability: feature importances
# ===============================
st.header("🧠 Explainability – Most Contributing Features")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Decision Tree (current)**")
    plot_importance(st.session_state.dt.feature_importances_, feat_cols, "DT Importances")
with col2:
    st.markdown("**Random Forest (current)**")
    plot_importance(st.session_state.rf.feature_importances_, feat_cols, "RF Importances")
with col3:
    st.markdown("**Gradient Boosting (current)**")
    plot_importance(st.session_state.gb.feature_importances_, feat_cols, "GB (CatBoost-like) Importances")

# Top features list
def toprepr(name, model):
    names, vals = top_features(model.feature_importances_, feat_cols, k=8)
    return f"**{name}:** " + ", ".join([f"{n} ({v:.3f})" for n, v in zip(names, vals)])

st.markdown(toprepr("RF top sensors", st.session_state.rf))
st.markdown(toprepr("GB top sensors", st.session_state.gb))



