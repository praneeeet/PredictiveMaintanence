import numpy as np
import pandas as pd
import math
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# ----------------------------
# Gaussian likelihood (continuous)
# ----------------------------
def gaussian_likelihood(x_train, x_test):
    mean = np.mean(x_train)
    std = np.std(x_train)
    if std == 0: std = 1e-6   # avoid divide by zero
    return (1 / (math.sqrt(2*math.pi)*std)) * math.exp(-((x_test-mean)**2)/(2*std**2))

# ----------------------------
# Discrete likelihood (categorical)
# ----------------------------
def discrete_likelihood(x_train, x_test):
    values, counts = np.unique(x_train, return_counts=True)
    prob_dict = {v:c/len(x_train) for v,c in zip(values, counts)}
    return prob_dict.get(x_test, 1e-6)  # smoothing

# ----------------------------
# Split samples by class
# ----------------------------
def separate_by_class(X, y):
    classes = {}
    for i in range(len(y)):
        classes.setdefault(y[i], []).append(X[i])
    return classes

# ----------------------------
# Confusion matrix (hardcoded)
# ----------------------------
def confusion_metrics(y_true, y_pred, positive_class=1):
    TP = FP = FN = TN = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == positive_class:
            if yp == positive_class: TP += 1
            else: FN += 1
        else:
            if yp == positive_class: FP += 1
            else: TN += 1
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP) if (TP+FP)!=0 else 0
    rec = TP/(TP+FN) if (TP+FN)!=0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)!=0 else 0
    return TP, FP, FN, TN, acc, prec, rec, f1

# ----------------------------
# Hardcoded ROC + AUC
# ----------------------------
def roc_curve_manual(y_true, y_scores, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_list, fpr_list = [], []

    for thresh in thresholds:
        y_pred = [1 if p >= thresh else 0 for p in y_scores]
        TP, FP, FN, TN, _, _, _, _ = confusion_metrics(y_true, y_pred)
        TPR = TP/(TP+FN) if (TP+FN)!=0 else 0
        FPR = FP/(FP+TN) if (FP+TN)!=0 else 0
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # AUC with trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i]-fpr_list[i-1]) * (tpr_list[i]+tpr_list[i-1]) / 2
    return fpr_list, tpr_list, auc

# ============================================================
# Load Cybersecurity Dataset
# ============================================================
df = pd.read_csv(r"C:\Users\gpran\Desktop\cybersecurity_intrusion_data.csv")

# Drop ID column if present
if "session_id" in df.columns:
    df = df.drop("session_id", axis=1)

# Separate features & labels
X = df.drop("attack_detected", axis=1)
y = df["attack_detected"].values

# Encode categorical features
feature_types = []
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])
        feature_types.append('discrete')
    else:
        feature_types.append('continuous')

X = X.values

# ============================================================
# Leave-One-Out Naive Bayes
# ============================================================
loo = LeaveOneOut()
y_true, y_pred, y_probs = [], [], []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Separate by class
    classes = separate_by_class(X_train, y_train)
    priors = {c:len(samples)/len(y_train) for c,samples in classes.items()}

    # Posterior for each class
    posteriors = {}
    for c, samples in classes.items():
        prob = priors[c]
        samples = np.array(samples)
        for j in range(X.shape[1]):
            if feature_types[j] == 'continuous':
                prob *= gaussian_likelihood(samples[:,j], X_test[0][j])
            else:
                prob *= discrete_likelihood(samples[:,j], X_test[0][j])
        posteriors[c] = prob

    # Normalize to get probabilities
    total = sum(posteriors.values())
    probs = {c: posteriors[c]/total for c in posteriors}

    # Predict class
    pred_class = max(posteriors, key=posteriors.get)

    y_true.append(y_test[0])
    y_pred.append(pred_class)
    y_probs.append(probs.get(1, 0))   # probability of intrusion

# ============================================================
# Evaluation
# ============================================================
TP, FP, FN, TN, acc, prec, rec, f1 = confusion_metrics(y_true, y_pred)
print("Final Results (Leave-One-Out CV):")
print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")

# ROC + AUC
fpr, tpr, auc = roc_curve_manual(y_true, y_probs)
print(f"AUC = {auc:.4f}")

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC={auc:.2f})")
plt.plot([0,1],[0,1],"r--", lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hardcoded Naive Bayes (Intrusion Detection)")
plt.legend(loc="lower right")
plt.show()
