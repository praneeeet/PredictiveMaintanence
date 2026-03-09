import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

class Evaluator:
    @staticmethod
    def evaluate_all(models, X, y):
        rows = []
        for name, m in models.items():
            y_pred = m.predict(X).astype(int)
            if hasattr(m, "predict_proba"):
                try:
                    proba = m.predict_proba(X)[:, 1]
                except (IndexError, ValueError):
                    proba = y_pred.astype(float)
            else:
                proba = y_pred.astype(float)
            
            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else np.nan
            except ValueError:
                auc = np.nan
                
            rows.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC-AUC": auc,
                "Score_Combined": (rec + auc + f1) / 3 if not np.isnan(auc) else (rec + f1) / 2
            })
            
        return pd.DataFrame(rows).sort_values("Score_Combined", ascending=False)
    
    @staticmethod
    def evaluate_kfold(models, X, y, k=5, random_state=42):
        from sklearn.model_selection import cross_validate
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        scoring = {
            'Accuracy': 'accuracy',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1': 'f1',
            'ROC-AUC': 'roc_auc'
        }
        
        all_results = []
        for name, model in models.items():
            cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring)
            
            res = {"Model": name}
            for metric in scoring.keys():
                mean_score = np.mean(cv_results[f'test_{metric}'])
                res[f"{metric} (mean)"] = mean_score
            res["Score_Combined"] = (res["Recall (mean)"] + res["ROC-AUC (mean)"] + res["F1 (mean)"]) / 3
            all_results.append(res)
            
        return pd.DataFrame(all_results).sort_values("Score_Combined", ascending=False)

    @staticmethod
    def permutation_importance(model, X, y, n_repeats=5, random_state=42):
        base_f1 = f1_score(y, model.predict(X), zero_division=0)
        rng = np.random.default_rng(random_state)
        imps = np.zeros(X.shape[1])
        
        for j in range(X.shape[1]):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                rng.shuffle(Xp[:, j])
                drop = max(0, base_f1 - f1_score(y, model.predict(Xp), zero_division=0))
                drops.append(drop)
            imps[j] = np.mean(drops)
        return imps

    @staticmethod
    def get_top_features(importances, names, k=10):
        idx = np.argsort(importances)[-k:][::-1]
        return [names[i] for i in idx], importances[idx]
