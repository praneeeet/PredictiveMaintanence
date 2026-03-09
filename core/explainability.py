# explainability.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SHAP + LIME
import shap
from lime.lime_tabular import LimeTabularExplainer

def get_importance_df(model, feature_names, top_k=12):
    if getattr(model, "feature_importances_", None) is None:
        return pd.DataFrame({"feature": [], "importance": []})
    imps = model.feature_importances_.copy()
    idx = np.argsort(imps)[-top_k:][::-1]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in idx],
        "importance": imps[idx]
    })

# ---------- SHAP ----------
def _proba_function(model):
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:,1]
    elif hasattr(model, "predict_proba_simple"):
        return lambda X: model.predict_proba_simple(X)[:,1]
    else:
        # fall back to 0/1 predictions as "probability"
        return lambda X: model.predict(X).astype(float)

def shap_explain_instance(model, X_background, x_instance, feature_names, link="logit"):
    try:
        f = _proba_function(model)
        # Use KernelExplainer for generic models
        bg = shap.kmeans(X_background, k=min(50, len(X_background))) if len(X_background) > 60 else X_background
        explainer = shap.KernelExplainer(f, bg, link=link)
        shap_values = explainer.shap_values(x_instance, nsamples=200)
        # Waterfall plot
        shap.initjs()
        fig = plt.figure(figsize=(7,4))
        shap.plots._waterfall.waterfall_legacy(
            shap.Explanation(values=shap_values,
                             base_values=np.mean(f(bg)),
                             data=x_instance.flatten(),
                             feature_names=feature_names),
            max_display=12, show=False
        )
        plt.tight_layout()
        return fig
    except Exception as e:
        return None

# ---------- LIME ----------
def lime_explain_instance(model, X_train, x_instance, feature_names, class_names):
    try:
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=True,
            mode="classification"
        )
        f = None
        if hasattr(model, "predict_proba"):
            f = model.predict_proba
        elif hasattr(model, "predict_proba_simple"):
            f = model.predict_proba_simple
        else:
            # wrap predict() into a 2-col proba for LIME
            def _wrap(X):
                pred = model.predict(X).astype(int)
                p1 = pred.astype(float)
                return np.vstack([1 - p1, p1]).T
            f = _wrap

        exp = explainer.explain_instance(x_instance, f, num_features=10)
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(7, 4)
        fig.tight_layout()
        return fig
    except Exception:
        return None
