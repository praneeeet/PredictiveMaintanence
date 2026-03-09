from core.custom_trees import DecisionTree, RandomForest, CatBoost

def get_models():
    """Return a dictionary of the user's hardcoded tree models."""
    return {
        "Decision Tree": DecisionTree(max_depth=5, min_samples_split=5),
        "Random Forest": RandomForest(n_trees=20, max_depth=6, min_samples_split=5),
        "CatBoost":      CatBoost(n_estimators=50, learning_rate=0.1, max_depth=3)
    }
