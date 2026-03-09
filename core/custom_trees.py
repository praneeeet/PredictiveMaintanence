import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class DecisionTree(BaseEstimator, ClassifierMixin):
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

    def predict_proba(self, X):
        """Proxy probability: returns 1.0 for the predicted class and 0.0 otherwise."""
        y_pred = self.predict(X)
        p1 = y_pred.astype(float)
        return np.vstack([1-p1, p1]).T

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

class RandomForest(BaseEstimator, ClassifierMixin):
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
            for j, f_orig in enumerate(feats):
                self.feature_importances_[f_orig] += tree.feature_importances_[j]

        s = self.feature_importances_.sum()
        if s > 0: self.feature_importances_ /= s
        return self

    def predict(self, X):
        votes = []
        for tree, feats in zip(self.trees, self.feat_indices):
            votes.append(tree.predict(X[:, feats]))
        votes = np.stack(votes, axis=1)
        return np.apply_along_axis(lambda row: np.bincount(row, minlength=2).argmax(), axis=1, arr=votes).astype(int)

    def predict_proba(self, X):
        """Probability estimated by fraction of trees voting for the positive class."""
        votes = []
        for tree, feats in zip(self.trees, self.feat_indices):
            votes.append(tree.predict(X[:, feats]))
        votes = np.stack(votes, axis=1)
        p1 = np.mean(votes, axis=1)
        return np.vstack([1-p1, p1]).T

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))

class CatBoost(BaseEstimator, ClassifierMixin):
    """Hardcoded Gradient Boosting (CatBoost-like implementation)."""
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
        f = np.zeros(m, dtype=float)

        for _ in range(self.n_estimators):
            p = self._sigmoid(f)
            residuals = y - p
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
