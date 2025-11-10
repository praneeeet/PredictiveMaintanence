# models.py — Non-tree models for Predictive Maintenance
# Logistic Regression, SVM, KNN, Naive Bayes

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def get_models():
    """Return a dictionary of classification models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }
    return models
