import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from river.drift import ADWIN
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    class ADWIN:
        def __init__(self, delta=0.002): pass
        def update(self, x): return False

class DriftDetector:
    def __init__(self, delta=0.002, num_features=1):
        self.delta = delta
        self.num_features = num_features
        self.adwins = [ADWIN(delta=self.delta) for _ in range(self.num_features)]

    def detect_drift_batch(self, X_batch):
        """Returns boolean array: True if feature i has drifted, False otherwise."""
        drift_flags = []
        for j, adw in enumerate(self.adwins):
            drift_happened = False
            for v in X_batch[:, j]:
                if adw.update(float(v)):
                    drift_happened = True
            drift_flags.append(drift_happened)
        return drift_flags

    def reset_adwins(self):
        self.adwins = [ADWIN(delta=self.delta) for _ in range(self.num_features)]
