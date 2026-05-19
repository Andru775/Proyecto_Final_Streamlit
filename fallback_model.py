import math
import pandas as pd

class DummyModel:
    """Simple deterministic fallback predictor.

    It computes a sigmoid of a weighted sum of a few numeric features
    to produce a pseudo-probability. This avoids heavy dependencies
    and ensures the Streamlit app runs even if XGBoost is not installed.
    """

    def __init__(self):
        # simple weights for selected features
        self.weights = {
            'satis': 0.25,
            'recomm': 0.25,
            'poverq': 0.2,
            'pq': 0.15,
            'customer_experience_index': 0.1
        }

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def _score_row(self, row):
        s = 0.0
        for k, w in self.weights.items():
            try:
                s += float(row.get(k, 7.5)) * w
            except Exception:
                s += 7.5 * w
        # normalize relative to scale (assuming inputs ~1-10)
        s = (s - 5) / 5
        return self._sigmoid(s)

    def predict_proba(self, X):
        # Accept pandas DataFrame or dict-like
        probs = []
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                p = self._score_row(row)
                probs.append([1-p, p])
        else:
            # single sample dict-like
            p = self._score_row(X)
            probs.append([1-p, p])
        return probs

    def predict(self, X):
        return [int(p[1] >= 0.5) for p in self.predict_proba(X)]
