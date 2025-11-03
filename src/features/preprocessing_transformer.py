# src/features/preprocessing_transformer.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, k_neighbors=5, k_outlier=1.5, apply_cap=True):
        self.numeric_cols = numeric_cols
        self.k_neighbors = k_neighbors
        self.k_outlier = k_outlier
        self.apply_cap = apply_cap
        self.imputer_ = None

    def fit(self, X, y=None):
        X = X.copy()
        self.imputer_ = KNNImputer(n_neighbors=self.k_neighbors)
        self.imputer_.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_cols] = self.imputer_.transform(X[self.numeric_cols])

        if self.apply_cap:
            for col in self.numeric_cols:
                s = pd.to_numeric(X[col], errors='coerce').astype(float)
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower = q1 - self.k_outlier * iqr
                upper = q3 + self.k_outlier * iqr
                s[s < lower] = lower
                s[s > upper] = upper
                X[col] = s

        return X
