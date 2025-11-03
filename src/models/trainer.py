from typing import Optional, Dict, Any
import joblib
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class ModelTrainer:
    """Entrenador ligero que acepta un estimator (puede ser Pipeline) y expone fit/evaluate/save."""

    def __init__(self, estimator: BaseEstimator, model_name: str = 'model'):
        self.estimator = estimator
        self.model_name = model_name
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, **fit_kwargs):
        self.estimator.fit(X, y, **fit_kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        if hasattr(self.estimator, 'predict_proba'):
            return self.estimator.predict_proba(X)
        # try to access underlying estimator (for pipeline)
        try:
            last = self.estimator.steps[-1][1]
            if hasattr(last, 'predict_proba'):
                return last.predict_proba(self._get_transformed(X))
        except Exception:
            pass
        return None

    def _get_transformed(self, X: pd.DataFrame):
        # if pipeline, apply preprocessing only
        try:
            pre = self.estimator.named_steps.get('preprocessor')
            return pre.transform(X)
        except Exception:
            return X

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred, average='weighted'))
        }
        proba = self.predict_proba(X)
        if proba is not None:
            try:
                # if binary, take column 1
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    proba_col = proba[:, 1]
                else:
                    proba_col = proba.ravel()
                metrics['roc_auc'] = float(roc_auc_score(y, proba_col))
            except Exception:
                metrics['roc_auc'] = float('nan')
        else:
            metrics['roc_auc'] = float('nan')

        return metrics

    def save(self, path: str, overwrite: bool = False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"Model file {path} already exists; set overwrite=True to replace")
        joblib.dump(self.estimator, path)
        return path
