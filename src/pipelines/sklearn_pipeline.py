from typing import List, Optional, Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


def build_pipeline(
    X: Optional[pd.DataFrame] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    model_name: str = 'logreg',
    model_params: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
) -> Pipeline:
    """Construye un sklearn Pipeline completo (preprocessing + estimator).

    Si `X` se proporciona y `numeric_features`/`categorical_features` son None,
    intentará inferir tipos automáticamentes (object -> categorical, number -> numeric).

    model_name: 'logreg'|'rf'|'xgb'
    """
    model_params = model_params or {}

    if X is not None and (numeric_features is None or categorical_features is None):
        cols_num = X.select_dtypes(include=['number']).columns.tolist()
        cols_cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col is not None and target_col in cols_num:
            cols_num = [c for c in cols_num if c != target_col]
        if target_col is not None and target_col in cols_cat:
            cols_cat = [c for c in cols_cat if c != target_col]
        numeric_features = cols_num
        categorical_features = cols_cat

    numeric_features = numeric_features or []
    categorical_features = categorical_features or []

    # numeric transformer: simple imputer + scaler
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # categorical transformer: impute missing then onehot
    # Build OneHotEncoder with compatibility for sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # newer sklearn uses sparse_output kwarg
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # choose estimator
    model_name = model_name.lower()
    if model_name == 'xgb' and XGBClassifier is not None:
        estimator = XGBClassifier(**model_params)
    elif model_name == 'rf':
        estimator = RandomForestClassifier(**model_params)
    else:
        # default to logistic regression
        estimator = LogisticRegression(max_iter=1000, **model_params)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ])

    return pipe
