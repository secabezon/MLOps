from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


class Preprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor compatible con sklearn (fit/transform).

    Funcionalidades mínimas:
    - Normalizar tokens de missing
    - Limpiar nombres de columnas (strip, replace, lower)
    - Convertir columnas a numéricas donde aplique
    - Imputar valores numéricos (SimpleImputer o KNNImputer)
    - Capa para capear outliers basada en IQR
    - El transform devuelve un pandas.DataFrame (no muta entrada)
    """

    def __init__(
        self,
        missing_values: Optional[List[str]] = None,
        imputer_strategy: str = 'simple',
        knn_neighbors: int = 5,
        cap_iqr_k: float = 1.5,
        drop_cols: Optional[List[str]] = None,
        clean_col_old: str = ' ',
        clean_col_new: str = '_',
    ):
        self.missing_values = missing_values or ["n/a", "na", "null", "?", "unknown", "error", "invalid", "none", "", " "]
        self.imputer_strategy = imputer_strategy
        self.knn_neighbors = knn_neighbors
        self.cap_iqr_k = cap_iqr_k
        self.drop_cols = drop_cols or ['mixed_type_col']
        self.clean_col_old = clean_col_old
        self.clean_col_new = clean_col_new

        # attributes set in fit
        self.numeric_cols_ = []
        self.imputer_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        # normalize missing tokens first
        df = self._normalize_missing(df)

        # clean column names
        df.columns = df.columns.str.strip().str.replace(self.clean_col_old, self.clean_col_new).str.lower()

        # drop configured columns if present
        for c in self.drop_cols:
            if c in df.columns:
                df = df.drop(columns=c)

        # infer numeric cols
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        # also try to coerce non-numeric strings to numeric and detect more
        for col in df.columns:
            if col not in self.numeric_cols_:
                try:
                    coerced = pd.to_numeric(df[col], errors='coerce')
                    # if many values convert, treat as numeric
                    non_na = coerced.notna().sum()
                    if non_na >= max(1, int(0.5 * len(coerced))):
                        self.numeric_cols_.append(col)
                except Exception:
                    pass

        # set up imputer
        if self.imputer_strategy == 'knn':
            self.imputer_ = KNNImputer(n_neighbors=self.knn_neighbors)
        else:
            self.imputer_ = SimpleImputer(strategy='mean')

        # fit imputer on numeric columns (if any)
        if self.numeric_cols_:
            numeric_array = df[self.numeric_cols_].to_numpy(dtype=float)
            # Fit imputer (SimpleImputer/KNNImputer)
            self.imputer_.fit(numeric_array)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._normalize_missing(df)

        # clean column names
        df.columns = df.columns.str.strip().str.replace(self.clean_col_old, self.clean_col_new).str.lower()

        # drop configured columns
        for c in self.drop_cols:
            if c in df.columns:
                df = df.drop(columns=c)

        # convert numeric cols
        for nc in self.numeric_cols_:
            if nc in df.columns:
                df[nc] = pd.to_numeric(df[nc], errors='coerce')

        # impute
        if self.numeric_cols_ and self.imputer_ is not None:
            # ensure order of columns
            cols = [c for c in self.numeric_cols_ if c in df.columns]
            if cols:
                numeric_array = df[cols].to_numpy(dtype=float)
                imputed = self.imputer_.transform(numeric_array)
                df[cols] = pd.DataFrame(imputed, index=df.index, columns=cols)

        # cap outliers
        for col in self.numeric_cols_:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors='coerce').astype(float)
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.cap_iqr_k * iqr
                upper = q3 + self.cap_iqr_k * iqr
                non_out = s[~((s < lower) | (s > upper))].dropna()
                cap_low = non_out.min() if not non_out.empty else lower
                cap_high = non_out.max() if not non_out.empty else upper
                s.loc[s < lower] = cap_low
                s.loc[s > upper] = cap_high
                df[col] = s

        # drop duplicates
        df = df.drop_duplicates().reset_index(drop=True)

        return df

    def _normalize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip().lower() in [m.strip().lower() for m in self.missing_values] else x)
        return df

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
