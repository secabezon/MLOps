import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


class load:

    def __init__(self, path):
        super().__init__()
        self.df=pd.read_csv(path)

    def __str__(self):
        return str(self.df)

class preprocessing:

    def __init__(self, df: load):
        super().__init__()
        self.df=df

    def delete_cols(self, delete_col):
        self.df = self.df.drop(columns=delete_col)
        return self

    def clean_colsname(self,old_val,new_val):
        self.df.columns = self.df.columns.str.strip()
        self.df.columns = self.df.columns.str.replace(old_val, new_val).str.lower()
        return self
    
    def dropduplicates(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        return self
    
    def normalize_miss_val(self,missingvals):
        df = self.df
        for i in df.columns:
            df[i]=df[i].apply(lambda x: np.nan if x in missingvals else x)
        return self
    
    def convert_num(self,numeric_col):
        for nc in numeric_col:
            self.df[nc] = pd.to_numeric(self.df[nc], errors='coerce')
        return self
    
    def imputer_val(self,numeric_cols):
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        numeric_array = self.df[numeric_cols].to_numpy(dtype=float)
        imputed_array = imputer.fit_transform(numeric_array)
        
        self.df[numeric_cols] = pd.DataFrame(imputed_array, index=self.df.index, columns=numeric_cols)
        return self
    
    def cap_outliers(self,numeric_cols, k, apply_cap=True):
        for col in numeric_cols:
            s = pd.to_numeric(self.df[col], errors='coerce').astype(float)

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr

            if apply_cap:
                non_out = s[~((s < lower) | (s > upper))].dropna()
                cap_low = non_out.min() if not non_out.empty else lower
                cap_high = non_out.max() if not non_out.empty else upper

                s[s < lower] = cap_low
                s[s > upper] = cap_high

                try:
                    self.df[col] = pd.Series(pd.to_numeric(s, errors='coerce')).round(0).astype('Int64')
                except Exception:
                    self.df[col] = s
        return self
      
    def __str__(self):
        return str(self.df) 

    def describe(self):
        return str(self.df.describe)           