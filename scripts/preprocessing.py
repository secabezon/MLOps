import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression


class load:

    def __init__(self, path):
        super().__init__
        self.df=pd.read_csv(path)

class preprocessing:

    def __init__(self, df: load):
        super().__init__
        self.df=df

    def delete_cols(self,delete_col):
        df = self.df
        df = df.drop(columns=delete_col)
        return df

    def clean_colsname(self,old_val,new_val):
        df=self.df
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(old_val, new_val).str.lower()
        return df
    
    def dropduplicates(self):
        df = self.df.drop_duplicates().reset_index(drop=True)
        return df
    
    def normalize_miss_val(self,missingvals):
        df = self.df
        for i in df.cols:
            df[i]=df[i].apply(lambda x: np.nan if x in missingvals else x)
        return df
    
    def convert_num(self,numeric_col):
        df = self.df
        for nc in numeric_col:
            df[nc] = pd.to_numeric(df[nc], errors='coerce')
        return df
    
    def imputer_val(self,numeric_cols):
        df = self.df
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        numeric_array = df[numeric_cols].to_numpy(dtype=float)
        imputed_array = imputer.fit_transform(numeric_array)
        
        df[numeric_cols] = pd.DataFrame(imputed_array, index=df.index, columns=numeric_cols)
        return df
    
    def delete_outliers(self,numeric_cols, k):
        df = self.df
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            df=df(df[col]>lower | df[col]<upper)
        return df
                