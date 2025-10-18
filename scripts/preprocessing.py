# Sección 1: Imports y rutas
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

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
                