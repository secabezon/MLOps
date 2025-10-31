# Sección 1: Imports y rutas
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocessing import preprocessing


class split:

    def __init__(self, preprocessing: preprocessing, col_obj):
        super().__init__()
        self.y=preprocessing.df[col_obj]
        self.X=preprocessing.delete_cols(col_obj).df

    def split_df(self):
        X=self.X
        y=self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()<=10 else None)
        return X_train, X_test, y_train, y_test
    
class train:

    def __init__(self, Split: split):
        super().__init__()
        self.Split = Split
        self.X_train, self.X_test, self.y_train, self.y_test = self.Split.split_df()

    def xgboost(self,n_estimators,learning_rate,max_depth):
        X_train= self.X_train
        y_train= self.y_train
        xgb = XGBClassifier(
            n_estimators=n_estimators,    
            learning_rate=learning_rate,    
            max_depth=max_depth,          
            subsample=0.8,         
            colsample_bytree=0.8,   
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'    
        )

        xgb_clasif=xgb.fit(X_train,y_train)
        return xgb_clasif
    
    def reglog(self,max_iter):
        X_train= self.X_train
        y_train= self.y_train
        reglog = LogisticRegression(max_iter=max_iter, solver='lbfgs')

        reglog_clasif=reglog.fit(X_train,y_train)
        return reglog_clasif
    
    def rf(self,n_estimators,random_state,n_jobs):
        X_train= self.X_train
        y_train= self.y_train
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)

        rf_clasif=rf.fit(X_train,y_train)
        return rf_clasif

