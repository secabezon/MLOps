# Sección 1: Imports y rutas
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class split:

    def __init__(self, X, y):
        super().__init__
        self.X=X
        self.y=y

    def split_df(self):
        X=self.X
        y=self.y
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()<=10 else None)
        else:
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train = y_test = None
        return X_train, X_test, y_train, y_test
    
class train:

    def __init__(self, Split: split):
        super().__init__()
        self.Split = Split
        self.X_train, self.X_test, self.y_train, self.y_test = self.Split.split_df()

    def xgboost(self):
        X_train= self.X_train
        y_train= self.y_train
        xgb = XGBClassifier(
            n_estimators=300,    
            learning_rate=0.1,    
            max_depth=6,          
            subsample=0.8,         
            colsample_bytree=0.8,   
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'    
        )

        xgb_clasif=xgb.fit(X_train,y_train)
        return xgb_clasif
    
    def reglog(self):
        X_train= self.X_train
        y_train= self.y_train
        logreg = LogisticRegression(max_iter=1000, solver='lbfgs')

        xgb_clasif=logreg.fit(X_train,y_train)
        return xgb_clasif
    
    def reglog(self):
        X_train= self.X_train
        y_train= self.y_train
        reglog = LogisticRegression(max_iter=1000, solver='lbfgs')

        reglog_clasif=reglog.fit(X_train,y_train)
        return reglog_clasif
    
    def rf(self):
        X_train= self.X_train
        y_train= self.y_train
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        rf_clasif=rf.fit(X_train,y_train)
        return rf_clasif

