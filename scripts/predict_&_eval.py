# Sección 1: Imports y rutas
from modeling_training import train
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class predict_eval:

    def __init__(self, model: train):
        super().__init__
        self.model=model

    def predict(self, x_test):
        model=self.model
        y_pred=model.predict(x_test)
        return y_pred
    
    def predict_proba(self, x_test):
        model=self.model
        y_pred=model.predict_proba(x_test)
        return y_pred
    
    def eval(self, x_test, y_test):
        y_pred=self.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        proba = self.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        return {'f1': f1, 'accurracy': acc, 'ROC': auc}
    
