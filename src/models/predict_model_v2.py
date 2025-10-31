# src/models/predict_model_v2.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import classification_report

DATA_PATH = Path("src/data/processed/german_credit_clean.csv")
MODEL_PATH = Path("src/models/pipeline_logreg_v2.joblib")
TARGET = "kredit"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
