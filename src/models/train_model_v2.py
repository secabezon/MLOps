from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.features.preprocessing_transformer import PreprocessingTransformer

# rutas
DATA_PATH = Path("src/data/processed/german_credit_clean.csv")
MODEL_PATH = Path("src/models/pipeline_logreg_v2.joblib")

# variable objetivo
TARGET = "kredit"

# columnas numéricas (sin el target)
NUMERIC_FEATURES = [
    'laufkont', 'laufzeit', 'moral', 'verw', 'hoehe', 'sparkont',
    'beszeit', 'rate', 'famges', 'buerge', 'wohnzeit', 'verm',
    'alter', 'weitkred', 'wohn', 'bishkred', 'beruf', 'pers',
    'telef', 'gastarb'
]


def main():
    print("➡️  Cargando datos...", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    if TARGET not in df.columns:
        raise ValueError(f"La columna objetivo '{TARGET}' no está en el dataset. Columnas: {df.columns.tolist()}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("➡️  Haciendo train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Split listo.")
    print("   Train:", X_train.shape, "Test:", X_test.shape)

    print("➡️  Creando pipeline (preprocesamiento + modelo)...")
    pipe = Pipeline(steps=[
        ("prep", PreprocessingTransformer(numeric_cols=NUMERIC_FEATURES)),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("➡️  Entrenando modelo...")
    pipe.fit(X_train, y_train)
    print("✅ Modelo entrenado.")

    print("➡️  Evaluando...")
    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("📊 Reporte de clasificación:\n", report)

    print("➡️  Guardando modelo en disco...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"💾 Modelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
