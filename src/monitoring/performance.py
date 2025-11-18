"""
Evaluación de desempeño del modelo antes y después del Data Drift.

Este script:
✔ Carga el modelo entrenado
✔ Evalúa métricas en dataset de validación (baseline)
✔ Evalúa métricas en dataset con drift
✔ Compara métricas
✔ Guarda el resultado en CSV para reportes automáticos
"""

import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys

# === RUTAS DEL PROYECTO ===
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

MODEL_PATH = ROOT / "src/models/pipeline_logreg_v2.joblib"
VALID_PATH = ROOT / "src/data/processed/german_credit_clean.csv"
DRIFT_PATH = ROOT / "src/data/drift/german_credit_drift.csv"
OUTPUT_PATH = ROOT / "reports/performance_comparison.csv"

TARGET = "kredit"   # Target real del dataset


# === FUNCIÓN PARA EVALUAR MÉTRICAS ===
def evaluate_model(model, X, y):
    """Calcula métricas de desempeño clásicas."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba)
    }


def main():
    print("\n=== EVALUACIÓN DE DESEMPEÑO DEL MODELO ===\n")

    # -------------------------------
    # 1. Cargar modelo
    # -------------------------------
    print("Cargando modelo...")
    model = joblib.load(MODEL_PATH)
    print("✔ Modelo cargado correctamente\n")

    # -------------------------------
    # 2. Cargar datasets
    # -------------------------------
    print("Leyendo datasets...")
    df_ref = pd.read_csv(VALID_PATH)
    df_drift = pd.read_csv(DRIFT_PATH)

    X_ref = df_ref.drop(columns=[TARGET])
    y_ref = df_ref[TARGET]

    X_drift = df_drift.drop(columns=[TARGET])
    y_drift = df_drift[TARGET]

    print("\n=== VALIDACIÓN DE COLUMNAS ===")
    print("Columnas REF:", set(X_ref.columns))
    print("Columnas DRIFT:", set(X_drift.columns))
    print("Columnas que faltan en DRIFT:", set(X_ref.columns) - set(X_drift.columns))
    print("Columnas nuevas en DRIFT:", set(X_drift.columns) - set(X_ref.columns))
    print("REF sparkont:", df_ref["sparkont"].unique()[:20])
    print("DRIFT sparkont:", df_drift["sparkont"].unique()[:20])






    # -------------------------------
    # 3. Evaluar modelo
    # -------------------------------
    print("Evaluando modelo en baseline (validación)...")
    baseline = evaluate_model(model, X_ref, y_ref)

    print("Evaluando modelo en dataset con drift...")
    drift_eval = evaluate_model(model, X_drift, y_drift)

    # -------------------------------
    # 4. Crear DataFrame comparativo
    # -------------------------------
    comparison = pd.DataFrame({
        "metric": ["accuracy", "f1_score", "roc_auc"],
        "baseline": [
            baseline["accuracy"],
            baseline["f1_score"],
            baseline["roc_auc"]
        ],
        "with_drift": [
            drift_eval["accuracy"],
            drift_eval["f1_score"],
            drift_eval["roc_auc"]
        ]
    })

    comparison["delta"] = comparison["with_drift"] - comparison["baseline"]

    print("\n=== RESULTADOS ===\n")
    print(comparison)

    # -------------------------------
    # 5. Guardar CSV en reports/
    # -------------------------------
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✔ Resultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
