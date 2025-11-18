

from pathlib import Path
import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently import ColumnMapping

ROOT = Path(__file__).resolve().parents[2]
VALID_PATH = ROOT / "src/data/processed/german_credit_clean.csv"
MONITOR_PATH = ROOT / "src/data/drift/german_credit_drift.csv"
REPORTS_DIR = ROOT / "reports"

TARGET_COL = "kredit"   # Tu target real

def main():

    REPORTS_DIR.mkdir(exist_ok=True)

    df_ref = pd.read_csv(VALID_PATH)
    df_cur = pd.read_csv(MONITOR_PATH)

    # Usar solo las columnas de features
    feature_cols = [c for c in df_ref.columns if c != TARGET_COL]
    df_ref = df_ref[feature_cols]
    df_cur = df_cur[feature_cols]

    num_cols = df_ref.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    mapping = ColumnMapping(
        numerical_features=num_cols,
        categorical_features=cat_cols
    )

    # Crear dashboard
    drift_dashboard = Dashboard(tabs=[DataDriftTab()])

    # Ejecutar cálculo (API de Evidently 0.2.2)
    drift_dashboard.calculate(
        reference_data=df_ref,
        current_data=df_cur,
        column_mapping=mapping
    )

    # Guardar salida HTML (API de Evidently 0.2.2)
    drift_dashboard.save(str(REPORTS_DIR / "data_drift_dashboard.html"))

    print("\n✔ Dashboard generado: data_drift_dashboard.html\n")

if __name__ == "__main__":
    main()
