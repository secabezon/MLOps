import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_PATH = ROOT / "src/data/processed/german_credit_clean.csv"
DRIFT_PATH = ROOT / "src/data/drift/german_credit_drift.csv"

TARGET = "kredit"

def main():
    df = pd.read_csv(VALID_PATH)

    # Identificar numéricas y categóricas
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols.remove(TARGET)  # no drift al target

    df_drift = df.copy()

    # Aplica drift suave SOLO en columnas numéricas
    for col in num_cols:
        df_drift[col] = df[col] * np.random.normal(1.05, 0.05, size=len(df))

    # Redondear para evitar problemas de pipeline
    df_drift[num_cols] = df_drift[num_cols].round(0)

    df_drift.to_csv(DRIFT_PATH, index=False)
    print("\n✔ Drift generado sin afectar columnas categóricas\n")

if __name__ == "__main__":
    main()
