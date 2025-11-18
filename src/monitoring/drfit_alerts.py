from pathlib import Path
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[2]
VALID_PATH = ROOT / "src/data/processed/german_credit_clean.csv"
MONITOR_PATH = ROOT / "src/data/drift/german_credit_drift.csv"

TARGET_COL = "kredit"
ALPHA = 0.05  # Significancia (como Evidently usa por defecto)


def compute_drift_ks():
    df_ref = pd.read_csv(VALID_PATH)
    df_cur = pd.read_csv(MONITOR_PATH)

    feature_cols = [c for c in df_ref.columns if c != TARGET_COL]

    drift_results = {}

    for col in feature_cols:
        stat, p_value = ks_2samp(df_ref[col], df_cur[col])
        drift_results[col] = {
            "p_value": float(p_value),
            "drift_detected": p_value < ALPHA
        }

    return drift_results


def generate_alerts():

    drift = compute_drift_ks()

    drifted_cols = [col for col, info in drift.items() if info["drift_detected"]]
    drift_share = len(drifted_cols) / len(drift)

    dataset_drift = drift_share > 0.3  # regla común


    print("\n=== DATA DRIFT ALERTS ===\n")
    print(f"➡ Dataset drift: {dataset_drift}")
    print(f"➡ Drift share: {drift_share:.2f}")
    print(f"➡ Columns with drift ({len(drifted_cols)}):")
    print(", ".join(drifted_cols) if drifted_cols else "Ninguna")

    print("\n=== RECOMMENDATIONS ===")
    if dataset_drift:
        print("⚠️ Drift severo → retraining recomendado.")
    elif drift_share > 0.3:
        print("⚠️ Drift moderado → revisar pipeline.")
    else:
        print("✓ Modelo estable.")


if __name__ == "__main__":
    generate_alerts()
