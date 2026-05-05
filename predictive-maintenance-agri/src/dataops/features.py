"""
Module d'ingenierie des features.
Etape 3 du pipeline DataOps : creation de variables temporelles,
rolling statistics, et augmentation de donnees synthetiques.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Colonnes capteurs
SENSOR_COLS = [
    "temperature_moteur", "vibration", "courant_electrique",
    "voltage", "pression_eau", "debit_eau", "rpm", "heures_fonctionnement"
]


def generate_features(df):
    """
    Ingenierie des variables pour la maintenance predictive.
    - Features temporelles (hour, day_of_week)
    - Augmentation de donnees synthetiques pour les pannes
    """
    print("=" * 60)
    print("  ETAPE 3 : INGENIERIE DES FEATURES")
    print("=" * 60)

    # ---- 1. FEATURES TEMPORELLES ----
    print("\n   [1] Creation des features temporelles...")
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="h")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    print(f"       hour : 0-23")
    print(f"       day_of_week : 0-6")

    # ---- 2. AUGMENTATION DE DONNEES SYNTHETIQUES ----
    print("\n   [2] Augmentation de donnees synthetiques...")
    n_failures_real = len(df[df["failure_next_24h"] == 1])
    n_normal_real = len(df[df["failure_next_24h"] == 0])
    print(f"       Avant augmentation : Normal={n_normal_real}, Panne={n_failures_real}")

    # On genere des cas de pannes synthetiques avec des valeurs extremes
    # Les valeurs sont sur l'echelle capteur (sensor units) pour etre coherent
    np.random.seed(42)
    n_synthetic = 10000

    # Statistiques normales du dataset pour reference
    normal_data = df[df["failure_next_24h"] == 0]
    normal_means = {col: normal_data[col].mean() for col in SENSOR_COLS}
    normal_stds = {col: normal_data[col].std() for col in SENSOR_COLS}

    # Generer des pannes avec des valeurs anormales (>2 ecarts-types au-dessus de la normale)
    synthetic = {}
    for col in SENSOR_COLS:
        mean = normal_means[col]
        std = normal_stds[col]
        if col in ["temperature_moteur", "vibration", "courant_electrique", "rpm", "heures_fonctionnement"]:
            # Ces capteurs augmentent en cas de panne
            synthetic[col] = np.random.uniform(mean + 2 * std, mean + 8 * std, n_synthetic)
        elif col in ["pression_eau", "debit_eau"]:
            # Ces capteurs diminuent en cas de panne
            synthetic[col] = np.random.uniform(max(0, mean - 6 * std), max(0, mean - 2 * std), n_synthetic)
        else:
            # Voltage : variation moderee
            synthetic[col] = np.random.uniform(mean + 1.5 * std, mean + 5 * std, n_synthetic)

    synthetic["failure_next_24h"] = np.ones(n_synthetic, dtype=int)
    synthetic["machine_id"] = ["pump_synthetic"] * n_synthetic
    synthetic["timestamp"] = pd.date_range(start="2024-01-01", periods=n_synthetic, freq="h")

    synthetic_df = pd.DataFrame(synthetic)
    synthetic_df["hour"] = synthetic_df["timestamp"].dt.hour
    synthetic_df["day_of_week"] = synthetic_df["timestamp"].dt.dayofweek

    df = pd.concat([df, synthetic_df], ignore_index=True)

    n_failures_final = len(df[df["failure_next_24h"] == 1])
    n_normal_final = len(df[df["failure_next_24h"] == 0])
    print(f"       Synthetiques ajoutes : {n_synthetic}")
    print(f"       Apres augmentation : Normal={n_normal_final}, Panne={n_failures_final}")

    # ---- 3. RESUME ----
    print(f"\n   [3] Resume des features :")
    feature_cols = SENSOR_COLS + ["hour", "day_of_week"]
    for col in feature_cols:
        if col in df.columns:
            print(f"       {col:25s} : min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")

    print(f"\n   Dataset final : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


if __name__ == "__main__":
    try:
        processed_dir = Path("data/processed")
        df = pd.read_csv(processed_dir / "irrigation_cleaned.csv")
        df_features = generate_features(df)
        df_features.to_csv(processed_dir / "pump_features.csv", index=False)
        print("\n[SUCCESS] Feature Engineering DataOps termine.")
    except Exception as e:
        print(f"\n[ERREUR] Feature Engineering echoue : {e}")
        exit(1)
