"""
Module de preprocessing des donnees.
Etape 2 du pipeline DataOps : nettoyage, mapping des colonnes,
gestion des valeurs manquantes, et traitement des outliers.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Mapping des capteurs bruts vers les noms metier irrigation
SENSOR_MAPPING = {
    "sensor_00": "temperature_moteur",
    "sensor_01": "vibration",
    "sensor_02": "courant_electrique",
    "sensor_03": "voltage",
    "sensor_04": "pression_eau",
    "sensor_05": "debit_eau",
    "sensor_06": "rpm",
    "sensor_07": "heures_fonctionnement"
}

# Colonnes metier finales
SENSOR_COLS = list(SENSOR_MAPPING.values())


def clean_data(df):
    """
    Nettoyage des donnees selon les standards DataOps.
    - Suppression des doublons
    - Mapping des colonnes capteurs vers les noms metier
    - Creation de la variable cible failure_next_24h
    - Gestion des valeurs manquantes (median)
    - Traitement des outliers sur les donnees normales (IQR)
    """
    print("=" * 60)
    print("  ETAPE 2 : PREPROCESSING DES DONNEES")
    print("=" * 60)

    # ---- 1. SUPPRESSION DES DOUBLONS ----
    initial_count = len(df)
    df = df.drop_duplicates()
    n_dupes = initial_count - len(df)
    print(f"\n   [1] Doublons supprimes : {n_dupes}")

    # ---- 2. CREATION DE LA VARIABLE CIBLE ----
    if "machine_status" in df.columns:
        df["failure_next_24h"] = df["machine_status"].apply(
            lambda x: 1 if x in ["BROKEN", "RECOVERING"] else 0
        )
        n_failures = df["failure_next_24h"].sum()
        n_normal = len(df) - n_failures
        print(f"   [2] Variable cible creee : Normal={n_normal}, Panne={n_failures}")
    else:
        raise ValueError("Colonne 'machine_status' manquante - impossible de creer la cible.")

    # ---- 3. MAPPING DES COLONNES CAPTEURS ----
    df = df.rename(columns=SENSOR_MAPPING)
    print(f"   [3] Colonnes renommees : {list(SENSOR_MAPPING.keys())} -> {SENSOR_COLS}")

    # Ajouter machine_id si manquant
    if "machine_id" not in df.columns:
        df["machine_id"] = "pump_01"

    # Garder uniquement les colonnes necessaires
    essential_cols = ["machine_id", "timestamp", "failure_next_24h"] + SENSOR_COLS
    available_cols = [col for col in essential_cols if col in df.columns]
    df = df[available_cols]
    print(f"   [3] Colonnes retenues : {len(available_cols)}")

    # ---- 4. GESTION DES VALEURS MANQUANTES ----
    print(f"\n   [4] Gestion des valeurs manquantes :")
    for col in SENSOR_COLS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                pct = n_missing / len(df) * 100
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"       {col:25s} : {n_missing:>6d} NaN ({pct:.1f}%) -> remplaces par median={median_val:.4f}")

    # ---- 5. TRAITEMENT DES OUTLIERS (uniquement sur les donnees normales) ----
    print(f"\n   [5] Traitement des outliers (IQR sur donnees normales) :")
    normal = df[df["failure_next_24h"] == 0].copy()
    abnormal = df[df["failure_next_24h"] == 1].copy()
    n_before = len(normal)

    for col in SENSOR_COLS:
        if col in normal.columns:
            Q1 = normal[col].quantile(0.25)
            Q3 = normal[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            normal = normal[(normal[col] >= lower) & (normal[col] <= upper)]

    n_removed = n_before - len(normal)
    print(f"       Outliers supprimes (normales) : {n_removed} lignes")

    df = pd.concat([normal, abnormal], ignore_index=True)

    print(f"\n   Dataset apres preprocessing : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"   Normal : {len(df[df['failure_next_24h']==0])}")
    print(f"   Panne  : {len(df[df['failure_next_24h']==1])}")

    return df


if __name__ == "__main__":
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_path = Path("data/raw/sensor.csv")
        if not raw_path.exists() or raw_path.stat().st_size < 1000:
            raise FileNotFoundError("Fichier sensor.csv manquant ou corrompu.")

        print("   Lecture depuis sensor.csv...")
        raw_df = pd.read_csv(raw_path, low_memory=False)

        # Supprimer colonne index si presente
        if "Unnamed: 0" in raw_df.columns:
            raw_df = raw_df.drop(columns=["Unnamed: 0"])

        clean_df = clean_data(raw_df)
        clean_df.to_csv(processed_dir / "irrigation_cleaned.csv", index=False)
        print("\n[SUCCESS] Preprocessing DataOps termine.")
    except Exception as e:
        print(f"\n[ERREUR] Preprocessing echoue : {e}")
        exit(1)
