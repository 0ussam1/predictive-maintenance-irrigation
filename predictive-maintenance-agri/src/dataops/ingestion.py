"""
Module d'ingestion des donnees brutes.
Etape 1 du pipeline DataOps : charge les donnees depuis data/raw/
et effectue les verifications de schema initiales.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Schema attendu du dataset brut sensor.csv
EXPECTED_COLUMNS = ["timestamp", "machine_status"]
SENSOR_COLUMNS = [f"sensor_{i:02d}" for i in range(52)]


def ingest_data(file_path):
    """
    Charge les donnees brutes des capteurs d'irrigation.
    Effectue la validation initiale du schema.
    """
    print("=" * 60)
    print("  ETAPE 1 : INGESTION DES DONNEES BRUTES")
    print("=" * 60)

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # Verification de la taille du fichier (protection contre les fichiers LFS corrompus)
    file_size = path.stat().st_size
    if file_size < 1000:
        raise ValueError(f"Fichier trop petit ({file_size} bytes) - probablement corrompu ou pointeur Git LFS.")

    print(f"   Source : {file_path}")
    print(f"   Taille : {file_size / 1024 / 1024:.1f} MB")

    df = pd.read_csv(file_path, low_memory=False)

    print(f"   Lignes : {df.shape[0]}")
    print(f"   Colonnes : {df.shape[1]}")

    # Validation du schema
    print("\n   Validation du schema :")
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")
    print(f"   [OK] Colonnes obligatoires presentes")

    # Verification de la colonne cible
    if "machine_status" in df.columns:
        status_counts = df["machine_status"].value_counts()
        print(f"\n   Distribution machine_status :")
        for status, count in status_counts.items():
            pct = count / len(df) * 100
            print(f"     {status:15s} : {count:>7d} ({pct:.1f}%)")

    # Suppression de la colonne index si presente
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print(f"\n   [OK] Colonne index 'Unnamed: 0' supprimee")

    print(f"\n   [OK] Ingestion terminee : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


if __name__ == "__main__":
    try:
        data = ingest_data("data/raw/sensor.csv")
        # Sauvegarde intermediaire
        interim_dir = Path("data/interim")
        interim_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(interim_dir / "sensor_ingested.csv", index=False)
        print("\n[SUCCESS] Ingestion DataOps terminee.")
    except Exception as e:
        print(f"\n[ERREUR] Ingestion echouee : {e}")
        exit(1)
