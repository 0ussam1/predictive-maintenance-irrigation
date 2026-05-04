import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(df):
    """
    Nettoyage des données selon les standards DataOps.
    """
    print("=== Nettoyage des données (DataOps) ===")
    
    # 1. Suppression des doublons
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Doublons supprimés : {initial_shape[0] - df.shape[0]}")

    # 2. Mapping vers le concept "Irrigation Intelligente"
    # On mappe les colonnes du dataset 'sensor.csv' vers le nouveau concept
    column_mapping = {
        'sensor_00': 'temperature_moteur',
        'sensor_01': 'vibration',
        'sensor_02': 'courant_electrique',
        'sensor_03': 'voltage',
        'sensor_04': 'pression_eau',
        'sensor_05': 'debit_eau',
        'sensor_06': 'rpm',
        'sensor_07': 'heures_fonctionnement'
    }
    
    # Création de la cible failure_next_24h basée sur machine_status
    if 'machine_status' in df.columns:
        df['failure_next_24h'] = df['machine_status'].apply(lambda x: 1 if x in ['BROKEN', 'RECOVERING'] else 0)
    
    # Renommer les capteurs
    df = df.rename(columns=column_mapping)
    
    # Garder uniquement les colonnes nécessaires au concept
    essential_cols = ['machine_id', 'timestamp', 'failure_next_24h'] + list(column_mapping.values())
    df = df[[col for col in essential_cols if col in df.columns]]

    # Ajouter machine_id si manquant
    if 'machine_id' not in df.columns:
        df['machine_id'] = "pump_01"

    # 3. Gestion des valeurs manquantes
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    df = df.drop(columns=cols_to_drop)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # 4. Traitement des outliers (Uniquement sur les données normales)
    if 'failure_next_24h' in df.columns:
        sensor_cols = list(column_mapping.values())
        normal = df[df['failure_next_24h'] == 0].copy()
        abnormal = df[df['failure_next_24h'] == 1].copy()
        
        for col in sensor_cols:
            if col in normal.columns:
                Q1 = normal[col].quantile(0.25)
                Q3 = normal[col].quantile(0.75)
                IQR = Q3 - Q1
                normal = normal[(normal[col] >= (Q1 - 1.5 * IQR)) & (normal[col] <= (Q3 + 1.5 * IQR))]
        
        df = pd.concat([normal, abnormal], ignore_index=True)

    # 5. AUGMENTATION DE DONNÉES (Expertise Métier)
    # On ajoute des cas de pannes par surchauffe/vibration pour que le modèle soit intuitif
    print("  → Ajout de données synthétiques (Surchauffe & Vibrations)...")
    np.random.seed(42)
    n_synthetic = 20000
    
    synthetic_failures = pd.DataFrame({
        'temperature_moteur': np.random.uniform(10.0, 50.0, n_synthetic),
        'vibration': np.random.uniform(60.0, 150.0, n_synthetic),
        'courant_electrique': np.random.uniform(60.0, 100.0, n_synthetic),
        'voltage': np.random.uniform(50.0, 80.0, n_synthetic),
        'pression_eau': np.random.uniform(600.0, 800.0, n_synthetic),
        'debit_eau': np.random.uniform(80.0, 120.0, n_synthetic),
        'rpm': np.random.uniform(20.0, 40.0, n_synthetic),
        'heures_fonctionnement': np.random.uniform(20.0, 30.0, n_synthetic),
        'failure_next_24h': 1,
        'machine_id': 'pump_synthetic'
    })
    
    df = pd.concat([df, synthetic_failures], ignore_index=True)

    print(f"Forme finale des données (augmentées) : {df.shape}")
    return df

if __name__ == "__main__":
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Test local
    try:
        raw_path = Path("data/raw/sensor.csv")
        cleaned_path = Path("data/processed/irrigation_cleaned.csv")
        
        if raw_path.exists() and raw_path.stat().st_size > 1000:
            print("  → Lecture depuis sensor.csv")
            raw_df = pd.read_csv(raw_path)
        elif cleaned_path.exists():
            print("  → Lecture depuis irrigation_cleaned.csv (sensor.csv manquant)")
            raw_df = pd.read_csv(cleaned_path)
        else:
            raise FileNotFoundError("Aucun fichier de données source trouvé.")
            
        clean_df = clean_data(raw_df)
        clean_df.to_csv(processed_dir / "irrigation_cleaned.csv", index=False)
        print("[SUCCESS] Prétraitement DataOps terminé.")
    except Exception as e:
        print(f"Erreur : {e}")
