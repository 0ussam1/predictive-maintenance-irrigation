import pandas as pd
import numpy as np
from pathlib import Path

def generate_features(df):
    """
    Ingénierie des variables pour la maintenance prédictive.
    """
    print("=== Engineering des Features (DataOps) ===")
    
    # 1. Features Temporelles (si timestamp existe, sinon on simule pour le concept)
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 2. On supprime les calculs de Lag qui causaient la suppression des données synthétiques
    # (Les données synthétiques n'ont pas d'historique, donc dropna() les supprimait toutes)
    
    print(f"Features générées. Colonnes totales : {len(df.columns)}")
    return df


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/irrigation_cleaned.csv")
        df_features = generate_features(df)
        df_features.to_csv("data/processed/pump_features.csv", index=False)
        print("[SUCCESS] Feature Engineering DataOps terminé.")
    except Exception as e:
        print(f"Erreur : {e}")
