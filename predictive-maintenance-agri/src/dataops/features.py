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
    
    # 2. Features Glissantes (Lags & Rolling)
    # Très important pour détecter les dégradations progressives
    sensors = ['temperature_moteur', 'vibration', 'pression_eau', 'debit_eau']
    
    for sensor in sensors:
        if sensor in df.columns:
            # Lag 1h
            df[f'{sensor}_lag1'] = df.groupby('machine_id')[sensor].shift(1)
            # Moyenne mobile 3h
            df[f'{sensor}_rolling_mean3'] = df.groupby('machine_id')[sensor].transform(lambda x: x.rolling(window=3).mean())

    # Nettoyage des NaNs créés par les lags
    df = df.dropna()
    
    print(f"Nouvelles features générées. Colonnes totales : {len(df.columns)}")
    return df

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/irrigation_cleaned.csv")
        df_features = generate_features(df)
        df_features.to_csv("data/processed/pump_features.csv", index=False)
        print("[SUCCESS] Feature Engineering DataOps terminé.")
    except Exception as e:
        print(f"Erreur : {e}")
