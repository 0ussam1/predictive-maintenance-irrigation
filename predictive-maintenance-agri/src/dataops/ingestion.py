import pandas as pd
import numpy as np
from pathlib import Path

def ingest_data(file_path):
    """
    Charge les données brutes des capteurs d'irrigation.
    """
    print(f"Chargement des données depuis : {file_path}")
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
    
    df = pd.read_csv(file_path)
    print(f"Données chargées avec succès : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df

if __name__ == "__main__":
    # Point d'entrée pour test ou DVC
    raw_data_path = "data/raw/sensor.csv" 
    try:
        data = ingest_data(raw_data_path)
        print(data.head())
    except Exception as e:
        print(f"Erreur lors de l'ingestion : {e}")
