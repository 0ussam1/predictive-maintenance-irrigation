import pandas as pd
from pathlib import Path

def validate_data(df):
    """
    Validation de la qualité des données avant entraînement.
    """
    print("=== Validation des Données (DataOps) ===")
    
    checks = {
        "Pas de valeurs nulles": df.isnull().sum().sum() == 0,
        "Cible présente (failure_next_24h)": "failure_next_24h" in df.columns,
        "Features temporelles présentes": "hour" in df.columns,
        "Plus de 100 lignes": len(df) > 100
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {check}")
        if not passed:
            all_passed = False
            
    return all_passed

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/pump_features.csv")
        if validate_data(df):
            print("[SUCCESS] Validation DataOps réussie.")
        else:
            print("[CRITICAL] Échec de la validation des données.")
            exit(1)
    except Exception as e:
        print(f"Erreur : {e}")
        exit(1)
