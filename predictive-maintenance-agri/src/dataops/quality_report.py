import pandas as pd
import json
from pathlib import Path

def generate_quality_report(df, stage="raw"):
    """
    Génère un rapport de qualité des données pour le DataOps.
    """
    report = {
        "stage": stage,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict()
    }
    
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f"quality_report_{stage}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"[DATAOPS] Rapport de qualité généré : {report_file}")

if __name__ == "__main__":
    try:
        # On teste sur les données processées
        df = pd.read_csv("data/processed/pump_features.csv")
        generate_quality_report(df, "final")
    except Exception as e:
        print(f"Erreur rapport : {e}")
