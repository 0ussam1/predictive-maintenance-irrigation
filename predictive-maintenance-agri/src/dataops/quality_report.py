"""
Module de rapport de qualite des donnees.
Etape 5 du pipeline DataOps : genere un rapport complet sur la qualite
des donnees a chaque etape du pipeline.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


def generate_quality_report(df, stage="raw"):
    """
    Genere un rapport de qualite detaille des donnees pour le DataOps.
    """
    print("=" * 60)
    print(f"  RAPPORT DE QUALITE - Stage: {stage.upper()}")
    print("=" * 60)

    report = {
        "stage": stage,
        "generated_at": datetime.now().isoformat(),
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": list(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        },
        "missing_values": {},
        "numeric_stats": {},
        "target_distribution": {},
        "data_quality_score": 0.0
    }

    # ---- VALEURS MANQUANTES ----
    total_missing = 0
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        pct = round(n_missing / len(df) * 100, 2)
        report["missing_values"][col] = {
            "count": n_missing,
            "percentage": pct
        }
        total_missing += n_missing

    # ---- STATISTIQUES NUMERIQUES ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        report["numeric_stats"][col] = {
            "mean": round(float(df[col].mean()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "q25": round(float(df[col].quantile(0.25)), 4),
            "median": round(float(df[col].quantile(0.50)), 4),
            "q75": round(float(df[col].quantile(0.75)), 4),
            "skewness": round(float(df[col].skew()), 4),
            "n_zeros": int((df[col] == 0).sum()),
            "n_negative": int((df[col] < 0).sum())
        }

    # ---- DISTRIBUTION DE LA CIBLE ----
    if "failure_next_24h" in df.columns:
        counts = df["failure_next_24h"].value_counts()
        report["target_distribution"] = {
            "normal_count": int(counts.get(0, 0)),
            "failure_count": int(counts.get(1, 0)),
            "failure_rate_pct": round(counts.get(1, 0) / len(df) * 100, 2),
            "imbalance_ratio": round(counts.get(0, 1) / max(counts.get(1, 1), 1), 2)
        }

    # ---- SCORE DE QUALITE ----
    # Score sur 100 base sur plusieurs criteres
    score = 100.0
    total_cells = len(df) * len(df.columns)
    missing_pct = total_missing / total_cells * 100 if total_cells > 0 else 0
    score -= min(missing_pct * 5, 30)  # -5 points par % de NaN, max -30

    if len(df) < 1000:
        score -= 20  # Penalite volume
    if "failure_next_24h" in df.columns:
        failure_rate = df["failure_next_24h"].mean() * 100
        if failure_rate < 1 or failure_rate > 50:
            score -= 15  # Desequilibre extreme

    # Doublons
    n_dupes = df.duplicated().sum()
    dupe_pct = n_dupes / len(df) * 100
    score -= min(dupe_pct * 2, 20)

    report["data_quality_score"] = round(max(score, 0), 1)
    report["duplicates"] = {
        "count": int(n_dupes),
        "percentage": round(dupe_pct, 2)
    }

    # ---- SAUVEGARDE ----
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_file = reports_dir / f"quality_report_{stage}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n   Lignes          : {report['dataset']['rows']}")
    print(f"   Colonnes        : {report['dataset']['columns']}")
    print(f"   Memoire         : {report['dataset']['memory_usage_mb']} MB")
    print(f"   NaN total       : {total_missing}")
    print(f"   Doublons        : {n_dupes}")
    if "failure_next_24h" in df.columns:
        print(f"   Taux de panne   : {report['target_distribution']['failure_rate_pct']}%")
    print(f"   Score qualite   : {report['data_quality_score']}/100")
    print(f"\n   Rapport sauvegarde : {report_file}")

    return report


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/pump_features.csv", low_memory=False)
        generate_quality_report(df, "final")
        print("\n[SUCCESS] Rapport de qualite genere.")
    except Exception as e:
        print(f"\n[ERREUR] Rapport echoue : {e}")
        exit(1)
