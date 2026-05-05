"""
Script d'evaluation complete du modele de maintenance predictive.
Evalue sur le meme split test que l'entrainement (meme random_state).
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)


def main():
    print("=" * 60)
    print("  EVALUATION DU MODELE DE MAINTENANCE PREDICTIVE")
    print("=" * 60)

    # Charger les donnees
    print("\nChargement des donnees et du modele...")
    df = pd.read_csv("data/processed/pump_features.csv", low_memory=False)
    model = joblib.load("models/model.pkl")

    # Charger les metadonnees des features
    with open("models/features_meta.json", "r") as f:
        features_meta = json.load(f)

    features = features_meta["feature_names"]
    df = df.dropna(subset=features)
    df["failure_next_24h"] = df["failure_next_24h"].astype(int)

    X = df[features]
    y = df["failure_next_24h"]

    # Equilibrage identique a l'entrainement
    df_normal = df[df["failure_next_24h"] == 0]
    df_failure = df[df["failure_next_24h"] == 1]
    n_failures = len(df_failure)
    n_normal_sample = min(len(df_normal), int(n_failures * 1.5))
    df_normal_sampled = df_normal.sample(n=n_normal_sample, random_state=42)
    df_balanced = pd.concat([df_normal_sampled, df_failure], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X_balanced = df_balanced[features]
    y_balanced = df_balanced["failure_next_24h"].astype(int)

    # Meme split que pour l'entrainement (stratifie, random_state=42)
    _, X_test, _, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    print(f"Taille du jeu de test : {X_test.shape[0]} echantillons")
    print(f"Distribution : Normal={sum(y_test==0)}, Panne={sum(y_test==1)}")

    # Predictions
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Metriques
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    print(f"\n  {'='*50}")
    print(f"    RESULTATS DE L'EVALUATION")
    print(f"  {'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC AUC   : {auc:.4f}")
    print(f"  {'='*50}")

    print(f"\nMatrice de confusion :")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    print(f"\nRapport de classification complet :")
    print(classification_report(y_test, preds, target_names=["Normal", "Panne"]))

    # Sauvegarder le rapport d'evaluation
    eval_report = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1])
        },
        "test_size": int(len(X_test)),
        "model_type": features_meta.get("model_type", "Unknown")
    }

    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    # Verification des performances
    if f1 >= 0.80:
        print(f"[OK] Le modele atteint les objectifs de performance (F1={f1:.4f} >= 0.80)")
    else:
        print(f"[ATTENTION] Le modele n'atteint pas encore les objectifs (F1={f1:.4f} < 0.80)")


if __name__ == "__main__":
    main()