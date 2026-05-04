"""
Script d'évaluation complète du modèle de maintenance prédictive.
Évalue sur le même split test que l'entraînement (même random_state).
"""

import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)


def main():
    print("=" * 60)
    print("  ÉVALUATION DU MODÈLE DE MAINTENANCE PRÉDICTIVE")
    print("=" * 60)

    # Charger les données
    print("\nChargement des données...")
    df = pd.read_csv("data/processed/pump_features.csv")
    model = joblib.load("models/model.pkl")

    # Charger les métadonnées des features
    with open("models/features_meta.json", "r") as f:
        features_meta = json.load(f)

    features = features_meta["feature_names"]
    X = df[features]
    y = df["failure_next_24h"]

    # Même split que pour l'entraînement (stratifié, random_state=42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Taille du jeu de test : {X_test.shape[0]} échantillons")
    print(f"Distribution : {dict(y_test.value_counts())}")

    # Prédictions
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, preds)

    print(f"\n┌───────────────────────────────────────┐")
    print(f"│    RÉSULTATS DE L'ÉVALUATION          │")
    print(f"├───────────────────────────────────────┤")
    print(f"│  Accuracy  : {acc:.4f}                │")
    print(f"│  Precision : {prec:.4f}                │")
    print(f"│  Recall    : {rec:.4f}                │")
    print(f"│  F1-Score  : {f1:.4f}                │")
    print(f"│  ROC AUC   : {auc:.4f}                │")
    print(f"└───────────────────────────────────────┘")

    print(f"\nMatrice de confusion :")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    print(f"\nRapport de classification complet :")
    print(classification_report(y_test, preds, target_names=["Normal", "Panne"]))

    # Vérification des performances
    if f1 >= 0.80:
        print("✅ Le modèle atteint les objectifs de performance (F1 >= 0.80)")
    else:
        print("⚠️ Le modèle n'atteint pas encore les objectifs (F1 < 0.80)")


if __name__ == "__main__":
    main()