"""
Module d'entraînement du modèle de maintenance prédictive.
Pipeline scikit-learn complet : StandardScaler + RandomForestClassifier.
Le scaler est embarqué dans le pipeline → aucune transformation manuelle à l'inférence.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    print("=" * 60)
    print("  ENTRAÎNEMENT DU MODÈLE DE MAINTENANCE PRÉDICTIVE")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 1. CHARGEMENT DES DONNÉES
    # ──────────────────────────────────────────────────────────
    print("\n[1/6] Chargement des données...")
    df = pd.read_csv("data/processed/pump_features.csv")
    print(f"  → Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Features = toutes les colonnes sauf la cible et les identifiants
    exclude_cols = ["failure_next_24h", "machine_id", "timestamp"]
    features = [c for c in df.columns if c not in exclude_cols]
    X = df[features]
    y = df["failure_next_24h"]

    print(f"  → Features utilisées ({len(features)}): {features}")
    print(f"  → Distribution cible : {dict(y.value_counts())}")
    print(f"  → Ratio de déséquilibre : 1:{int(y.value_counts()[0] / y.value_counts()[1])}")

    # Sauvegarder les noms des features pour garantir la cohérence à l'inférence
    features_meta = {
        "feature_names": features,
        "feature_count": len(features),
        "target": "failure_next_24h"
    }
    meta_path = Path("models/features_meta.json")
    meta_path.parent.mkdir(exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(features_meta, f, indent=2)
    print(f"  → Métadonnées des features sauvegardées : {meta_path}")

    # ──────────────────────────────────────────────────────────
    # 2. SPLIT TRAIN/TEST STRATIFIÉ
    # ──────────────────────────────────────────────────────────
    print("\n[2/6] Split Train/Test stratifié (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  → Train : {X_train.shape[0]} échantillons")
    print(f"  → Test  : {X_test.shape[0]} échantillons")
    print(f"  → Distribution train : {dict(y_train.value_counts())}")
    print(f"  → Distribution test  : {dict(y_test.value_counts())}")

    # ──────────────────────────────────────────────────────────
    # 3. PIPELINE ML AVEC RÉÉQUILIBRAGE
    # ──────────────────────────────────────────────────────────
    print("\n[3/6] Construction du pipeline ML...")
    print("  → StandardScaler + SMOTE + RandomForestClassifier")

    # Pipeline avec SMOTE pour rééquilibrer les classes
    # SMOTE est appliqué UNIQUEMENT sur le train set (jamais sur le test)
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
        ('model', RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    # ──────────────────────────────────────────────────────────
    # 4. RECHERCHE D'HYPERPARAMÈTRES (GridSearchCV)
    # ──────────────────────────────────────────────────────────
    print("\n[4/6] Recherche d'hyperparamètres avec GridSearchCV...")

    param_grid = {
        'model__n_estimators': [200, 300],
        'model__max_depth': [15, 20, 25],
        'model__min_samples_split': [5, 10],
        'model__min_samples_leaf': [2, 4],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    print(f"\n  → Meilleurs hyperparamètres : {grid_search.best_params_}")
    print(f"  → Meilleur F1 en cross-validation : {grid_search.best_score_:.4f}")

    # ──────────────────────────────────────────────────────────
    # 5. ÉVALUATION SUR LE JEU DE TEST (données jamais vues)
    # ──────────────────────────────────────────────────────────
    print("\n[5/6] Évaluation sur le jeu de test (données inconnues)...")

    # Pour la prédiction, on a besoin d'un pipeline SANS SMOTE
    # Extraire scaler et model du best pipeline
    best_scaler = best_pipeline.named_steps['scaler']
    best_model = best_pipeline.named_steps['model']

    # Créer un pipeline d'inférence propre (sans SMOTE)
    inference_pipeline = Pipeline([
        ('scaler', best_scaler),
        ('model', best_model)
    ])

    test_preds = inference_pipeline.predict(X_test)
    test_proba = inference_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_proba)
    cm = confusion_matrix(y_test, test_preds)

    print(f"\n  ┌───────────────────────────────────────┐")
    print(f"  │    RÉSULTATS SUR JEU DE TEST          │")
    print(f"  ├───────────────────────────────────────┤")
    print(f"  │  Accuracy  : {acc:.4f}                │")
    print(f"  │  Precision : {precision:.4f}                │")
    print(f"  │  Recall    : {recall:.4f}                │")
    print(f"  │  F1-Score  : {f1:.4f}                │")
    print(f"  │  ROC AUC   : {auc:.4f}                │")
    print(f"  └───────────────────────────────────────┘")

    print(f"\n  Matrice de confusion :")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    print(f"\n  Rapport de classification complet :")
    print(classification_report(y_test, test_preds, target_names=["Normal", "Panne"]))

    # ──────────────────────────────────────────────────────────
    # 6. SAUVEGARDE DU MODÈLE + TRACKING MLFLOW
    # ──────────────────────────────────────────────────────────
    print("[6/6] Sauvegarde et tracking MLflow...")

    mlflow.set_experiment("predictive-maintenance-agri")

    with mlflow.start_run(run_name="rf_optimized_smote"):
        # Log des paramètres
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("pipeline", "StandardScaler + SMOTE + RF")
        mlflow.log_param("train_size", X_train.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])
        mlflow.log_param("n_features", len(features))

        # Log des métriques
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_roc_auc": auc,
            "cv_best_f1": grid_search.best_score_
        })

        # Log du modèle d'inférence (SANS SMOTE)
        mlflow.sklearn.log_model(inference_pipeline, "maintenance_model")

    # Sauvegarde locale du pipeline d'inférence
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(inference_pipeline, models_dir / "model.pkl")

    # Sauvegarder aussi les statistiques de référence du dataset pour monitoring
    ref_stats = {}
    for col in features:
        ref_stats[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "q25": float(X_train[col].quantile(0.25)),
            "q75": float(X_train[col].quantile(0.75))
        }
    with open(models_dir / "reference_stats.json", "w") as f:
        json.dump(ref_stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  ✅ MODÈLE SAUVEGARDÉ : models/model.pkl")
    print(f"  ✅ MÉTADONNÉES       : models/features_meta.json")
    print(f"  ✅ STATS RÉFÉRENCE   : models/reference_stats.json")
    print(f"  ✅ MLFLOW TRACKING   : OK")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()