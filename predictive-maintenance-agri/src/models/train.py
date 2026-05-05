"""
Module d'entrainement du modele de maintenance predictive.
Pipeline scikit-learn : StandardScaler + RandomForestClassifier.
Le scaler est embarque dans le pipeline - aucune transformation manuelle a l'inference.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Features utilisees par le modele
FEATURES = [
    "temperature_moteur", "vibration", "courant_electrique",
    "voltage", "pression_eau", "debit_eau", "rpm", "heures_fonctionnement",
    "hour", "day_of_week"
]


def main():
    print("=" * 60)
    print("  ENTRAINEMENT - RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. CHARGEMENT DES DONNEES
    # ----------------------------------------------------------
    print("\n[1/6] Chargement des donnees...")
    df = pd.read_csv("data/processed/pump_features.csv", low_memory=False)
    print(f"   Dataset brut : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # Nettoyage des NaN sur les features
    df = df.dropna(subset=FEATURES)
    df["failure_next_24h"] = df["failure_next_24h"].astype(int)

    X = df[FEATURES]
    y = df["failure_next_24h"]

    print(f"   Dataset propre : {len(df)} lignes")
    print(f"   Distribution : Normal={sum(y==0)}, Panne={sum(y==1)}")
    print(f"   Taux de panne : {y.mean()*100:.1f}%")

    # ----------------------------------------------------------
    # 2. EQUILIBRAGE DU DATASET
    # ----------------------------------------------------------
    print("\n[2/6] Equilibrage du dataset...")

    df_normal = df[df["failure_next_24h"] == 0]
    df_failure = df[df["failure_next_24h"] == 1]
    n_failures = len(df_failure)

    # On sous-echantillonne la classe normale pour equilibrer
    # On garde 1.5x le nombre de pannes pour un leger desequilibre realiste
    n_normal_sample = min(len(df_normal), int(n_failures * 1.5))
    df_normal_sampled = df_normal.sample(n=n_normal_sample, random_state=42)
    df_balanced = pd.concat([df_normal_sampled, df_failure], ignore_index=True)
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X_balanced = df_balanced[FEATURES]
    y_balanced = df_balanced["failure_next_24h"].astype(int)

    print(f"   Dataset equilibre : {len(df_balanced)} lignes")
    print(f"   Normal={sum(y_balanced==0)}, Panne={sum(y_balanced==1)}")

    # ----------------------------------------------------------
    # 3. SPLIT TRAIN / TEST (80/20 stratifie)
    # ----------------------------------------------------------
    print("\n[3/6] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced
    )
    print(f"   Train : {len(X_train)} lignes")
    print(f"   Test  : {len(X_test)} lignes")

    # ----------------------------------------------------------
    # 4. PIPELINE RANDOM FOREST
    # ----------------------------------------------------------
    print("\n[4/6] Construction du pipeline RandomForest...")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Cross-validation sur le train set pour evaluer la generalisation
    print("   Cross-validation (5 folds)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"   CV F1-Scores : {cv_scores.round(4)}")
    print(f"   CV F1 Mean   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Entrainement final sur tout le train set
    print("   Entrainement final...")
    pipeline.fit(X_train, y_train)

    # ----------------------------------------------------------
    # 5. EVALUATION SUR LE JEU DE TEST
    # ----------------------------------------------------------
    print("\n[5/6] Evaluation sur le jeu de test...")

    test_preds = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    auc = roc_auc_score(y_test, test_proba)
    cm = confusion_matrix(y_test, test_preds)

    print(f"\n  {'='*50}")
    print(f"      RESULTATS SUR JEU DE TEST")
    print(f"  {'='*50}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC AUC   : {auc:.4f}")
    print(f"  {'='*50}")

    print(f"\n  Matrice de confusion :")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

    print(f"\n  Rapport de classification :")
    print(classification_report(y_test, test_preds, target_names=["Normal", "Panne"]))

    # Feature importances
    rf_model = pipeline.named_steps['model']
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("  Feature Importances :")
    for i in sorted_idx:
        print(f"    {FEATURES[i]:25s} : {importances[i]:.4f}")

    # ----------------------------------------------------------
    # 6. SAUVEGARDE
    # ----------------------------------------------------------
    print("\n[6/6] Sauvegarde du modele et des metadonnees...")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Sauvegarde du pipeline complet (imputer + scaler + model)
    joblib.dump(pipeline, models_dir / "model.pkl")
    print(f"   Pipeline sauvegarde : models/model.pkl")

    # Sauvegarder les metadonnees des features
    features_meta = {
        "feature_names": FEATURES,
        "model_type": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 12,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "test_f1": float(f1),
        "test_accuracy": float(acc),
        "test_roc_auc": float(auc)
    }
    with open(models_dir / "features_meta.json", "w") as f:
        json.dump(features_meta, f, indent=2)
    print(f"   Metadonnees : models/features_meta.json")

    # Sauvegarder les statistiques de reference pour le monitoring de drift
    ref_stats = {}
    for col in FEATURES:
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
    print(f"   Stats de reference : models/reference_stats.json")

    # Feature importances JSON
    feat_importances = {FEATURES[i]: float(importances[i]) for i in range(len(FEATURES))}
    with open(models_dir / "feature_importances.json", "w") as f:
        json.dump(feat_importances, f, indent=2)
    print(f"   Feature importances : models/feature_importances.json")

    # Configuration de conversion Humain → Capteur (calculee depuis les donnees)
    SENSOR_COLS = [f for f in FEATURES if f not in ["hour", "day_of_week"]]
    normal_data_full = df[df["failure_next_24h"] == 0]
    sensor_means = {col: float(normal_data_full[col].mean()) for col in SENSOR_COLS}

    # Valeurs de reference humaines (calibration UI)
    human_references = {
        "temperature_moteur": 45.0,
        "vibration": 4.5,
        "courant_electrique": 12.0,
        "voltage": 230.0,
        "pression_eau": 5.0,
        "debit_eau": 60.0,
        "rpm": 1500.0,
        "heures_fonctionnement": 1000.0
    }

    conversion_config = {}
    for col in SENSOR_COLS:
        conversion_config[col] = {
            "sensor_mean": sensor_means[col],
            "human_reference": human_references[col],
            "factor": sensor_means[col] / human_references[col]
        }

    with open(models_dir / "conversion_config.json", "w") as f:
        json.dump(conversion_config, f, indent=2)
    print(f"   Conversion config : models/conversion_config.json")

    # Tracking MLflow
    try:
        mlflow.set_experiment("predictive-maintenance-agri")
        with mlflow.start_run(run_name="random_forest_v2"):
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 12)
            mlflow.log_param("pipeline", "Imputer + StandardScaler + RandomForest")
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size", X_test.shape[0])
            mlflow.log_param("n_features", len(FEATURES))
            mlflow.log_param("class_weight", "balanced")

            mlflow.log_metrics({
                "test_accuracy": acc,
                "test_f1": f1,
                "test_precision": precision,
                "test_recall": recall,
                "test_roc_auc": auc,
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std())
            })

            mlflow.sklearn.log_model(pipeline, "maintenance_model")
            print("   MLflow tracking : OK")
    except Exception as e:
        print(f"   MLflow (ignore) : {e}")

    print(f"\n{'=' * 60}")
    print(f"  ENTRAINEMENT TERMINE AVEC SUCCES")
    print(f"  Modele : RandomForestClassifier (200 arbres)")
    print(f"  F1-Score Test : {f1:.4f}")
    print(f"  ROC AUC Test  : {auc:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()