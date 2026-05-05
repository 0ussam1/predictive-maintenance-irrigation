"""
Module de validation des donnees.
Etape 4 du pipeline DataOps : verification de la qualite des donnees
avant l'entrainement du modele.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Features attendues dans le dataset final
EXPECTED_FEATURES = [
    "temperature_moteur", "vibration", "courant_electrique",
    "voltage", "pression_eau", "debit_eau", "rpm", "heures_fonctionnement",
    "hour", "day_of_week"
]


def validate_data(df):
    """
    Validation complete de la qualite des donnees avant entrainement.
    Retourne True si toutes les verifications passent.
    """
    print("=" * 60)
    print("  ETAPE 4 : VALIDATION DES DONNEES")
    print("=" * 60)

    checks = {}
    all_passed = True

    # ---- 1. VERIFICATIONS DE SCHEMA ----
    print("\n   [SCHEMA]")

    # Cible presente
    has_target = "failure_next_24h" in df.columns
    checks["Colonne cible 'failure_next_24h' presente"] = has_target
    if not has_target:
        all_passed = False

    # Features presentes
    missing_features = [f for f in EXPECTED_FEATURES if f not in df.columns]
    has_features = len(missing_features) == 0
    checks[f"Toutes les features presentes ({len(EXPECTED_FEATURES)})"] = has_features
    if not has_features:
        print(f"       MANQUANTES : {missing_features}")
        all_passed = False

    # ---- 2. VERIFICATIONS DE QUALITE ----
    print("   [QUALITE]")

    # Pas de NaN dans les features
    n_nan = df[EXPECTED_FEATURES].isnull().sum().sum()
    no_nans = n_nan == 0
    checks[f"Pas de NaN dans les features (trouve: {n_nan})"] = no_nans
    if not no_nans:
        all_passed = False

    # Volume minimum
    min_rows = 1000
    enough_data = len(df) >= min_rows
    checks[f"Volume minimum ({min_rows}+ lignes, actuel: {len(df)})"] = enough_data
    if not enough_data:
        all_passed = False

    # ---- 3. VERIFICATIONS DE DISTRIBUTION ----
    print("   [DISTRIBUTION]")

    # Les deux classes sont presentes
    if has_target:
        unique_classes = df["failure_next_24h"].nunique()
        both_classes = unique_classes >= 2
        checks["Les deux classes (0, 1) sont presentes"] = both_classes
        if not both_classes:
            all_passed = False

        # Equilibre minimum (au moins 5% de la classe minoritaire)
        class_counts = df["failure_next_24h"].value_counts()
        minority_pct = class_counts.min() / len(df) * 100
        balanced_enough = minority_pct >= 5.0
        checks[f"Equilibre acceptable (minorite: {minority_pct:.1f}%)"] = balanced_enough
        if not balanced_enough:
            all_passed = False

    # ---- 4. VERIFICATIONS DES TYPES ----
    print("   [TYPES]")

    # Toutes les features doivent etre numeriques
    numeric_ok = True
    for col in EXPECTED_FEATURES:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                numeric_ok = False
                break
    checks["Toutes les features sont numeriques"] = numeric_ok
    if not numeric_ok:
        all_passed = False

    # ---- 5. VERIFICATIONS DE PLAUSIBILITE ----
    print("   [PLAUSIBILITE]")

    # Pas de valeurs negatives pour les capteurs physiques
    positive_cols = ["temperature_moteur", "vibration", "rpm", "heures_fonctionnement"]
    negative_check = True
    for col in positive_cols:
        if col in df.columns and (df[col] < 0).any():
            negative_check = False
            break
    checks["Pas de valeurs negatives sur capteurs physiques"] = negative_check

    # ---- AFFICHAGE DES RESULTATS ----
    print("\n   " + "-" * 50)
    for check_name, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        icon = "+" if passed else "X"
        print(f"   [{icon}] [{status}] {check_name}")
    print("   " + "-" * 50)

    n_passed = sum(checks.values())
    n_total = len(checks)
    print(f"\n   Resultat : {n_passed}/{n_total} verifications passees")

    return all_passed


if __name__ == "__main__":
    try:
        df = pd.read_csv("data/processed/pump_features.csv", low_memory=False)
        if validate_data(df):
            print("\n[SUCCESS] Validation DataOps reussie.")
        else:
            print("\n[CRITICAL] Echec de la validation des donnees.")
            exit(1)
    except Exception as e:
        print(f"\n[ERREUR] Validation echouee : {e}")
        exit(1)
