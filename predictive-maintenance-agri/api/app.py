"""
API FastAPI de maintenance prédictive.
Le modèle est un Pipeline scikit-learn (SimpleImputer + StandardScaler + RandomForestClassifier).
AUCUNE valeur codée en dur. AUCUN safety override.
Le Random Forest décide seul basé sur son entraînement.
Les facteurs de conversion humain→capteur sont chargés depuis conversion_config.json.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import json
import pandas as pd
import numpy as np
import datetime
import os
from prometheus_client import Counter, Histogram, make_asgi_app

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

# Métriques Prometheus
PREDICTION_COUNT = Counter("prediction_total", "Total number of predictions")
FAILURE_PROBABILITY = Histogram("failure_probability_distribution", "Distribution of failure probabilities")

app = FastAPI(title="Predictive Maintenance API - Irrigation Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage de l'endpoint /metrics pour Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Service des fichiers statiques (Dashboard)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ──────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE ET DES MÉTADONNÉES
# ──────────────────────────────────────────────────────────

models_dir = os.path.join(os.path.dirname(__file__), "..", "models")

# Pipeline ML (SimpleImputer + StandardScaler + RandomForest)
model = joblib.load(os.path.join(models_dir, "model.pkl"))

# Métadonnées des features (ordre strict)
with open(os.path.join(models_dir, "features_meta.json"), "r") as f:
    features_meta = json.load(f)
FEATURE_NAMES = features_meta["feature_names"]

# Configuration de conversion humain→capteur (calculée à l'entraînement)
with open(os.path.join(models_dir, "conversion_config.json"), "r") as f:
    conversion_config = json.load(f)

# Statistiques de référence pour détection de drift
with open(os.path.join(models_dir, "reference_stats.json"), "r") as f:
    ref_stats = json.load(f)

print(f"[INFO] Modèle chargé : {features_meta.get('model_type', 'Unknown')}")
print(f"[INFO] Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")
print(f"[INFO] Conversion config chargée depuis conversion_config.json (apprise des données)")

# ──────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────

@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/history")
def read_history():
    return FileResponse(os.path.join(frontend_path, "history.html"))

@app.get("/guide")
def read_guide():
    return FileResponse(os.path.join(frontend_path, "guide.html"))


class MachineInput(BaseModel):
    machine_id: str
    temperature_moteur: float
    vibration: float
    courant_electrique: float
    voltage: float
    pression_eau: float
    debit_eau: float
    rpm: float
    heures_fonctionnement: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": features_meta.get("model_type", "RandomForestClassifier"),
        "model_features": len(FEATURE_NAMES),
        "pipeline_steps": [step[0] for step in model.steps]
    }


@app.post("/predict")
def predict(data: MachineInput):
    """
    Prédiction 100% IA par le modèle Random Forest.
    AUCUNE valeur codée en dur. AUCUN safety override.
    Le modèle décide seul basé sur son entraînement sur sensor.csv.
    """

    # 1. Construire le DataFrame avec les valeurs humaines brutes
    sensor_cols = [
        "temperature_moteur", "vibration", "courant_electrique",
        "voltage", "pression_eau", "debit_eau", "rpm", "heures_fonctionnement"
    ]
    raw_data = {col: getattr(data, col) for col in sensor_cols}
    df = pd.DataFrame([raw_data])

    # 2. Features temporelles (heure et jour actuels)
    now = datetime.datetime.now()
    df["hour"] = now.hour
    df["day_of_week"] = now.weekday()

    # 3. Conversion unités humaines → unités capteur
    #    Facteurs chargés depuis conversion_config.json (calculés à l'entraînement)
    drift_warnings = []
    for col in sensor_cols:
        if col in conversion_config:
            factor = conversion_config[col]["factor"]
            df[col] = df[col] * factor

            # Détection informative de drift (NE modifie PAS la prédiction)
            val = float(df[col].iloc[0])
            stats = ref_stats.get(col, {})
            if stats:
                ref_max = stats.get("max", float("inf"))
                ref_min = stats.get("min", 0)
                if val > ref_max * 1.3 or val < ref_min * 0.7:
                    drift_warnings.append(col)

    # 4. Ordonner les features selon l'ordre EXACT d'entraînement
    X = df[FEATURE_NAMES]

    # 5. Prédiction par le pipeline ML (Imputer → StandardScaler → RandomForest)
    prob = float(model.predict_proba(X)[0][1])

    print(f"[PREDICT] {data.machine_id} | prob={prob:.4f} | inputs={X.iloc[0].to_dict()}")

    # 6. Métriques Prometheus
    PREDICTION_COUNT.inc()
    FAILURE_PROBABILITY.observe(prob)

    # 7. Niveau d'alerte (basé UNIQUEMENT sur la probabilité du modèle RF)
    if prob < 0.20:
        etat = "optimisé"
        alerte = "optimisé"
        notif = "Machine en état de fonctionnement parfait."
        reco = "Excellentes conditions. Continuez l'exploitation."
    elif prob < 0.50:
        etat = "normale"
        alerte = "normale"
        notif = "Machine en état de fonctionnement standard."
        reco = "Surveillance routinière recommandée."
    elif prob < 0.80:
        etat = "need fix"
        alerte = "need_fix"
        notif = "Comportement anormal détecté par le modèle IA."
        reco = "Planifier une maintenance préventive rapidement."
    else:
        etat = "critique"
        alerte = "critique"
        notif = "Risque de panne imminent détecté par l'IA."
        reco = "Arrêt recommandé. Intervention immédiate requise !"

    return {
        "machine_id": data.machine_id,
        "etat_machine": etat,
        "probabilite_panne": round(prob, 4),
        "niveau_alerte": alerte,
        "notification": notif,
        "recommendation": reco,
        "warnings": drift_warnings if drift_warnings else None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)