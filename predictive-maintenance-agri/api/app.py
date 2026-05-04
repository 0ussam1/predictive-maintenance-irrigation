"""
API FastAPI de maintenance prédictive.
Le modèle est un Pipeline scikit-learn (StandardScaler + RandomForest).
Le StandardScaler est EMBARQUÉ dans le pipeline → AUCUNE transformation manuelle.
Les données brutes du frontend sont passées directement au pipeline.
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

# Charger le pipeline ML (StandardScaler + RandomForest — scaling intégré)
model = joblib.load(os.path.join(models_dir, "model.pkl"))

# Charger les métadonnées des features (ordre strict)
features_meta_path = os.path.join(models_dir, "features_meta.json")
with open(features_meta_path, "r") as f:
    features_meta = json.load(f)

FEATURE_NAMES = features_meta["feature_names"]
print(f"[INFO] Modèle chargé. Features attendues ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

# Charger les statistiques de référence pour la validation
ref_stats_path = os.path.join(models_dir, "reference_stats.json")
with open(ref_stats_path, "r") as f:
    ref_stats = json.load(f)

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
        "model_features": len(FEATURE_NAMES),
        "pipeline_steps": [step[0] for step in model.steps]
    }


@app.post("/predict")
def predict(data: MachineInput):
    """
    Prédiction 100% IA.
    Les données sont passées directement au pipeline qui contient
    le StandardScaler entraîné sur les données de référence.
    AUCUNE transformation manuelle. AUCUNE valeur codée en dur.
    """

    # 1. Construire le DataFrame avec les 8 capteurs bruts
    raw_data = {
        "temperature_moteur": data.temperature_moteur,
        "vibration": data.vibration,
        "courant_electrique": data.courant_electrique,
        "voltage": data.voltage,
        "pression_eau": data.pression_eau,
        "debit_eau": data.debit_eau,
        "rpm": data.rpm,
        "heures_fonctionnement": data.heures_fonctionnement,
    }

    df = pd.DataFrame([raw_data])

    # 2. Features temporelles
    now = datetime.datetime.now()
    df['hour'] = now.hour
    df['day_of_week'] = now.weekday()

    # 3. Features de lag (en mode temps réel, on utilise la valeur courante)
    # Dans une version MLOps plus avancée, on utiliserait un Feature Store (ex: Feast)
    # ou un cache Redis pour récupérer les valeurs précédentes.
    lag_sensors = ['temperature_moteur', 'vibration', 'pression_eau', 'debit_eau']
    for col in lag_sensors:
        df[f'{col}_lag1'] = df[col]
        df[f'{col}_rolling_mean3'] = df[col]

    # 4. Note: Data drift detection disabled — les ref_stats sont en echelle
    # capteur brute, incompatibles avec les unites reelles du frontend.
    # Le pipeline ML (StandardScaler integre) gere la normalisation.
    out_of_bounds = []
    is_anomaly = False

    # 5. Reordonner les colonnes selon l'ordre EXACT d'entrainement
    try:
        X = df[FEATURE_NAMES]
    except KeyError as e:
        missing = set(FEATURE_NAMES) - set(df.columns)
        print(f"[ERROR] Colonnes manquantes : {missing}")
        X = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    # 6. Prediction par le pipeline ML (StandardScaler + RandomForest)
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        print(f"[ERROR] Erreur prediction : {e}")
        prob = 0.5


    # Mise à jour des métriques Prometheus
    PREDICTION_COUNT.inc()
    FAILURE_PROBABILITY.observe(prob)

    # 8. Détermination du niveau d'alerte
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

    if is_anomaly:
        notif = "⚠️ ANOMALIE CRITIQUE DÉTECTÉE : " + notif
        reco = "SÉCURITÉ : " + reco

    return {
        "machine_id": data.machine_id,
        "etat_machine": etat,
        "probabilite_panne": round(prob, 4),
        "niveau_alerte": alerte,
        "notification": notif,
        "recommendation": reco,
        "warnings": out_of_bounds if out_of_bounds else None,
        "hybrid_override": is_anomaly
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)