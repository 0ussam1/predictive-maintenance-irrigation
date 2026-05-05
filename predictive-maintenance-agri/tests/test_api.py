import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ajout du chemin pour permettre l'import du module api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app
import json

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_normal():
    payload = {
        "machine_id": "TEST-01",
        "temperature_moteur": 25.0,
        "vibration": 0.2,
        "courant_electrique": 10.0,
        "voltage": 220.0,
        "pression_eau": 1.5,
        "debit_eau": 20.0,
        "rpm": 2000.0,
        "heures_fonctionnement": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probabilite_panne" in data
    assert "etat_machine" in data
    assert "niveau_alerte" in data

def test_predict_critical():
    # Vibration très élevée (45.0 alors que le max training est ~4.3)
    payload = {
        "machine_id": "TEST-CRIT",
        "temperature_moteur": 25.0,
        "vibration": 70.0, 
        "courant_electrique": 10.0,
        "voltage": 220.0,
        "pression_eau": 1.5,
        "debit_eau": 20.0,
        "rpm": 2000.0,
        "heures_fonctionnement": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # On s'attend à une probabilité élevée ou au moins des warnings
    assert data["probabilite_panne"] > 0.5
    assert data["warnings"] is not None
    assert "vibration" in data["warnings"]
