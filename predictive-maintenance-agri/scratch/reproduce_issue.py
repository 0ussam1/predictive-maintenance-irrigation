import requests
import json

url = "http://127.0.0.1:8000/predict"
payload = {
    "machine_id": "PUMP-AGRI-01",
    "temperature_moteur": 45.0, # Maps to 2.16 (Mean)
    "vibration": 4.5, # Maps to 44.4 (Mean)
    "courant_electrique": 12.0, # Maps to 46.9 (Mean)
    "voltage": 230.0, # Maps to 41.6 (Mean)
    "pression_eau": 5.0, # Maps to 406.9 (Mean)
    "debit_eau": 60.0, # Maps to 62.5 (Mean)
    "rpm": 1500.0, # Maps to 11.6 (Mean)
    "heures_fonctionnement": 1000.0 # Maps to 14.1 (Mean)
}

try:
    response = requests.post(url, json=payload)
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
