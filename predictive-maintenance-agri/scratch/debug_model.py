import json
import joblib
import pandas as pd
from pathlib import Path

def debug():
    meta_path = Path("models/features_meta.json")
    model_path = Path("models/model.pkl")
    
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    pipe = joblib.load(model_path)
    
    print(f"DEBUG: Features expected by Meta: {meta['feature_names']}")
    print(f"DEBUG: Model steps: {[s[0] for s in pipe.steps]}")
    
    # Test a prediction with Normal values
    test_data = {
        "temperature_moteur": 2.41,
        "vibration": 47.4,
        "courant_electrique": 51.0,
        "voltage": 43.8,
        "pression_eau": 592.7,
        "debit_eau": 72.8,
        "rpm": 13.5,
        "heures_fonctionnement": 15.8,
        "hour": 12,
        "day_of_week": 0
    }
    df = pd.DataFrame([test_data])
    X = df[meta['feature_names']]
    prob = pipe.predict_proba(X)[0][1]
    print(f"DEBUG: Normal Test Probability: {prob}")

    # Test extreme
    test_data["temperature_moteur"] = 5000.0
    df = pd.DataFrame([test_data])
    X = df[meta['feature_names']]
    prob_extreme = pipe.predict_proba(X)[0][1]
    print(f"DEBUG: Extreme Test Probability: {prob_extreme}")

if __name__ == "__main__":
    debug()
