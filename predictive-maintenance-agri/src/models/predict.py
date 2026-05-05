import joblib
import pandas as pd
import datetime
import os

def predict(input_dict):
    # Mapping factors (Human -> Sensor)
    mapping_factors = {
        "temperature_moteur": 2.16 / 45.0,
        "vibration": 44.4 / 4.5,
        "courant_electrique": 46.9 / 12.0,
        "voltage": 41.6 / 230.0,
        "pression_eau": 406.9 / 5.0,
        "debit_eau": 62.5 / 60.0,
        "rpm": 11.6 / 1500.0,
        "heures_fonctionnement": 14.1 / 1000.0
    }

    df = pd.DataFrame([input_dict])
    
    # 1. Temporal Features
    now = datetime.datetime.now()
    df['hour'] = now.hour
    df['day_of_week'] = now.weekday()

    # 2. Mapping
    for col, factor in mapping_factors.items():
        if col in df.columns:
            df[col] = df[col] * factor

    # 3. Load Metadata for exact order
    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    import json
    with open(os.path.join(models_dir, "features_meta.json"), "r") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    # 4. Reorder
    X = df[feature_names]

    # 5. Load and Predict
    model = joblib.load(os.path.join(models_dir, "model.pkl"))
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return {
        "prediction": pred,
        "failure_probability": round(prob, 4),
        "status": "broken" if pred == 1 else "operational"
    }