import pandas as pd
import joblib
import json

model = joblib.load('models/model.pkl')
df = pd.read_csv('data/processed/pump_features.csv')
with open('models/features_meta.json', 'r') as f:
    meta = json.load(f)
features = meta['feature_names']

df['prob'] = model.predict_proba(df[features])[:, 1]
normal_row = df.sort_values('prob').iloc[0]

print("=== MOST NORMAL ROW (Lowest Probability) ===")
print(normal_row[features + ['prob']].to_dict())

# Calculate what UI values would lead to this
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

ui_values = {}
for col, factor in mapping_factors.items():
    ui_values[col] = normal_row[col] / factor

print("\n=== EQUIVALENT UI VALUES ===")
print(json.dumps(ui_values, indent=2))
