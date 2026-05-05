import json
import pandas as pd
import joblib

with open("models/reference_stats.json", "r") as f:
    ref_stats = json.load(f)

raw_data = {
    "temperature_moteur": 80.0,
    "vibration": 20.0,
    "courant_electrique": 12.0,
    "voltage": 230.0,
    "pression_eau": 5.0,
    "debit_eau": 60.0,
    "rpm": 1500.0,
    "heures_fonctionnement": 1000.0,
}
df = pd.DataFrame([raw_data])
df['hour'] = 12
df['day_of_week'] = 2

mapping_factors = {
    "temperature_moteur": 2.467765 / 45.0,
    "vibration": 48.1218 / 4.5,
    "courant_electrique": 51.9606 / 12.0,
    "voltage": 44.304 / 230.0,
    "pression_eau": 633.98 / 5.0,
    "debit_eau": 75.162 / 60.0,
    "rpm": 13.968 / 1500.0,
    "heures_fonctionnement": 16.195 / 1000.0
}

is_anomaly = False
out_of_bounds = []
df_scaled = df.copy()

for col, factor in mapping_factors.items():
    if col in df_scaled.columns:
        mapped_val = df_scaled[col] * factor
        df_scaled[col] = mapped_val
        
        stats = ref_stats.get(col, {})
        if stats:
            iqr = stats['q75'] - stats['q25']
            margin = max(iqr * 1.5, stats['mean'] * 0.2)
            upper_bound = stats['q75'] + margin
            lower_bound = stats['q25'] - margin
            
            val = mapped_val.iloc[0]
            if val > upper_bound * 1.2 or val < lower_bound * 0.8:
                out_of_bounds.append(col)
                print(f"OOB: {col} val={val} upper={upper_bound*1.2} lower={lower_bound*0.8}")
            if val > upper_bound * 1.8 or val < lower_bound * 0.5:
                is_anomaly = True
                print(f"ANOMALY: {col} val={val}")

print(f"is_anomaly={is_anomaly}")
print(f"out_of_bounds={out_of_bounds}")

model = joblib.load("models/model.pkl")
with open("models/features_meta.json", "r") as f:
    features_meta = json.load(f)
FEATURE_NAMES = features_meta["feature_names"]

try:
    X = df_scaled[FEATURE_NAMES]
except KeyError as e:
    missing = set(FEATURE_NAMES) - set(df_scaled.columns)
    print(f"[ERROR] Colonnes manquantes : {missing}")
    X = df_scaled.reindex(columns=FEATURE_NAMES, fill_value=0)

print("X:", X)
prob = float(model.predict_proba(X)[0][1])
print(f"Prediction Probability: {prob}")

# Let's try an extreme value
df_scaled.loc[0, "temperature_moteur"] = 80.0 * (2.16/45.0) # very hot
df_scaled.loc[0, "vibration"] = 20.0 * (44.4/4.5) # very high
X2 = df_scaled[FEATURE_NAMES]
prob2 = float(model.predict_proba(X2)[0][1])
print(f"Prediction Probability (Extreme): {prob2}")
