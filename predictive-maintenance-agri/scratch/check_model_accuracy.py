import pandas as pd
import joblib
import json

model = joblib.load('models/model.pkl')
df = pd.read_csv('data/processed/pump_features.csv')
with open('models/features_meta.json', 'r') as f:
    meta = json.load(f)
features = meta['feature_names']

X = df[features].head(100)
y_true = df['failure_next_24h'].head(100)
y_prob = model.predict_proba(X)[:, 1]

print(f"True Labels: {y_true.values[:10]}")
print(f"Probabilities: {y_prob[:10]}")
print(f"Mean Prob on 100 rows: {y_prob.mean():.4f}")
