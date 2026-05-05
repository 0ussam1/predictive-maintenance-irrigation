import pandas as pd
import json

df = pd.read_csv('data/processed/pump_features.csv')
features = json.load(open('models/features_meta.json'))['feature_names']
df = df.dropna(subset=features)

print('=== DATASET ANALYSIS ===')
print(f'Total rows: {len(df)}')
print(f'Features: {features}')
print(f'\nTarget distribution:')
print(df['failure_next_24h'].value_counts())
fail_rate = df['failure_next_24h'].mean() * 100
print(f'\nFailure rate: {fail_rate:.1f}%')

print('\n=== NORMAL vs FAILURE means ===')
for f in features[:8]:
    nm = df[df['failure_next_24h'] == 0][f].mean()
    fm = df[df['failure_next_24h'] == 1][f].mean()
    ratio = fm / nm if nm > 0 else 0
    print(f'{f:25s} Normal={nm:.4f}  Failure={fm:.4f}  Ratio={ratio:.2f}')

print('\n=== NORMAL stats ===')
print(df[df['failure_next_24h'] == 0][features[:8]].describe().round(3))
print('\n=== FAILURE stats ===')
print(df[df['failure_next_24h'] == 1][features[:8]].describe().round(3))
