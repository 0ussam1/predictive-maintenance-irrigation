import requests

base = {"vibration": 4.5, "courant_electrique": 12, "voltage": 230,
        "pression_eau": 5, "debit_eau": 60, "rpm": 1500, "heures_fonctionnement": 1000}

tests = [
    ("45C  normal", 45, 4.5, 5.0, 60),
    ("70C  eleve",  70, 6.0, 3.5, 45),
    ("100C haut",  100, 4.5, 5.0, 60),
    ("200C extreme",200, 4.5, 5.0, 60),
]

print("=" * 65)
print("  PREUVE : Le Random Forest predit SEUL (zero valeur hardcodee)")
print("=" * 65)

for label, temp, vib, pres, debit in tests:
    payload = {
        "machine_id": "TEST",
        "temperature_moteur": temp,
        "vibration": vib,
        "courant_electrique": 12,
        "voltage": 230,
        "pression_eau": pres,
        "debit_eau": debit,
        "rpm": 1500,
        "heures_fonctionnement": 1000
    }
    r = requests.post("http://127.0.0.1:8000/predict", json=payload).json()
    prob = r["probabilite_panne"]
    alerte = r["niveau_alerte"]
    warn = r.get("warnings", [])
    print(f"  {label:15s} -> prob = {prob:.4f} ({prob*100:.1f}%)  | alerte = {alerte:10s} | warnings = {warn}")

print("=" * 65)
