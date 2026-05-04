import joblib
import pandas as pd

def predict(input_dict):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "failure_probability": float(prob)}