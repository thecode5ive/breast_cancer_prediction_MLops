from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load latest model
artifact = joblib.load("../models/model_v1.pkl")
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["features"]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([{f: data[f] for f in feature_names}])
    df_scaled = scaler.transform(df)
    pred = model.predict(df_scaled)
    return {"prediction": int(pred[0])}
