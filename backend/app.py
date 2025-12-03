from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Define FastAPI app
app = FastAPI(title="Breast Cancer Prediction API")

# Load the latest model
model_path = "models/breast_cancer_model.pkl"
model_artifact = joblib.load(model_path)
model = model_artifact["model"]
scaler = model_artifact["scaler"]
feature_names = model_artifact["features"]

# Define request schema
class InputData(BaseModel):
    data: list  # list of feature values

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    X = np.array(input_data.data).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    pred_proba = model.predict_proba(X_scaled)[:, 1]
    return {"prediction": int(pred[0]), "probability": float(pred_proba[0])}
