# src/train.py
import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from src.utils import load_data, preprocess_data

# ---------------------------
# 1️⃣ Load and preprocess data
# ---------------------------
X, y, feature_names = load_data()
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

# ---------------------------
# 2️⃣ Train model
# ---------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 3️⃣ Evaluate model
# ---------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability for positive class

accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_prob)
rmse = np.sqrt(mean_squared_error(y_test, y_prob))  # use probabilities for RMSE

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Log Loss: {loss:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# ---------------------------
# 4️⃣ Save metrics to JSON
# ---------------------------
os.makedirs("results", exist_ok=True)
results = {
    "accuracy": accuracy,
    "log_loss": loss,
    "rmse": rmse
}
with open("results/experiment_metrics.json", "w") as f:
    json.dump(results, f)

# ---------------------------
# 5️⃣ Save model artifact
# ---------------------------
os.makedirs("models", exist_ok=True)
model_path = "models/breast_cancer_model.pkl"
joblib.dump({
    "model": model,
    "scaler": scaler,
    "features": feature_names.tolist()
}, model_path)

print("✅ Model and metrics saved successfully.")

