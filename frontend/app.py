import streamlit as st
import requests

st.title("Breast Cancer Prediction")

# Inputs
features = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

inputs = {}
for f in features:
    inputs[f] = st.number_input(f, value=0.0)

if st.button("Predict"):
    response = requests.post("http://localhost:8000/predict", json=inputs)
    st.write("Prediction (0 = Benign, 1 = Malignant):", response.json()["prediction"])

