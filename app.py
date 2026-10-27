
import streamlit as st
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Student Marks Prediction")

st.title("Student Marks Prediction")

# Find the model in the same directory as app.py
MODEL_PATH = Path(__file__).parent / "linear_regression_model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file not found. Please generate and upload linear_regression_model.pkl.")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

hours = st.number_input(
    "Enter study hours:",
    min_value=0.0,
    max_value=24.0,
    value=1.0,
    step=0.5
)

if st.button("Predict"):
    prediction = model.predict(np.array([[hours]]))[0]
    st.success(f"Predicted Marks: {prediction:.2f}")
