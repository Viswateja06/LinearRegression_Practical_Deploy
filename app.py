
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("linear_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page configuration
st.set_page_config(page_title="Marks Predictor", page_icon="📚")

st.title("Student Marks Prediction")
st.write("Predict your marks based on study hours.")

# User input
hours = st.number_input(
    "Enter study hours:",
    min_value=0.0,
    max_value=24.0,
    value=1.0,
    step=0.5
)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(np.array([[hours]]))[0]
    st.success(f"Predicted Marks: {prediction:.2f}")
