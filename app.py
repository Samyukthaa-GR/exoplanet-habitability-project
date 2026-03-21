import streamlit as st
import joblib
import pandas as pd

st.title("🌍 Exoplanet Habitability Predictor")

# Load model
st.write("Loading model...")
model = joblib.load("models/habitability_model.pkl")
st.success("✅ Model loaded successfully!")

# Sidebar input
st.sidebar.header("Input Features")

log_pl_orbsmax = st.sidebar.slider(
    "log Orbital Distance (log_pl_orbsmax)",
    min_value=-2.0,
    max_value=2.0,
    value=0.0
)

# Sample input (ONLY first value dynamic)
st.write("Running prediction...")

sample = pd.DataFrame([[
    log_pl_orbsmax,  # from slider
    -1.0,
    0.0,
    2.5,
    3.7,
    0.0
]], columns=[
    'log_pl_orbsmax',
    'log_st_lum',
    'log_pl_rade',
    'log_teq',
    'log_st_teff',
    'log_stellar_flux'
])

# Prediction
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

# Output
st.write("Prediction:", prediction)
st.write("Habitability Probability:", probability)