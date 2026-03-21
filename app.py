import streamlit as st

st.title("🌍 Exoplanet Habitability Predictor")
import joblib

st.write("Loading model...")

model = joblib.load("models/habitability_model.pkl")

st.success("✅ Model loaded successfully!")

import pandas as pd

st.write("Running test prediction...")

# Dummy input (must match training features EXACTLY)
sample = pd.DataFrame([[
    -0.8,   # closer orbit (but not too close)
    -1.5,   # lower luminosity star
    0.0,    # small rocky planet
    2.4,    # moderate temperature
    3.5,    # cooler star
    -0.5
]], columns=[
    'log_pl_orbsmax',
    'log_st_lum',
    'log_pl_rade',
    'log_teq',
    'log_st_teff',
    'log_stellar_flux'
])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

st.write("Prediction:", prediction)
st.write("Habitability Probability:", probability)