import streamlit as st
import joblib
import pandas as pd

st.title("🌍 Exoplanet Habitability Predictor")

# Load model
st.write("Loading model...")
model = joblib.load("models/habitability_model.pkl")
st.success("✅ Model loaded successfully!")
#st.write(model.named_steps)
# Sidebar input
st.sidebar.header("Input Features")

def user_input():
    log_pl_orbsmax = st.sidebar.slider("log_pl_orbsmax", -2.0, 2.0, 0.0)
    log_st_lum = st.sidebar.slider("log_st_lum", -3.0, 1.0, -1.0)
    log_pl_rade = st.sidebar.slider("log_pl_rade", -1.0, 2.0, 0.0)
    log_teq = st.sidebar.slider("log_teq", 2.0, 4.0, 2.5)
    log_st_teff = st.sidebar.slider("log_st_teff", 3.0, 4.5, 3.7)
    log_stellar_flux = st.sidebar.slider("log_stellar_flux", -2.0, 3.0, 0.0)

    data = pd.DataFrame([[
        log_pl_orbsmax,
        log_st_lum,
        log_pl_rade,
        log_teq,
        log_st_teff,
        log_stellar_flux
    ]], columns=[
        'log_pl_orbsmax',
        'log_st_lum',
        'log_pl_rade',
        'log_teq',
        'log_st_teff',
        'log_stellar_flux'
    ])

    return data

sample = user_input()

# Threshold control
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

st.write("Input Data:")
st.write(sample)

# Prediction using probability + threshold
probability = model.predict_proba(sample)[0][1]
prediction = 1 if probability >= threshold else 0

# Output UI
st.subheader("Prediction Result")

if prediction == 1:
    st.success("🌱 Potentially Habitable")
else:
    st.error("❌ Not Habitable")

st.subheader("Habitability Probability")
st.write(f"{probability:.4f}")
st.progress(float(probability))

import shap

st.subheader("🔍 Model Explanation (SHAP)")

# Correct pipeline steps
scaler = model.named_steps['scaler']
lr_model = model.named_steps['model']

# Transform input
scaled_input = scaler.transform(sample)

# SHAP explainer
explainer = shap.LinearExplainer(lr_model, scaled_input)

shap_values = explainer.shap_values(scaled_input)

# Convert to readable format
shap_df = pd.DataFrame({
    "Feature": sample.columns,
    "Impact": shap_values[0]
}).sort_values(by="Impact", key=abs, ascending=False)

st.write(shap_df)