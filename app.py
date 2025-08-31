import pickle
import numpy as np
import streamlit as st

# =========================
# Load Model
# =========================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter **PCA-transformed features** + normalized fields to predict fraud.")

# =========================
# Input Section
# =========================
st.subheader("🔢 Input PCA Features (V1–V28) + normAmount + normTime")

feature_names = [f"V{i}" for i in range(1, 29)] + ["normAmount", "normTime"]

input_values = []
cols = st.columns(3)  # 3 columns for neat input layout

for idx, feat in enumerate(feature_names):
    with cols[idx % 3]:
        val = st.number_input(f"{feat}", value=0.0, format="%.6f")
        input_values.append(val)

# =========================
# Prediction
# =========================
if st.button("🚀 Predict"):
    try:
        features = np.array(input_values).reshape(1, -1)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0, 1]

        st.subheader("📌 Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ Fraud Detected! (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2f})")

    except Exception as e:
        st.warning(f"Error in prediction: {e}")
