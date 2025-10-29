import streamlit as st 
import pandas as pd
import numpy as np
import pickle 
import tensorflow as tf

# -----------------------------
# Load Saved Artifacts
# -----------------------------
st.set_page_config(page_title="Salary Prediction App", page_icon="💰")

st.title("💡 Estimated Salary Prediction App")
st.write("Predict your estimated salary based on demographics and financial details.")

# Load model and preprocessing objects
model = tf.keras.models.load_model("artifacts/salary_regression_model.h5")

with open("artifacts/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("artifacts/onehot_encoder_geography.pkl", "rb") as file:
    onehot_encoder_geography = pickle.load(file)

with open("artifacts/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Try loading feature names
try:
    with open("artifacts/features.pkl", "rb") as file:
        feature_names = pickle.load(file)
except FileNotFoundError:
    feature_names = scaler.feature_names_in_  # fallback

# -----------------------------
# User Inputs
# -----------------------------
geography = st.selectbox("🌍 Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
age = st.slider("🎂 Age", 18, 100, 30)
balance = st.number_input("💰 Balance", value=0.0)
credit_score = st.number_input("💳 Credit Score", value=600)
exited = st.selectbox("🚪 Exited", [0, 1])
tenure = st.slider("📅 Tenure (Years)", 0, 10)
num_of_products = st.slider("🛍 Number of Products", 1, 4)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member", [0, 1])

# -----------------------------
# Data Preprocessing
# -----------------------------
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
)

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})

# Merge one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure order of columns matches model training
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Apply scaling
scaled_input = scaler.transform(input_data)

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("💡 Predict Salary"):
    prediction = model.predict(scaled_input)
    estimated_salary = prediction[0][0]
    st.success(f"💰 Estimated Salary: ₹{estimated_salary:,.2f}")
