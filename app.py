import streamlit as st 
import pandas as pd
import numpy as np
import pickle 
import tensorflow as tf

# -----------------------------
# Load Saved Artifacts
# -----------------------------
model = tf.keras.models.load_model("artifacts/churn_model.h5")

with open("artifacts/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("artifacts/onehot_encoder_geography.pkl", "rb") as file:
    onehot_encoder_geography = pickle.load(file)

with open("artifacts/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# If you saved feature order (recommended)
try:
    with open("artifacts/features.pkl", "rb") as file:
        feature_names = pickle.load(file)
except FileNotFoundError:
    feature_names = scaler.feature_names_in_  # fallback if features.pkl not found

# -----------------------------
# Streamlit App
# -----------------------------
st.title("💡 Customer Churn Prediction App")

st.write("Provide the customer details below to predict the likelihood of churn.")

# User Input
geography = st.selectbox("🌍 Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
age = st.slider("🎂 Age", 18, 100, 30)
balance = st.number_input("💰 Balance", value=0.0)
credit_score = st.number_input("💳 Credit Score", value=600)
estimated_salary = st.number_input("💵 Estimated Salary", value=50000)
tenure = st.number_input("📅 Tenure (Years)", 0, 10, 3)
num_of_products = st.slider("🛍 Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member", [0, 1])

# -----------------------------
# Data Preparation
# -----------------------------
# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Encode geography
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
)

# Combine all features
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# Merge one-hot encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure feature order matches training
input_data = input_data.reindex(columns=feature_names, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
try:
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = float(prediction[0][0])

    st.subheader("🔍 Prediction Result:")
    st.write(f"**Churn Probability:** {prediction_proba:.2f}")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")

except Exception as e:
    st.error(f"⚠️ Error: {e}")

# -----------------------------
# Debugging Info (Optional)
# -----------------------------
with st.expander("See Debug Info"):
    st.write("Expected Features:", list(scaler.feature_names_in_))
    st.write("Input Data Columns:", list(input_data.columns))
