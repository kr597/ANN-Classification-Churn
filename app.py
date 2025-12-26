import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# =========================
# Load trained model
# ========================= 
model = tf.keras.models.load_model('model.h5')

# =========================
# Load encoders & scaler
# =========================
with open('label_encode_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

# =========================
# Streamlit UI
# =========================
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# =========================
# Prepare input data (DICT)
# =========================
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Geography': geography
}

# =========================
# Convert to DataFrame (CRITICAL)
# =========================
input_df = pd.DataFrame([input_data])

# =========================
# One-hot encode Geography
# =========================
geo_encoded = one_hot_encoder_geo.transform(
    input_df[['Geography']]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
)

# =========================
# Drop original Geography column
# =========================
input_df.drop('Geography', axis=1, inplace=True)

# =========================
# Combine all features
# =========================
input_df = pd.concat(
    [input_df.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# =========================
# Ensure correct feature order
# =========================
input_df = input_df[scaler.feature_names_in_]

# =========================
# Scale input
# =========================
input_scaled = scaler.transform(input_df)

# =========================
# Predict
# =========================
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# =========================
# Output
# =========================
st.subheader(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("❌ The customer is likely to churn.")
else:
    st.success("✅ The customer is unlikely to churn.")
