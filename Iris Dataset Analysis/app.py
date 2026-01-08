import joblib
import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

# Load model
model = joblib.load("iris_logistic_regression_model.pkl")

# Title & description
st.markdown("<h1 style='text-align: center;'>🌸 Iris Flower Species Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Adjust the flower measurements and click <b>Predict</b> to see the species.</p>",
    unsafe_allow_html=True
)

st.divider()

# Input layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)

with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

st.divider()

# Centered Predict button
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict = st.button("🔍 Predict Species", use_container_width=True)

# Prediction logic
if predict:
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    species_map = {
        0: "🌼 Iris-setosa",
        1: "🌷 Iris-versicolor",
        2: "🌹 Iris-virginica"
    }

    st.success(f"**Predicted Species:** {species_map[prediction]}")



