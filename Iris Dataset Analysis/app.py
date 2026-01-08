import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered"
)

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("iris_logistic_regression_model.pkl")
model = load_model()

# -------------------------
# Load Iris dataset for preview
# -------------------------
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    df["species"] = df["species"].map({
        0: "Iris-setosa",
        1: "Iris-versicolor",
        2: "Iris-virginica"
    })
    return df
df = load_iris_data()

# -------------------------
# Header with image
# -------------------------
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    width=500,
    caption="Iris Flower Example"
)

st.markdown("<h1 style='text-align: center; color: #4B0082;'>🌸 Iris Flower Species Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Move the sliders to enter the flower measurements and click Predict.</p>", unsafe_allow_html=True)
st.divider()

# -------------------------
# Input sliders
# -------------------------
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# -------------------------
# Predict button
# -------------------------
predict_col1, predict_col2, predict_col3 = st.columns([1,2,1])
with predict_col2:
    predict = st.button("🔍 Predict Species", use_container_width=True)

# -------------------------
# Prediction & confidence
# -------------------------
if predict:
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    species_map = {
        0: "🌼 Iris-setosa",
        1: "🌷 Iris-versicolor",
        2: "🌹 Iris-virginica"
    }

    confidence = probabilities[prediction]

    # Display predicted species
    st.markdown(
        f"<h2 style='text-align: center; color:#4B0082;'>Predicted Species: {species_map[prediction]}</h2>",
        unsafe_allow_html=True
    )

    # Colored confidence bar
    if confidence > 0.8:
        color = "green"
    elif confidence > 0.6:
        color = "orange"
    else:
        color = "red"

    st.markdown(f"<p style='text-align:center; color:{color}; font-size:20px;'>Confidence: {confidence*100:.2f}%</p>", unsafe_allow_html=True)
    st.progress(float(confidence))

    # Show all class probabilities
    st.markdown("### 📊 Class Probabilities")
    st.table({
        "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "Probability (%)": [f"{prob*100:.2f}" for prob in probabilities]
    })

# -------------------------
# Dataset preview
# -------------------------
st.divider()
with st.expander("📂 View Dataset Preview"):
    st.markdown("### 🌼 Iris Dataset Sample")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### 📊 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", df.shape[0])
    col2.metric("Features", 4)
    col3.metric("Classes", df["species"].nunique())

    st.caption(
        "Dataset loaded from sklearn for demo purposes. Model trained on same feature structure."
    )
