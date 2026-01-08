import joblib
import streamlit as st

# Load the trained model
model = joblib.load('iris_logistic_regression_model.pkl')

# Define the Streamlit app
st.title("Iris Flower Species Prediction")
st.write("Enter the features of the Iris flower to predict its species.")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict Species"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    species = prediction[0]
    if species == 0:
        species = "Iris-setosa"
    elif species == 1:
        species = "Iris-versicolor"
    else:
        species = "Iris-virginica"

    
    st.write(f"The predicted species is: {species}")


