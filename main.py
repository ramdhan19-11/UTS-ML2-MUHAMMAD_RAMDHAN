import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('clothes_price_prediction_model.joblib')
scaler = joblib.load('scaler.pkl')

# Create input fields for user input
st.title("Clothes Price Prediction")

brand = st.number_input("Brand (Encoded Value)", min_value=0)
category = st.number_input("Category (Encoded Value)", min_value=0)
color = st.number_input("Color (Encoded Value)", min_value=0)
material = st.number_input("Material (Encoded Value)", min_value=0)
size = st.number_input("Size (Encoded Value)", min_value=0)

# Create a button to trigger prediction
if st.button("Predict Price"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Category': [category],
        'Color': [color],
        'Material': [material],
        'Size': [size]
    })

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)

    # Make the prediction
    prediction = model.predict(scaled_input)

    # Display the prediction
    st.write(f"Predicted Price: {prediction[0]}")
