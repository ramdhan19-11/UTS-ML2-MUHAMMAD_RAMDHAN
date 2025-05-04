import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('clothes_price_prediction_model.h5')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Load the Label Encoder (if needed)
# label_encoder = joblib.load('label_encoder.pkl')

# Create input fields for user
st.title("Clothes Price Prediction")

Brand = st.number_input("Brand (Encoded Value)", min_value=0, value=0, step=1)
Category = st.number_input("Category (Encoded Value)", min_value=0, value=0, step=1)
Color = st.number_input("Color (Encoded Value)", min_value=0, value=0, step=1)
Material = st.number_input("Material (Encoded Value)", min_value=0, value=0, step=1)
Size = st.number_input("Size (Encoded Value)", min_value=0, value=0, step=1)


# Create a button to trigger prediction
if st.button("Predict Price"):
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'Brand': [Brand],
        'Category': [Category],
        'Color': [Color],
        'Material': [Material],
        'Size': [Size]
    })


    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)


    # Display the prediction
    st.write(f"Predicted Price: {prediction[0][0]}")
