# prompt: Buatkan streamlit sesuai model simulation saya tidak menggunakan tensorflow

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the trained model and preprocessors
model = joblib.load('clothes_price_prediction_model.joblib')
scaler = joblib.load('scaler.pkl')

# Load the original DataFrame (for feature names and order)
df = pd.read_csv('/content/clothes-price-prediction/clothes_price_prediction_data.csv')


# Function to preprocess user input
def preprocess_input(input_data, df, scaler):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data], columns=['Brand', 'Category', 'Color', 'Material', 'Size'])

    # Perform Label Encoding for categorical columns
    categorical_cols = ['Brand', 'Category', 'Color', 'Material', 'Size']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col])  # Fit the encoder on the original data
        input_df[col] = le.transform(input_df[col])
        
    # Handle missing columns
    missing_cols = set(df.drop('Price',axis=1).columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    
    input_df = input_df[df.drop('Price',axis=1).columns]
    
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    return scaled_input


# Streamlit app
st.title("Clothes Price Prediction")

# Input fields for user
brand = st.selectbox("Brand", df['Brand'].unique())
category = st.selectbox("Category", df['Category'].unique())
color = st.selectbox("Color", df['Color'].unique())
material = st.selectbox("Material", df['Material'].unique())
size = st.selectbox("Size", df['Size'].unique())


# Prediction button
if st.button("Predict Price"):
    try:
        input_data = [brand, category, color, material, size]
        scaled_input = preprocess_input(input_data, df, scaler)
        prediction = model.predict(scaled_input)
        st.success(f"Predicted Price: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
