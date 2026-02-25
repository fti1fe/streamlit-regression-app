# app.py
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('linear_regression_model.pkl')

st.title('Auto MPG Predictor')

cylinders    = st.slider('Cylinders', 3, 8, 4)
displacement = st.number_input('Displacement', 50.0, 500.0, 150.0)
horsepower   = st.number_input('Horsepower', 40.0, 250.0, 100.0)
weight       = st.number_input('Weight', 1500.0, 5000.0, 2500.0)
acceleration = st.number_input('Acceleration', 8.0, 25.0, 15.0)
model_year   = st.slider('Model Year', 70, 82, 76)
origin       = st.selectbox('Origin', [1, 2, 3])

input_df = pd.DataFrame([[cylinders, displacement, horsepower,
                           weight, acceleration, model_year, origin]],
                         columns=['cylinders','displacement','horsepower',
                                  'weight','acceleration','model_year','origin'])

if st.button('Predict MPG'):
    prediction = model.predict(input_df)[0]
    st.success(f'Predicted MPG: {prediction:.2f}')
