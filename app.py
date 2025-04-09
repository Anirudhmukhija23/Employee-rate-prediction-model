
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import joblib

# Load the trained model
try:
    gb = joblib.load("ATTRITION.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Web app interface
img = Image.open("data.jpeg")
st.image(img, width=650)
st.title("Employee Attrition Rate Model")

# Input fields
Age = st.number_input("Age", min_value=18, max_value=100, step=1)
DailyRate = st.number_input("Daily Rate", min_value=0, step=1)
BusinessTravel = st.selectbox("Business Travel", [0, 1])  # 0: Rarely, 1: Frequently
Department = st.number_input("Department (Numeric Encoding)", min_value=0, step=1)
EducationField = st.number_input("Education Field (Numeric Encoding)", min_value=0, step=1)
TotalWorkingYears = st.number_input("Total Working Years", min_value=0, step=1)
YearsAtCompany = st.number_input("Years at Company", min_value=0, step=1)
YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, step=1)

# Submit button
if st.button("Submit"):
    input_features = np.array([
        Age, DailyRate, BusinessTravel, Department, 
        EducationField, TotalWorkingYears, YearsAtCompany, YearsSinceLastPromotion
    ]).reshape(1, -1)

    try:
        # Make prediction
        prediction = gb.predict(input_features)

        # Display result
        if prediction[0] == 1:
            st.success("✅ Attrition is likely.")
        else:
            st.error("❌ Attrition is unlikely.")
    except Exception as e:
        st.error(f"Prediction error: {e}")

