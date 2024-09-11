import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load model
diabet_loaded_rf_model = joblib.load("diabet_rf_model.pkl")

# Load the column names used during training (expecting 8 features)
with open("diabet_training_columns.pkl", "rb") as f:
    diabet_training_columns = joblib.load(f)

# Define the input form for diabetes prediction
st.title("Diabetes Prediction")

# Collect input data based on your diabetes dataset columns
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=0)

# Create a DataFrame from the input data
diabet_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [diabetes_pedigree],
    "Age": [age]
})

# Yeni özellikler oluşturma
diabet_data["NEW_AGE_CAT"] = np.where((diabet_data["Age"] >= 50), "senior", "mature")
diabet_data["NEW_BMI"] = pd.cut(x=diabet_data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])
diabet_data["NEW_GLUCOSE"] = pd.cut(x=diabet_data["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])
diabet_data["NEW_INSULIN_SCORE"] = diabet_data.apply(lambda row: "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal", axis=1)
diabet_data["NEW_GLUCOSE*Insulin"] = diabet_data["Glucose"] * diabet_data["Insulin"]
diabet_data["NEW_GLUCOSE*Pregnancies"] = diabet_data["Glucose"] * diabet_data["Pregnancies"]

diabet_data = pd.get_dummies(diabet_data)

# Align with training columns
diabet_data = diabet_data.reindex(columns=diabet_training_columns, fill_value=0)

# Ensure the input data aligns with the model's expected features
diabet_data = diabet_data[diabet_training_columns]  # Use only the expected columns,

# Print columns for debugging
st.write("Data Columns:", diabet_data.columns.tolist())
st.write("Training Columns:", diabet_training_columns)

# Predict using the Random Forest model
if st.button("Predict Diabetes"):
    prediction_rf = diabet_loaded_rf_model.predict(diabet_data)

    st.write(f"Random Forest Prediction: {'Diabetic' if prediction_rf[0] else 'Not Diabetic'}")

#"['PREGNANCIES', 'GLUCOSE', 'BLOODPRESSURE', 'SKINTHICKNESS', 'INSULIN', 'DIABETESPEDIGREEFUNCTION', 'AGE', 'OUTCOME', 'NEW_AGE_CAT', 'NEW_INSULIN_SCORE', 'NEW_GLUCOSE*INSULIN', 'NEW_GLUCOSE*PREGNANCIES', 'NEW_AGE_BMI_NOM_obesesenior', 'NEW_AGE_BMI_NOM_underweightmature', 'NEW_AGE_GLUCOSE_NOM_hiddensenior', 'NEW_AGE_GLUCOSE_NOM_highmature', 'NEW_AGE_GLUCOSE_NOM_highsenior', 'NEW_AGE_GLUCOSE_NOM_lowmature', 'NEW_AGE_GLUCOSE_NOM_lowsenior', 'NEW_AGE_GLUCOSE_NOM_normalmature', 'NEW_AGE_GLUCOSE_NOM_normalsenior', 'NEW_BMI_Healthy', 'NEW_BMI_Overweight', 'NEW_BMI_Obese', 'NEW_GLUCOSE_Prediabetes', 'NEW_GLUCOSE_Diabetes'] not in index"
