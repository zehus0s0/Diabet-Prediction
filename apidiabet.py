import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load model
diabet_loaded_rf_model = joblib.load("diabet_rf_model.pkl")

# Load the column names used during training
with open("diabet_training_columns.pkl", "rb") as f:
    diabet_training_columns = joblib.load(f)

# Streamlit arayüz başlığı
st.title("Diabetes Prediction")

# Kullanıcıdan girişleri al
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age", min_value=0)

# Modelin beklediği tüm özellikleri içeren DataFrame oluştur
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

# Eksik olan 19 özelliği tamamla
diabet_data["NEW_AGE_CAT"] = np.where(diabet_data["Age"] >= 50, "senior", "mature")
diabet_data["NEW_BMI"] = pd.cut(diabet_data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])
diabet_data["NEW_GLUCOSE"] = pd.cut(diabet_data["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])
diabet_data["NEW_INSULIN_SCORE"] = diabet_data.apply(lambda row: "Normal" if 16 <= row["Insulin"] <= 166 else "Abnormal", axis=1)
diabet_data["NEW_GLUCOSE*Insulin"] = diabet_data["Glucose"] * diabet_data["Insulin"]
diabet_data["NEW_GLUCOSE*Pregnancies"] = diabet_data["Glucose"] * diabet_data["Pregnancies"]

# Kategorik değişkenleri one-hot encoding ile sayısala çevir
diabet_data = pd.get_dummies(diabet_data)

# Modelin beklediği sütunlara göre hizala
diabet_data = diabet_data.reindex(columns=diabet_training_columns, fill_value=0)

# Debugging için sütunları yazdır
st.write("Data Columns:", diabet_data.columns.tolist())
st.write("Training Columns:", diabet_training_columns)

# Model tahmini
if st.button("Predict Diabetes"):
    prediction_rf = diabet_loaded_rf_model.predict(diabet_data)
    st.write(f"Random Forest Prediction: {'Diabetic' if prediction_rf[0] == 1 else 'Not Diabetic'}")
