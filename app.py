import streamlit as st
import numpy as np
import joblib

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Cardiovascular Risk Predictor", page_icon="🫀")

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.write("Enter your health details below:")
st.divider()

gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
age = st.number_input("Age (years)", min_value=1, max_value=100, value=45)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_hi = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=200, value=80)

cholesterol = st.selectbox("Cholesterol Level", [1,2,3],
              format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x])
gluc = st.selectbox("Glucose Level", [1,2,3],
       format_func=lambda x: {1:"Normal", 2:"Above Normal", 3:"Well Above Normal"}[x])
smoke  = st.selectbox("Do you smoke?", [0,1], format_func=lambda x: "Yes" if x else "No")
alco   = st.selectbox("Alcohol intake?", [0,1], format_func=lambda x: "Yes" if x else "No")
active = st.selectbox("Physically active?", [0,1], format_func=lambda x: "Yes" if x else "No")

st.divider()
if st.button("🔍 Predict Risk", use_container_width=True):
    age_days = age * 365.25
    bmi = weight / (height/100)**2
    input_data = np.array([[1, age*365.25, gender, height, weight, ap_hi, ap_lo,
                        cholesterol, gluc, smoke, alco, active, bmi]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk — {probability[1]*100:.1f}% probability of cardiovascular disease")
        st.write("Please consult a medical professional.")
    else:
        st.success(f"✅ Low Risk — {probability[0]*100:.1f}% probability of no cardiovascular disease")
        st.write("Keep maintaining a healthy lifestyle!")
