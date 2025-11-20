import streamlit as st
import pandas as pd
import joblib
import time

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# 1. মডেল লোড
model = joblib.load('xgboost_heart_failure_model.pkl')

# 2. পেজ শিরোনাম
st.title("❤️ Heart Disease Prediction")
st.write("ইনপুট দিন এবং দেখুন আপনার হার্ট ফেইলিউর হওয়ার সম্ভাবনা!")
st.markdown("""
    <style>
    .pulse-title h1 {display:inline-block; animation:pulse 1.5s ease-in-out infinite;}
    @keyframes pulse {0%{transform:scale(1)}50%{transform:scale(1.06)}100%{transform:scale(1)}}
    </style>
""", unsafe_allow_html=True)
st.caption("শুধুমাত্র শিক্ষামূলক উদ্দেশ্যে — চিকিৎসা পরামর্শের জন্য ডাক্তার দেখান।")
st.sidebar.title("ℹ️ About")
st.sidebar.write("এই ডেমো একটি ট্রেইনড মডেল ব্যবহার করে হার্ট ডিজিজ রিস্ক অনুমান করে।")
st.sidebar.write("মেডিক্যাল সিদ্ধান্তের জন্য পেশাদারের সাথে পরামর্শ করুন।")

# 3. ইউজার ইনপুট
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("Fasting BS > 120 mg/dl", [0, 1])
with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=202, value=130)
    exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=0.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
st.caption("টিপ: ইনপুট পরিবর্তন করলে প্রেডিকশন আপডেট দেখতে বাটনে ক্লিক করুন।")

# 4. ইনপুট ডেটা ফ্রেম তৈরি
input_data = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': chest_pain,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingECG': resting_ecg,
    'MaxHR': max_hr,
    'ExerciseAngina': exercise_angina,
    'Oldpeak': oldpeak,
    'ST_Slope': st_slope
}

input_df = pd.DataFrame([input_data])
with st.expander("Input Summary"):
    st.dataframe(input_df, use_container_width=True)

# 5. প্রেডিকশন
if st.button("Predict"):
    with st.spinner("Analyzing risk..."):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)
        risk = float(proba[0][1])
        risk_pct = int(risk * 100)

        prog = st.progress(0)
        for i in range(0, risk_pct + 1, 5):
            prog.progress(i)
            time.sleep(0.02)

    if prediction[0] == 1:
        st.error("❌ Predicted: Heart Disease (Class 1)")
        st.info(f"Confidence: {risk * 100:.2f}%")
        st.snow()
        risk_label = "High Risk" if risk_pct >= 70 else ("Moderate Risk" if risk_pct >= 40 else "Low Risk")
        st.warning(f"Risk Level: {risk_label}")
    else:
        st.success("✅ Predicted: No Heart Disease (Class 0)")
        st.info(f"Confidence: {(1 - risk) * 100:.2f}%")
        st.balloons()
        st.info("Keep up a heart-healthy lifestyle!")

    m1, m2 = st.columns(2)
    m1.metric("Risk Probability", f"{risk * 100:.1f}%")
    m2.metric("Max Heart Rate", f"{max_hr} bpm")