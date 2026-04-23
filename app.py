import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="CVD Risk Assessment",
    page_icon="CVD",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    rf_model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return lr_model, rf_model, scaler

lr_model, rf_model, scaler = load_models()

# =========================
# FEATURE ORDER
# =========================

FEATURE_NAMES = [
    '_AGEG5YR', '_SEX', 'SMOKE100', 'SMOKDAY2', 'EXERANY2', 'ALCDAY4',
    'DIABETE4', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', '_BMI5', 'WEIGHT2',
    'HEIGHT3', 'CHECKUP1', 'MEDCOST1', '_TOTINDA'
]

# =========================
# FUNCTIONS
# =========================

def calculate_bmi(weight, height):
    height_m = height / 100
    return weight / (height_m ** 2)

def get_risk_category(prob):
    if prob < 0.15:
        return "Low Risk"
    elif prob < 0.30:
        return "Moderate Risk"
    else:
        return "High Risk"

def predict(features):
    df = pd.DataFrame([features])
    #add missing columns with default value: 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_NAMES]  # enforce correct order
    
    #scale
    scaled = scaler.transform(df)

    lr = lr_model.predict_proba(scaled)[0][1]
    rf = rf_model.predict_proba(scaled)[0][1]

    return lr, rf

# =========================
# UI
# =========================

st.title("Cardiovascular Risk Assessment Tool")
st.write("Enter your details to estimate cardiovascular risk.")

tab1, tab2, tab3 = st.tabs(["Assessment", "Results", "About"])

# =========================
# TAB 1 - INPUT
# =========================

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        age_map = {
    "18-24": 1,
    "25-29": 2,
    "30-34": 3,
    "35-39": 4,
    "40-44": 5,
    "45-49": 6,
    "50-54": 7,
    "55-59": 8,
    "60-64": 9,
    "65-69": 10,
    "70-74": 11,
    "75-79": 12,
    "80+": 13
}

        age_label = st.selectbox("Age Group", list(age_map.keys()))
        age = age_map[age_label]
        sex = st.radio("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

    with col2:
        weight = st.number_input("Weight (kg)", 30, 300, 70)
        height = st.number_input("Height (cm)", 100, 250, 170)
        bmi = calculate_bmi(weight, height)
        st.write(f"BMI: {bmi:.1f}")

    col3, col4 = st.columns(2)

    with col3:
        smoke100 = st.radio("Smoked 100+ cigarettes?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
        smoking_map = {
            "Every day": 1,
            "Some days": 2,
            "Not at all": 3
        }
        smoking_label = st.selectbox("Current smoking frequency", list(smoking_map.keys()))
        smokday2 = smoking_map[smoking_label]
        exerany2 = st.radio("Regular exercise?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

    with col4:
        diabete4 = st.radio("Diabetes?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
        genhlth = st.slider("General health (1=Excellent, 5=Poor)", 1, 5, 3)
        physhlth = st.slider("Physical unhealthy days", 0, 30, 0)

    col5, col6 = st.columns(2)

    with col5:
        menthlth = st.slider("Mental unhealthy days", 0, 30, 0)
        alcday4 = st.number_input("Alcohol days/week", 0, 7, 2)

    with col6:
        checkup_map = {
            "Within the past year": 1,
            "1-2 years ago": 2,
            "2-5 years ago": 3,
            "More than 5 years ago": 4,
            "Never": 5
        }

        checkup_label = st.selectbox("Time since last medical checkup", list(checkup_map.keys()))
        checkup1 = checkup_map[checkup_label]
        medcost1 = st.radio("Could not afford care?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

    activity_map = {
        "Inactive (no regular activity)": 1,
        "Moderately active": 2,
        "Highly active": 3
    }

    activity_label = st.selectbox("Physical activity level", list(activity_map.keys()))
    totinda = activity_map[activity_label]

    features = {
        '_AGEG5YR': age,
        '_SEX': sex,
        'SMOKE100': smoke100,
        'SMOKDAY2': smokday2,
        'EXERANY2': exerany2,
        'ALCDAY4': alcday4,
        'DIABETE4': diabete4,
        'GENHLTH': genhlth,
        'PHYSHLTH': physhlth,
        'MENTHLTH': menthlth,
        '_BMI5': bmi,
        'WEIGHT2': weight,
        'HEIGHT3': height,
        'CHECKUP1': checkup1,
        'MEDCOST1': medcost1,
        '_TOTINDA': totinda
    }

    if st.button("Assess Risk"):
        st.session_state.features = features
        st.session_state.run = True

# =========================
# TAB 2 - RESULTS
# =========================

with tab2:

    if 'run' in st.session_state:

        lr, rf = predict(st.session_state.features)
        avg = (lr + rf) / 2
        category = get_risk_category(avg)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg * 100,
            title={'text': "Overall Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 15], 'color': "lightgreen"},
                    {'range': [15, 30], 'color': "orange"},
                    {'range': [30, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)
        col1, col2, col3 = st.columns(3)

        col1.metric("Logistic Regression", f"{lr*100:.1f}%")
        col2.metric("Random Forest", f"{rf*100:.1f}%")
        col3.metric("Average Risk", f"{avg*100:.1f}%")
        st.subheader("Risk Level")
        st.write(f"Estimated category: {category}")

        st.subheader("Recommendations")
        recommendations = []

        if st.session_state.features['SMOKE100'] == 1:
            recommendations.append("Consider quitting smoking to reduce cardiovascular risk.")

        if st.session_state.features['_BMI5'] > 25:
            recommendations.append("Weight management may help reduce your risk.")

        if st.session_state.features['_TOTINDA'] == 1:
            recommendations.append("Increase physical activity (at least 150 minutes per week).")

        if st.session_state.features['GENHLTH'] >= 3:
            recommendations.append("Consider consulting a healthcare provider for a full checkup.")

        if st.session_state.features['DIABETE4'] == 1:
            recommendations.append("Proper diabetes management is important for heart health.")

        if len(recommendations) == 0:
            st.write("No major risk factors identified. Continue maintaining a healthy lifestyle.")
        else:
            for rec in recommendations:
                st.write(f"- {rec}")
      
        st.subheader("Key Inputs")
        st.table(pd.DataFrame(st.session_state.features.items(), columns=["Feature", "Value"]))
        
    else:
        st.write("Enter details in the Assessment tab and click Assess Risk.")

# =========================
# TAB 3 - ABOUT
# =========================

with tab3:
    st.write("This tool estimates cardiovascular risk based on behavioral and health data.")
    st.write("It is intended for educational purposes and not as a medical diagnosis tool.")