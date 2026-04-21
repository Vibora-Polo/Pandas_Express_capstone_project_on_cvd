import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CVD Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS & SCALER
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models and scaler"""
    try:
        lr_model = joblib.load('models/logistic_regression_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return lr_model, rf_model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please ensure models are in the 'models/' directory.")
        st.stop()

lr_model, rf_model, scaler = load_models()

# Feature names (must match training data)
FEATURE_NAMES = [
    '_AGEG5YR', '_SEX', 'SMOKE100', 'SMOKDAY2', 'EXERANY2', 'ALCDAY4',
    'DIABETE4', 'GENHLTH', 'PHYSHLTH', 'MENTHLTH', '_BMI5', 'WEIGHT2',
    'HEIGHT3', 'CHECKUP1', 'MEDCOST1', '_TOTINDA'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight and height"""
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_risk_category(probability):
    """Categorize risk level"""
    if probability < 0.15:
        return "Low Risk", "green"
    elif probability < 0.30:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def predict_risk(features_dict):
    """Make predictions with both models"""
    # Convert to dataframe and scale
    features_df = pd.DataFrame([features_dict])
    features_scaled = scaler.transform(features_df)
    
    # Get predictions
    lr_pred = lr_model.predict_proba(features_scaled)[0, 1]
    rf_pred = rf_model.predict_proba(features_scaled)[0, 1]
    
    return lr_pred, rf_pred

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("❤️ Cardiovascular Risk Assessment Tool")
    st.markdown("""
    This tool helps assess your cardiovascular risk based on behavioral and 
    health factors. **Please note:** This is a screening tool only and should 
    not replace professional medical advice.
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Results", "About"])
    
    with tab1:
        st.header("Personal Information & Risk Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            age_group = st.selectbox(
                "Age Group",
                options=range(1, 14),
                help="Select your age category (1-13, where higher = older)"
            )
            sex = st.radio("Biological Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        
        with col2:
            st.subheader("Body Measurements")
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            bmi = calculate_bmi(weight_kg, height_cm)
            st.metric("Calculated BMI", f"{bmi:.1f}")
        
        st.divider()
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Lifestyle Factors")
            smoke_100 = st.radio(
                "Have you smoked 100+ cigarettes in your lifetime?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            smokday2 = st.selectbox(
                "Current smoking frequency",
                options=[1, 2, 3],
                format_func=lambda x: ["Every day", "Some days", "Not at all"][x-1]
            )
            exerany2 = st.radio(
                "Exercise regularly (≥30 min, ≥5 days/week)?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        with col4:
            st.subheader("Health Conditions")
            diabete4 = st.radio(
                "Doctor told you have diabetes?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            genhlth = st.slider(
                "General health rating (1=Excellent, 5=Poor)",
                min_value=1, max_value=5, value=3
            )
            physhlth = st.slider(
                "Days physical health not good (0-30)",
                min_value=0, max_value=30, value=0
            )
        
        st.divider()
        
        col5, col6 = st.columns(2)
        
        with col5:
            menthlth = st.slider(
                "Days mental health not good (0-30)",
                min_value=0, max_value=30, value=0
            )
            alcday4 = st.number_input(
                "Alcohol consumption days/week (0-7)",
                min_value=0, max_value=7, value=2
            )
        
        with col6:
            checkup1 = st.selectbox(
                "Time since routine checkup",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: [
                    "Within 1 year",
                    "1-2 years",
                    "2-5 years",
                    "5+ years",
                    "Never"
                ][x-1]
            )
            medcost1 = st.radio(
                "Couldn't afford medical care?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        totinda = st.selectbox(
            "Overall physical activity level",
            options=[1, 2, 3],
            format_func=lambda x: ["Insufficient", "Sufficient", "High"][x-1]
        )
        
        # Prepare features
        features = {
            '_AGEG5YR': age_group,
            '_SEX': sex,
            'SMOKE100': smoke_100,
            'SMOKDAY2': smokday2,
            'EXERANY2': exerany2,
            'ALCDAY4': alcday4,
            'DIABETE4': diabete4,
            'GENHLTH': genhlth,
            'PHYSHLTH': physhlth,
            'MENTHLTH': menthlth,
            '_BMI5': bmi,
            'WEIGHT2': weight_kg,
            'HEIGHT3': height_cm,
            'CHECKUP1': checkup1,
            'MEDCOST1': medcost1,
            '_TOTINDA': totinda
        }
        
        if st.button("🔍 Assess Risk", use_container_width=True):
            st.session_state.features = features
            st.session_state.predict = True
    
    with tab2:
        if 'predict' in st.session_state and st.session_state.predict:
            st.header("Risk Assessment Results")
            
            # Get predictions
            lr_prob, rf_prob = predict_risk(st.session_state.features)
            avg_prob = (lr_prob + rf_prob) / 2
            
            category, color = get_risk_category(avg_prob)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Logistic Regression", f"{lr_prob*100:.1f}%")
            with col2:
                st.metric("Random Forest", f"{rf_prob*100:.1f}%")
            with col3:
                st.metric("Average Risk", f"{avg_prob*100:.1f}%", delta=category)
            
            # Risk category with color
            st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">{category}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk interpretation
            st.subheader("What This Means")
            if avg_prob < 0.15:
                st.success("""
                Your estimated risk is **low**. Continue with healthy lifestyle practices:
                - Regular exercise (150+ min/week)
                - Balanced diet
                - Stress management
                - Regular health checkups
                """)
            elif avg_prob < 0.30:
                st.warning("""
                Your estimated risk is **moderate**. Consider:
                - Increased focus on risk factors
                - Consulting with healthcare provider
                - Dietary modifications
                - Stress reduction techniques
                """)
            else:
                st.error("""
                Your estimated risk is **high**. 
                **⚠️ Please consult with a healthcare professional immediately for:**
                - Proper evaluation and testing
                - Personalized intervention plan
                - Possible medications
                - Intensive lifestyle changes
                """)
            
            # Feature importance (simplified)
            st.subheader("Key Risk Drivers in Your Profile")
            top_features = {
                'Age': st.session_state.features['_AGEG5YR'],
                'General Health': st.session_state.features['GENHLTH'],
                'BMI': f"{st.session_state.features['_BMI5']:.1f}",
                'Diabetes': st.session_state.features['DIABETE4'],
                'Physical Activity': st.session_state.features['_TOTINDA']
            }
            
            st.json(top_features)
            
            st.divider()
            st.info("""
            **Disclaimer:** This tool is for educational purposes only. It is not a substitute 
            for professional medical diagnosis. Please consult with a healthcare provider 
            for proper evaluation and treatment.
            """)
        else:
            st.info("👈 Please fill out the form and click 'Assess Risk' to see results.")
    
    with tab3:
        st.header("About This Tool")
        
        st.subheader("📊 Data Source")
        st.write("""
        This tool is built using the CDC BRFSS (Behavioral Risk Factor Surveillance System) 
        2024 dataset with 457,670 respondents. It represents a proof-of-concept for screening 
        cardiovascular disease risk using self-reported behavioral and health factors.
        """)
        
        st.subheader("🔬 Methodology")
        st.write("""
        - **Target:** Heart attack risk (self-reported)
        - **Models Used:** Logistic Regression & Random Forest
        - **Features:** 16 behavioral and clinical indicators
        - **Class Balance:** SMOTE applied to handle imbalance
        """)
        
        st.subheader("⚠️ Important Limitations")
        st.write("""
        1. **US Data Only:** Built on US population data; may not apply to other regions
        2. **Self-Reported:** Based on self-reported conditions, not clinical diagnosis
        3. **No Biomarkers:** Doesn't include blood pressure, cholesterol, or ECG
        4. **Research Use:** Not validated for clinical deployment
        5. **Not a Diagnosis:** Cannot replace professional medical evaluation
        """)
        
        st.subheader("✅ When to See a Doctor")
        st.error("""
        - Chest pain or pressure
        - Shortness of breath
        - Family history of heart disease
        - Sudden onset of symptoms
        - High-risk assessment from this tool
        """)
        
        st.subheader("📚 References")
        st.write("""
        - CDC BRFSS: https://www.cdc.gov/brfss/
        - Kenya STEPwise Survey: https://www.who.int/teams/noncommunicable-diseases/monitoring-and-surveillance
        - Feature Importance: Age, General Health, BMI, Diabetes Status
        """)

if __name__ == "__main__":
    main()