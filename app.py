import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="CVD Risk Screening",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d62728;
        margin-bottom: 1rem;
    }
    .risk-high { color: #d62728; font-weight: bold; }
    .risk-medium { color: #ff7f0e; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load pre-trained models"""
    try:
        with open('models/logistic_regression_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return lr_model, rf_model, scaler
    except FileNotFoundError:
        st.error("Models not found. Please train models first.")
        return None, None, None

def get_risk_category(probability):
    """Categorize risk level"""
    if probability < 0.2:
        return "Low Risk", "risk-low"
    elif probability < 0.5:
        return "Moderate Risk", "risk-medium"
    else:
        return "High Risk", "risk-high"

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page:", 
                            ["Home", "Risk Assessment", "About", "FAQ"])
    
    if page == "Home":
        show_home()
    elif page == "Risk Assessment":
        show_risk_assessment()
    elif page == "About":
        show_about()
    elif page == "FAQ":
        show_faq()

def show_home():
    """Home page"""
    st.markdown('<p class="main-header">🫀 Cardiovascular Risk Screening for Kenyans</p>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### About This Tool
        
        Cardiovascular disease (CVD) is the leading cause of death globally and a rapidly 
        growing crisis in Kenya.
        
        **Key Statistics:**
        - CVD accounts for ~30% of all deaths in Kenya
        - Driven by modifiable risk factors: tobacco use, physical inactivity, obesity, 
          diabetes, hypertension
        - Most Kenyans interact with health system only when acutely ill
        
        This tool provides a data-driven screening method to identify individuals at risk 
        of cardiovascular disease early.
        """)
    
    with col2:
        st.info("""
        ℹ️ **DISCLAIMER**
        
        This tool is for RESEARCH and EDUCATION purposes only.
        
        ⚠️ NOT validated for clinical use in Kenya
        ⚠️ NOT a replacement for medical judgment
        ⚠️ Should NOT be used without confirmation from a healthcare provider
        """)
    
    st.markdown("---")
    st.subheader("Getting Started")
    st.markdown("""
    1. Navigate to **Risk Assessment** in the sidebar
    2. Enter your health information
    3. Get your personalized risk score
    4. Receive recommendations for next steps
    """)

def show_risk_assessment():
    """Risk assessment page"""
    st.markdown('<p class="main-header">📋 Cardiovascular Risk Assessment</p>', 
                unsafe_allow_html=True)
    
    lr_model, rf_model, scaler = load_models()
    
    if lr_model is None or rf_model is None:
        st.error("Models not available. Please ensure models are trained and saved.")
        return
    
    # Create input form
    with st.form("risk_assessment_form"):
        st.subheader("Please provide your health information:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age_group = st.selectbox(
                "Age Group",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                help="Select your age group (1=18-24, 13=80+)"
            )
            
            sex = st.radio(
                "Biological Sex",
                options=[1, 2],
                format_func=lambda x: "Male" if x == 1 else "Female"
            )
            
            smoke100 = st.radio(
                "Have you smoked 100+ cigarettes in your lifetime?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            
            smokday = st.selectbox(
                "Do you currently smoke?",
                options=[1, 2, 3],
                format_func=lambda x: ["Every day", "Some days", "Not at all"][x-1]
            )
            
            exercise = st.radio(
                "Do you exercise ≥30 min, ≥5 days/week?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        with col2:
            alcohol = st.number_input(
                "Days per week with alcohol consumption",
                min_value=0, max_value=7, step=1
            )
            
            diabetes = st.radio(
                "Have you been diagnosed with diabetes?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
            
            gen_health = st.selectbox(
                "General Health Status",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x-1]
            )
            
            phys_health = st.number_input(
                "Days physical health was not good (0-30)",
                min_value=0, max_value=30, step=1
            )
            
            ment_health = st.number_input(
                "Days mental health was not good (0-30)",
                min_value=0, max_value=30, step=1
            )
        
        # BMI calculation
        st.subheader("Body Measurements")
        col3, col4 = st.columns(2)
        
        with col3:
            height_cm = st.number_input(
                "Height (cm)",
                min_value=100, max_value=250, step=1
            )
        
        with col4:
            weight_kg = st.number_input(
                "Weight (kg)",
                min_value=30, max_value=200, step=1
            )
        
        # Calculate BMI
        bmi = weight_kg / ((height_cm / 100) ** 2)
        st.write(f"Your BMI: **{bmi:.1f}**")
        
        # Additional inputs
        col5, col6 = st.columns(2)
        
        with col5:
            checkup = st.selectbox(
                "Last routine checkup",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: ["< 1 year ago", "1-2 years ago", "2-5 years ago", 
                                       "> 5 years ago", "Never"][x-1]
            )
        
        with col6:
            medcost = st.radio(
                "Could not afford medical care?",
                options=[1, 2],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )
        
        # Activity level (derived)
        activity = 1 if exercise == 1 else 2
        
        # Create input array
        input_data = np.array([[
            age_group, sex, smoke100, smokday, exercise, alcohol,
            diabetes, gen_health, phys_health, ment_health, bmi,
            weight_kg, height_cm, checkup, medcost, activity
        ]])
        
        submit_button = st.form_submit_button("🔍 Calculate Risk Score")
    
    if submit_button:
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Get predictions from both models
        lr_prob = lr_model.predict_proba(input_scaled)[0, 1]
        rf_prob = rf_model.predict_proba(input_scaled)[0, 1]
        
        # Average predictions
        avg_prob = (lr_prob + rf_prob) / 2
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Your Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_label, risk_class = get_risk_category(avg_prob)
            st.markdown(f'<p class="{risk_class}">{risk_label}</p>', 
                       unsafe_allow_html=True)
            st.metric("Risk Score", f"{avg_prob:.1%}")
        
        with col2:
            st.metric("Logistic Regression", f"{lr_prob:.1%}")
            st.caption("Score from Logistic Regression model")
        
        with col3:
            st.metric("Random Forest", f"{rf_prob:.1%}")
            st.caption("Score from Random Forest model")
        
        # Provide recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        recommendations = []
        
        if smoke100 == 1 or smokday != 3:
            recommendations.append("🚭 **Smoking Cessation**: Quitting smoking significantly reduces CVD risk")
        
        if exercise == 2:
            recommendations.append("🏃 **Physical Activity**: Aim for at least 150 minutes of moderate activity per week")
        
        if bmi > 25:
            recommendations.append(f"⚖️ **Weight Management**: Your BMI is {bmi:.1f}. Consider weight reduction strategies")
        
        if diabetes == 1:
            recommendations.append("🩺 **Diabetes Management**: Regular monitoring and management is critical")
        
        if gen_health >= 4:
            recommendations.append("👨‍⚕️ **Medical Consultation**: Schedule a check-up with your healthcare provider")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ Keep up the healthy lifestyle!")
        
        # Next steps
        st.markdown("---")
        st.subheader("🔄 Next Steps")
        st.markdown("""
        1. **Share with Healthcare Provider**: Take this assessment to your doctor
        2. **Get Confirmed Testing**: Request blood pressure, cholesterol, and glucose tests
        3. **Develop Action Plan**: Work with your provider on behavior change strategies
        4. **Follow-up**: Reassess your risk annually or after major lifestyle changes
        """)

def show_about():
    """About page"""
    st.markdown('<p class="main-header">ℹ️ About This Project</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Project Overview
    
    This project develops a data-driven cardiovascular risk screening tool tailored for 
    Kenyans using machine learning techniques.
    
    ### Data Source
    
    **CDC BRFSS 2024 Dataset**
    - 457,670 adult respondents
    - Self-reported behavioral and clinical factors
    - No invasive tests required
    
    ### Methodology
    
    **Machine Learning Models:**
    - Logistic Regression
    - Random Forest Classifier
    - SMOTE for handling class imbalance
    
    **Risk Factors Included:**
    - Demographic: Age, Sex
    - Behavioral: Smoking, Physical Activity, Alcohol Use
    - Clinical: Diabetes, General Health, BMI
    - Healthcare Access: Routine Checkups, Medical Cost Barriers
    
    ### Team
    This is a group project by the team at Vibora-Polo.
    
    ### Citation
    CDC BRFSS: https://www.cdc.gov/brfss/
    
    ### Important Note
    ⚠️ This tool uses U.S. data as a proof-of-concept and should be recalibrated 
    with Kenyan population data before clinical deployment.
    """)

def show_faq():
    """FAQ page"""
    st.markdown('<p class="main-header">❓ Frequently Asked Questions</p>', 
                unsafe_allow_html=True)
    
    faqs = {
        "How accurate is this tool?": 
            "The model achieves 80% accuracy and 0.80 ROC-AUC on test data. However, this "
            "is based on U.S. data and should be validated on Kenyan populations.",
        
        "Can this replace a doctor's diagnosis?":
            "No. This is a screening tool only. Always consult with a healthcare provider "
            "for proper diagnosis and treatment.",
        
        "What does the risk score mean?":
            "The score (0-100%) represents the probability of having cardiovascular disease "
            "based on your entered information.",
        
        "Why use U.S. data?":
            "Kenya doesn't have a large-scale CVD survey yet. U.S. data is used for "
            "proof-of-concept. Future versions will use Kenyan data.",
        
        "What are the main CVD risk factors?":
            "Smoking, physical inactivity, obesity (high BMI), diabetes, hypertension, "
            "and high cholesterol.",
        
        "Can this predict future CVD?":
            "This assesses current risk based on reported factors. It's not a longitudinal "
            "prediction but a current risk snapshot.",
    }
    
    for question, answer in faqs.items():
        with st.expander(f"**{question}**"):
            st.write(answer)

if __name__ == "__main__":
    main()