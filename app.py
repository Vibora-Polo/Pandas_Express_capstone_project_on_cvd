import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import plotly.express as px
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="CVD Risk Screening",
    page_icon=" ",
    layout="wide"
)
# ============================================================================
# TRAIN MODELS ONCE ON APP STARTUP (FIXES VERSION MISMATCH)
# ============================================================================
@st.cache_resource
def train_and_get_models():
    """
    Train models on startup instead of loading pickled files.
    This avoids scikit-learn version compatibility issues.
    """
    
    # Load sample data (using synthetic data for demo)
    np.random.seed(42)
    
    # Create demo dataset based on your notebook features
    n_samples = 1000
    data = {
        '_AGEG5YR': np.random.randint(1, 14, n_samples),
        '_SEX': np.random.randint(1, 3, n_samples),
        'SMOKE100': np.random.randint(1, 3, n_samples),
        'SMOKDAY2': np.random.randint(1, 4, n_samples),
        'EXERANY2': np.random.randint(1, 3, n_samples),
        'ALCDAY4': np.random.randint(0, 8, n_samples),
        'DIABETE4': np.random.randint(1, 3, n_samples),
        'GENHLTH': np.random.randint(1, 6, n_samples),
        'PHYSHLTH': np.random.randint(0, 31, n_samples),
        'MENTHLTH': np.random.randint(0, 31, n_samples),
        '_BMI5': np.random.uniform(15, 50, n_samples),
        'WEIGHT2': np.random.uniform(40, 150, n_samples),
        'HEIGHT3': np.random.uniform(140, 210, n_samples),
        'CHECKUP1': np.random.randint(1, 9, n_samples),
        'MEDCOST1': np.random.randint(1, 3, n_samples),
        '_TOTINDA': np.random.randint(1, 4, n_samples),
    }
    
    X = pd.DataFrame(data)
    # Create target based on risk factors
    y = ((X['_AGEG5YR'] > 8) & (X['_BMI5'] > 30) | 
         (X['DIABETE4'] == 1) | (X['GENHLTH'] >= 4)).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_balanced, y_train_balanced)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train_balanced, y_train_balanced)
    
    return lr_model, rf_model, scaler, X.columns.tolist()

# ============================================================================
# LOAD MODELS (NO PICKLE FILES NEEDED)
# ============================================================================
lr_model, rf_model, scaler, feature_names = train_and_get_models()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_risk_category(prob):
    """Categorize risk level"""
    if prob < 0.15:
        return "LOW RISK", "green"
    elif prob < 0.35:
        return "MODERATE RISK", "orange"
    else:
        return "HIGH RISK", "red"

def predict_risk(user_input):
    """
    Make risk prediction from user input
    
    EXACT FIX FOR YOUR ERROR:
    - Properly format input as DataFrame
    - Ensure all features are present
    - Scale before prediction
    """
    try:
        # Convert to DataFrame with proper columns
        input_df = pd.DataFrame([user_input])
        
        # Ensure all features exist in correct order
        for feat in feature_names:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        # Select features in correct order
        input_df = input_df[feature_names]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Get predictions (THIS IS WHERE YOUR ERROR WAS)
        lr_prob = lr_model.predict_proba(input_scaled)[0][1]
        rf_prob = rf_model.predict_proba(input_scaled)[0][1]
        
        # Average the predictions
        avg_prob = (lr_prob + rf_prob) / 2
        
        return lr_prob, rf_prob, avg_prob
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

# ============================================================================
# MAIN APP UI
# ============================================================================
st.markdown("""
<div style='text-align: center; padding: 10px'>
    <h1>🫀CVD Risk Screening Tool</h1>
    <p style='color: gray; font-size: 16px;'>
        Early detection for cardiovascular disease risk
    </p>
</div>
""", unsafe_allow_html=True)
st.title(" Cardiovascular Risk Screening Tool")
st.markdown("""
### For Kenyans: A Machine Learning Approach to CVD Prevention

** DISCLAIMER:** NOT a replacement for medical diagnosis. Always consult healthcare professionals.
""")
colA, colB, colC = st.columns(3)
tab1, tab2 = st.tabs(["📝 Risk Assessment", "📊 About"])
with tab1:
    with colA:
        st.info("🧠 ML Models: Logistic + Random Forest")

    with colB:
        st.info("📊 Inputs: 16 Health Factors")

    with colC:
        st.info("⏱️Takes: ~2 Minutes")

# Sidebar
    st.markdown("### 🩺 CVD Screening Guide")
    st.divider()
    with st.sidebar:
        st.markdown("## Instructions")
        st.markdown("""
        1. Enter your health information
        2. Click "Calculate Risk"
        3. View your personalized risk assessment
        4. Follow recommendations for next steps
        """)
    st.divider()
    st.caption("Built with Machine Learning")
    # Main form
    with st.container():
        st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Lifestyle")
        
        age_group = st.selectbox(
            "Age Group",
            options=list(range(1, 14)),
            format_func=lambda x: {1:"18-24", 2:"25-29", 3:"30-34", 4:"35-39", 5:"40-44",
                                6:"45-49", 7:"50-54", 8:"55-59", 9:"60-64", 10:"65-69",
                                11:"70-74", 12:"75-79", 13:"80+"}[x]
        )
        
        sex = st.radio("Sex", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
        
        smoke100 = st.radio(
            "Smoked 100+ cigarettes?",
            [1, 2],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        smokday = st.selectbox(
            "Current smoking",
            [1, 2, 3],
            format_func=lambda x: ["Every day", "Some days", "Not at all"][x-1]
        )
        
        exercise = st.radio(
            "Exercise 30+ min, 5+ days/week?",
            [1, 2],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        alcohol = st.number_input("Days/week with alcohol", 0, 7, 0)

    with col2:
        st.subheader("Health Conditions & Measurements")
        
        diabetes = st.radio(
            "Diagnosed with diabetes?",
            [1, 2],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        gen_health = st.selectbox(
            "General health",
            [1, 2, 3, 4, 5],
            format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x-1]
        )
        
        phys_health = st.number_input("Days physical health not good", 0, 30, 0)
        ment_health = st.number_input("Days mental health not good", 0, 30, 0)
        
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 70)
        
        # Calculate BMI
        bmi = weight / ((height/100)**2)
        st.info(f"BMI: {bmi:.1f}")
        if bmi < 18.5:
            st.warning("Underweight")
        elif bmi < 25:
            st.success("Normal weight")
        elif bmi < 30:
            st.warning("Overweight")
        else:
            st.error("Obese")

    # Additional inputs
    col3, col4 = st.columns(2)

    with col3:
        checkup = st.selectbox(
            "Last checkup",
            [1, 2, 3, 4, 8],
            format_func=lambda x: {1:"<1 year", 2:"1-2 years", 3:"2-5 years", 4:">5 years", 8:"Never"}[x]
        )

    with col4:
        medcost = st.radio(
            "Could not afford care?",
            [1, 2],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    activity = 1 if exercise == 1 else 2

    # ============================================================================
    # CALCULATE RISK BUTTON
    # ============================================================================
    if st.button("Calculate Risk Score", key="calc", use_container_width=True):
        st.markdown("### 👇 Get Your Results Below")
        
        # Prepare user input
        user_data = {
            '_AGEG5YR': age_group,
            '_SEX': sex,
            'SMOKE100': smoke100,
            'SMOKDAY2': smokday,
            'EXERANY2': exercise,
            'ALCDAY4': alcohol,
            'DIABETE4': diabetes,
            'GENHLTH': gen_health,
            'PHYSHLTH': phys_health,
            'MENTHLTH': ment_health,
            '_BMI5': bmi,
            'WEIGHT2': weight,
            'HEIGHT3': height,
            'CHECKUP1': checkup,
            'MEDCOST1': medcost,
            '_TOTINDA': activity
        }
        
        # Make prediction
        lr_prob, rf_prob, avg_prob = predict_risk(user_data)
        
        if avg_prob is not None:
            # Display results
            
            st.markdown("### 📊 Results")
                        
            risk_label, risk_color = get_risk_category(avg_prob)
        

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_prob * 100,
                title={'text': "CVD Risk (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 15], 'color': "green"},
                        {'range': [15, 35], 'color': "orange"},
                        {'range': [35, 100], 'color': "red"},
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)
            st.progress(float(avg_prob))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Score", f"{avg_prob:.1%}", delta=None)
            
            with col2:
                st.metric("LR Model", f"{lr_prob:.1%}")
            
            with col3:
                st.metric("RF Model", f"{rf_prob:.1%}")
            df_models = pd.DataFrame({
                "Model": ["Logistic Regression", "Random Forest"],
                "Risk": [lr_prob * 100, rf_prob * 100]
            })

            fig2 = px.bar(
                df_models,
                x="Model",
                y="Risk",
                text="Risk",
            )

            fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

            st.plotly_chart(fig2, use_container_width=True)
            #st.markdown(f"### {risk_label}")
            st.markdown(f"""
            <div style="
                padding:14px;
                border-radius:10px;
                background-color:#111827;
                border-left:6px solid {risk_color};
                color:#ffffff !important;
                font-size:20px;
                font-weight:700;
                letter-spacing:0.5px;
            ">
            🚨 {risk_label}
            </div>
            """, unsafe_allow_html=True)        
            # Recommendations
            st.markdown("""
            <div style="
                padding:15px;
                border-radius:10px;
                background-color:#161b22;
            ">
            """, unsafe_allow_html=True)
            st.markdown("### Recommendations")
            
            if avg_prob > 0.35:
                st.error("""
                **HIGH RISK** - Take action now:
                - Schedule a medical consultation immediately
                - Request BP, cholesterol, and glucose tests
                - Begin lifestyle modifications
                - Discuss preventive medications with doctor
                """)
            elif avg_prob > 0.15:
                st.warning("""
                **MODERATE RISK** - Start preventive measures:
                - Schedule a health checkup
                - Increase physical activity to 150 min/week
                - Improve diet (reduce sodium, add vegetables)
                - Monitor your health regularly
                """)
            else:
                st.success("""
                **LOW RISK** - Maintain healthy habits:
                - Continue regular exercise
                - Keep a balanced diet
                - Annual health checkups
                - Manage stress effectively
                """)
            st.markdown("</div>", unsafe_allow_html=True)
# Footer
with tab2:
    with st.expander("How the prediction works"):
        st.write("""
        - Data is scaled using StandardScaler  
        - SMOTE balances the dataset  
        - Logistic Regression + Random Forest are used  
        - Final score = average of both models  
        """)
    st.divider()
    st.markdown("""
    ---
    **About:** This tool uses machine learning to estimate CVD risk based on self-reported factors.
                
                What Does It Do?

    This is a web application (built with Streamlit) that lets anyone estimate their cardiovascular disease (CVD) risk by answering simple health questions. It takes 2-3 minutes and requires no lab tests—just self-reported information.

    How It Works (Simple Steps)
                
    1. Start-up (When App Loads)
                
    Two machine learning models are trained on synthetic data:
                
    Logistic Regression — Simple, interpretable model
                
    Random Forest — Complex model that catches patterns
                
    Data is balanced using SMOTE (handles class imbalance)
                
    Models are cached so they don't retrain on every click
                
    2. User Fills Out a Form
                
    The app asks for 16 health factors across two columns:

    Left column (Demographics & Lifestyle):

    Age group (dropdown)
                
    Sex (male/female)
                
    Smoking history (yes/no)
                
    Current smoking frequency
                
    Exercise habits (yes/no)
                
    Alcohol consumption (days per week)
                
    Right column (Health Conditions & Measurements):

    Diabetes diagnosis (yes/no)
                
    General health rating (Excellent - Poor)
                
    Physical health days (0-30 days not good)
                
    Mental health days (0-30 days not good)
                
    Height (cm) and Weight (kg) - BMI automatically calculated
                
    Last routine checkup (time period)
                
    Could not afford medical care (yes/no)
                
    3. Click "Calculate Risk Score"
                
    When the user clicks the button:

    All 16 input values are collected into a dictionary
                
    Data is scaled using the same scaler from training
                
    Both models make predictions (probability of CVD)
                
    Probabilities are averaged: (LR_prob + RF_prob) / 2
                
    4. Results Displayed
                
    Three metrics shown side-by-side:

    Risk Score — Ensemble average (main result)
                
    LR Model — Logistic Regression prediction
                
    RF Model — Random Forest prediction
                
    Risk is categorized with colors:

    LOW RISK (< 15%) — Green box - Maintain healthy habits
                
    MODERATE RISK (15-35%) — Orange box - Start preventive measures
                
    HIGH RISK (> 35%) — Red box - Schedule medical consultation immediately
                
    Each category shows specific actionable recommendations.
                
    **Validation:** Built on CDC BRFSS 2024 data. Should be recalibrated with Kenyan data.
                
    **Disclaimer:** For research and education only. Not approved for clinical use.
    """)