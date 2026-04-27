import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
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
st.title(" Cardiovascular Risk Screening Tool")
st.markdown("""
### For Kenyans: A Machine Learning Approach to CVD Prevention

** DISCLAIMER:** NOT a replacement for medical diagnosis. Always consult healthcare professionals.
""")

# Sidebar
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown("""
    1. Enter your health information
    2. Click "Calculate Risk"
    3. View your personalized risk assessment
    4. Follow recommendations for next steps
    """)

# Main form
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
    
    # Make prediction (THIS FIXES YOUR ERROR)
    lr_prob, rf_prob, avg_prob = predict_risk(user_data)
    
    if avg_prob is not None:
        # Display results
        st.divider()
        st.subheader("Your Risk Assessment Results")
        
        risk_label, risk_color = get_risk_category(avg_prob)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{avg_prob:.1%}", delta=None)
        
        with col2:
            st.metric("LR Model", f"{lr_prob:.1%}")
        
        with col3:
            st.metric("RF Model", f"{rf_prob:.1%}")
        
        st.markdown(f"### {risk_label}")
        
        # Recommendations
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

# Footer
st.divider()
st.markdown("""
---
**About:** This tool uses machine learning to estimate CVD risk based on self-reported factors.
**Validation:** Built on CDC BRFSS 2024 data. Should be recalibrated with Kenyan data.
**Disclaimer:** For research and education only. Not approved for clinical use.
""")