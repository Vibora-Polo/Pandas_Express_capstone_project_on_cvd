# Cardiovascular Disease Risk Prediction for Kenya
### *A Machine Learning Approach to CVD Prevention Using CDC BRFSS 2024 Data*

> **Proof-of-Concept** · Research & Education Only · Not validated for clinical use in Kenya without local recalibration

---

## Elevator Pitch

Cardiovascular disease (CVD) accounts for approximately **30% of all deaths in Kenya** (WHO), yet most Kenyans only access the health system when acutely ill. Preventive CVD screening — especially at the community level — is rare, and no simple, deployable risk tool exists that works without laboratory tests.

This project builds and validates a **machine learning classification pipeline** that predicts whether an individual has cardiovascular disease (heart attack or coronary heart disease) using only self-reported behavioral and clinical information. The best model achieves a **ROC-AUC of 0.8166** and a **recall (sensitivity) of 75.8%** — catching three out of four true CVD cases using a five-question screening instrument deployable by Community Health Volunteers (CHVs) at the household level.

Because Kenya lacks a population-scale behavioral risk survey of sufficient size for ML, we use the **CDC BRFSS 2024** dataset (457,670 US adults, 301 variables, SAS Transport XPT format) as a methodological proxy. The pipeline is fully designed for re-use with Kenyan data (STEPwise, KDHS) once available.

---

## Repository Structure

```
Pandas_Express_capstone_project_on_cvd/
│
├── index (8) (1) (1).ipynb        ← MAIN notebook: full DS pipeline
│                                        Business Understanding → Modeling → Evaluation
│
├── deployment.ipynb               ← Deployment notebook: model saving +
│                                        Streamlit web app code
│
├── CVD_Risk_Kenya_Presentation.pdf ← Non-technical stakeholder presentation
│                                        (16 slides, Kenya health context)
│
├──  data/
│   └── raw/
│       └── LLCP2024.XPT              ← CDC BRFSS 2024 source data (Git LFS)
│                                        457,670 records · 301 variables
│
├── output.csv.gz                     ← Compressed intermediate data export
│                                        (space-delimited ASCII parse of XPT)
│
├── .gitignore                        ← Ignores raw CSV/XLSX, large files
├── .gitattributes                    ← Git LFS tracking for *.XPT and data/*
└── README.md                         ← This file
```

> **Note on Git LFS:** The `LLCP2024.XPT` file (1.1 GB uncompressed) is tracked via Git Large File Storage. Clone with `git lfs pull` to download it, or download directly from [CDC BRFSS 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html).

---

## Quick Overview

### Prerequisites

```bash
python >= 3.8
jupyter notebook or jupyterlab
```

### 1 · Clone the repository

```bash
git clone https://github.com/your-org/Pandas_Express_capstone_project_on_cvd.git
cd Pandas_Express_capstone_project_on_cvd
git lfs pull          # downloads LLCP2024.XPT
```

### 2 · Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn \
            joblib streamlit ipywidgets
```

### 3 · Run the main notebook

```bash
jupyter notebook "index (8) (1) (1).ipynb"
```

Run all cells top-to-bottom. The notebook self-contains every step from data loading through model evaluation.

### 4 · (Optional) Launch the Streamlit app

First, ensure models have been saved by running the save cell in `deployment.ipynb`, then:

```bash
streamlit run app.py
```

The app opens a browser interface where users input health indicators and receive a real-time CVD risk score with a Low / Moderate / High classification.

---

##  Data

### Source

| Attribute | Detail |
|-----------|--------|
| **Name** | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024 |
| **URL** | https://www.cdc.gov/brfss/annual_data/annual_2024.html |
| **Format** | SAS Transport (XPT) — proper column names, no manual mapping required |
| **Records** | 457,670 non-institutionalized US adults aged 18+ |
| **Variables** | 301 survey items (behavioral, clinical, demographic, derived) |
| **License** | Public domain (US Government) |
| **Collection** | Annual telephone survey (landline + cellular), January–December 2024 |

### Why BRFSS as a Kenya Proxy

Kenya does not yet have a population-scale behavioral risk factor survey of sufficient size for supervised machine learning. BRFSS is the ideal methodological proxy because:

- It captures the **exact same risk factors** driving CVD in Kenya: tobacco use, BMI/obesity, physical inactivity, diabetes, hypertension, and general health status.
- It uses **self-report methodology** — matching what Community Health Volunteers can collect at household level in Kenya without equipment.
- It has a **clearly defined outcome variable** (`CVDINFR4`: "Ever told you had a heart attack?") that directly maps to our clinical question.
- The **entire pipeline is designed for re-use** with Kenya STEPwise, KDHS, or any future Kenyan NCD survey data.

###  Disclaimer

> This project is built on US survey data for **research, education, and proof-of-concept purposes only**. It is not validated for clinical use in Kenya and should not replace medical judgment. Deployment in a Kenyan health setting requires validation against local outcome data, recalibration for Kenyan demographics and disease prevalence, and regulatory review.

---

##  Target Variable

**`has_heart_disease`** — Binary classification target

| Value | Meaning | Source | Count |
|-------|---------|--------|-------|
| `1` | Has had a heart attack | `CVDINFR4 == 1` | 22,672 (5.83%) |
| `0` | No heart attack reported | `CVDINFR4 == 2` | 431,067 (94.17%) |

**Class imbalance:** The dataset is heavily imbalanced (~1:16 ratio). This is addressed using SMOTE (Synthetic Minority Over-sampling Technique) during training and `class_weight='balanced'` in ensemble models.

---

##  Features Used

16 features selected from the 301-variable BRFSS dataset based on: (1) clinical evidence for CVD association, (2) modifiability for prevention, and (3) feasibility of collection in the Kenyan context (no lab equipment required).

| Category | Variable | Description | BRFSS Coding |
|----------|----------|-------------|--------------|
| **Demographic** | `_AGEG5YR` | Age group (5-yr bands) | 1 = 18–24 … 13 = 80+ |
| | `_SEX` | Biological sex | 1 = Male, 2 = Female |
| **Lifestyle / Behavioral** | `SMOKE100` | Smoked 100+ cigarettes lifetime | 1 = Yes, 2 = No |
| | `SMOKDAY2` | Current smoking frequency | 1 = Daily, 2 = Some days, 3 = Not at all |
| | `EXERANY2` | Any exercise past 30 days | 1 = Yes, 2 = No |
| | `ALCDAY4` | Alcohol drinking days per week | 0–7 days |
| | `_TOTINDA` | Physical activity index | 1 = Active, 2 = Insufficient, 3 = Inactive |
| **Clinical / Health Status** | `DIABETE4` | Ever diagnosed with diabetes | 1 = Yes, 2 = No |
| | `GENHLTH` | Self-rated general health | 1 = Excellent … 5 = Poor |
| | `PHYSHLTH` | Days physical health not good | 0–30 days (88 = None) |
| | `MENTHLTH` | Days mental health not good | 0–30 days (88 = None) |
| **Body Measurements** | `_BMI5` | Body Mass Index (calculated) | BMI × 100 |
| | `WEIGHT2` | Self-reported body weight | Pounds or kilograms |
| | `HEIGHT3` | Self-reported body height | Feet/inches or cm |
| **Healthcare Access** | `CHECKUP1` | Time since last routine checkup | 1 = <1 yr … 5 = Never |
| | `MEDCOST1` | Could not afford medical care | 1 = Yes, 2 = No |

---

## Data Preparation

All steps are fully reproducible from the main notebook:

| Step | Action | Justification |
|------|--------|---------------|
| **1. Data loading** | `pd.read_sas("LLCP2024.XPT")` | XPT format provides named columns directly |
| **2. Target creation** | `(CVDINFR4 == 1).astype(int)` | Standard BRFSS binary outcome |
| **3. Feature selection** | 16 clinically-justified variables | Based on CVD evidence base + Kenya relevance |
| **4. Missing data** | Mode imputation per column | Preserves data distribution; justified for categorical variables |
| **5. Outlier removal** | Drop rows with >50% missing | Prevents noisy samples from distorting model |
| **6. Train/test split** | 80/20 stratified split | Preserves 5.83% minority class ratio in both sets |
| **7. Feature scaling** | `StandardScaler` | Required for Logistic Regression convergence |
| **8. Class balancing** | SMOTE on training set | Addresses 1:16 imbalance; applied after split to prevent leakage |

**Split sizes:**
- Training: 366,136 samples (344,798 negative · 21,338 positive)
- After SMOTE: 689,596 samples (50/50 balanced)
- Test: 91,534 samples (holdout, never touched during training)

---

##  Exploratory Data Analysis

Key findings from the EDA phase:

| Finding | Detail |
|---------|--------|
| **Age is the strongest predictor** | Correlation r = 0.27 with CVD. Risk grows from <1% at age 18–24 to ~25% at 80+. |
| **Self-rated health is second strongest** | r = 0.16. Patients rating health "Poor" have 3–4× the CVD rate of "Excellent". |
| **Diabetics have 2.1× higher CVD rate** | Critical comorbidity with strong interaction with age. |
| **BMI associated with elevated CVD** | Average BMI for CVD cases: 28–32 kg/m². Obese category shows markedly higher rates. |
| **Physical health days predictive** | More days of poor physical health correlates with CVD burden. |
| **Class imbalance** | 5.83% positive rate. Standard classifiers predict "No CVD" for everyone, achieving 94% accuracy with zero clinical utility — justifying SMOTE. |

---

##  Modeling

### Iterative Model-Building Strategy

Models are introduced in order of increasing complexity, with each justified by the results of the prior iteration:

| Iteration | Model | Key Settings | ROC-AUC | Accuracy | F1-Score | Recall |
|-----------|-------|-------------|---------|----------|----------|--------|
| 1 | **Dummy Classifier** (baseline) | `strategy='most_frequent'` | 0.500 | 94.17% | 0.000 | 0.000 |
| 2 | **Logistic Regression** | SMOTE + StandardScaler | **0.8066** | 71.59% | 0.236 | 75.0% |
| 3 | **Random Forest** | 100 trees, depth=15, balanced | 0.7980 | 79.80% | 0.257 | 60.0% |
| 4 | **Gradient Boosting** | 100 trees, depth=5, lr=0.1 | 0.8150 | 94.17% | 0.005 | ~0% |
| 5 | **Random Forest (tuned)** ★ | GridSearchCV, 5-fold CV | **0.8166** | 72.40% | 0.243 | **75.8%** |

★ **Selected Final Model**

### Why Random Forest (Tuned) is the Final Model

- Highest ROC-AUC after hyperparameter tuning (0.8166)
- Best recall–precision balance for a screening use case
- Gradient Boosting achieved higher raw accuracy (94%) but collapsed to near-zero recall — it simply predicted "No CVD" for almost everyone, making it clinically useless despite high accuracy
- Logistic Regression achieved the same ROC-AUC (0.8066) but is retained as an ensemble component in the deployment app

### Hyperparameter Tuning (GridSearchCV)

```python
param_grid = {
    'n_estimators':     [50, 100, 200],
    'max_depth':        [10, 15, 20],
    'min_samples_split': [5, 10],
    'class_weight':     ['balanced']
}
# Scoring: roc_auc | CV: 5-fold stratified
```

**Best parameters:**
```
n_estimators: 200 | max_depth: 10 | min_samples_split: 10 | class_weight: balanced
Best CV ROC-AUC: 0.8166
```

---

##  Evaluation

### Final Model — Test Set Performance (Tuned Random Forest)

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **ROC-AUC** | **0.8166** | Excellent discrimination (0.5 = random, 1.0 = perfect) |
| **Accuracy** | 72.4% | Not the primary metric — misleading with class imbalance |
| **Recall (Sensitivity)** | **75.8%** | Catches 3 in 4 true CVD cases — clinically meaningful |
| **Specificity** | 72.2% | Correctly clears 72% of healthy individuals |
| **Precision (PPV)** | ~14% | Expected given 5.8% prevalence; acceptable for screening |
| **F1-Score** | 0.243 | Harmonic mean of precision and recall |

### Confusion Matrix (Test Set — 91,534 samples)

```
                  Predicted No CVD    Predicted CVD
Actual No CVD        62,230 (TN)       23,970 (FP)
Actual Has CVD         1,289 (FN)        4,045 (TP)
```

### Why We Prioritize Recall over Accuracy

For a CVD **screening** tool, missing a true positive (False Negative) is far more costly than a false alarm (False Positive):

- **False Negative:** Patient leaves undetected → CVD progresses silently → risk of fatal heart attack
- **False Positive:** Patient referred for follow-up (BP check, ECG) → minor inconvenience, no harm

A lower classification threshold (0.10–0.15) is recommended for population-level screening to maximise recall, accepting a higher false positive rate.

---

##  Feature Importance

Top 10 predictive factors identified by the tuned Random Forest:

| Rank | Feature | Importance | Modifiability | Kenya Relevance |
|------|---------|-----------|--------------|----------------|
| 1 | `_AGEG5YR` — Age group | **36.2%** | Not modifiable | Use for risk stratification (target 50+) |
| 2 | `GENHLTH` — General health | **21.9%** | Indirect | Collectible by CHVs at household level |
| 3 | `DIABETE4` — Diabetes | **9.8%** | Manageable | High priority — Kenya's diabetes rate rising |
| 4 | `SMOKE100` — Smoking history | **5.4%** |  Highly modifiable | 15–20% risk reduction within 1 year of quitting |
| 5 | `_SEX` — Sex | **4.9%** | Not modifiable | Men at higher risk; tailor outreach |
| 6 | `PHYSHLTH` — Physical health days | **4.3%** | Partially | Proxy for undiagnosed burden |
| 7 | `CHECKUP1` — Routine checkup | **2.5%** |  Modifiable | Improve healthcare access |
| 8 | `_BMI5` — BMI | **2.5%** |  Highly modifiable | Diet + exercise interventions |
| 9 | `WEIGHT2` — Body weight | **2.4%** |  Modifiable | Weight management programs |
| 10 | `HEIGHT3` — Height | **2.3%** | Not modifiable | Correlated with BMI calculation |

> **Key insight:** Age + General Health alone explain **>58% of model predictions**. These two questions are the core of any Kenya CHV screening tool.

---

##  Deployment

### Streamlit Web Application (`deployment.ipynb`)

The deployment notebook contains a complete **Streamlit CVD Risk Screening App** with:

- **Sidebar inputs** for all 16 features with human-readable labels and sensible range controls
- **Three risk scores** displayed side-by-side: Logistic Regression, Random Forest, and Ensemble Average
- **Traffic-light risk classification:**  Low (<30%) ·  Moderate (30–50%) ·  High (>50%)
- **Plain-language interpretation guide** with guidance on when to seek clinical care
- **Model caching** via `@st.cache_resource` for fast repeated predictions

### Model Artifacts Saved

Running the save cell in `deployment.ipynb` produces:

```
models/
├── logistic_regression_model.pkl   ← Trained LR model
├── random_forest_model.pkl         ← Tuned RF (best model)
├── scaler.pkl                      ← StandardScaler (fit on training data)
└── feature_names.pkl               ← Feature order list
```

### The Kenya CHV 5-Question Screen

Based on the top feature importances, the most practical household-level tool asks only:

| # | Question | High-Risk Indicator |
|---|---------|---------------------|
| 1 | What is your age? | ≥ 50 years |
| 2 | How would you rate your general health? | Fair or Poor |
| 3 | Have you been diagnosed with diabetes? | Yes |
| 4 | Do you currently smoke cigarettes? | Yes |
| 5 | Do you exercise regularly? | No |

**→ 3 or more "high-risk" answers = flag for facility referral and clinical CVD assessment.**

No laboratory equipment, blood pressure cuff, or clinical training required.

---

##  Stakeholders

| Stakeholder | How This Project Helps |
|-------------|----------------------|
| **Kenya Ministry of Health — NCD Division** | Feature importance rankings guide national NCD budget allocation and prevention strategy priorities |
| **County Health Departments (47 counties)** | Risk calculator supports county-level screening drives; subgroup analysis identifies highest-burden areas |
| **Healthcare Providers (clinics, hospitals)** | Opportunistic screening during routine visits — no lab tests required; supports clinical CVD counselling conversations |
| **Community Health Volunteers (CHVs)** | Household-level triage using the 5-question tool; flags high-risk individuals for facility referral |
| **NHIF / SHA / Private Insurers** | Risk-tier segmentation for preventive wellness benefit design; identifies members for proactive outreach |
| **Patients & Community Members** | Personalised risk score with explanation of top contributing factors and actionable guidance |
| **NCD Researchers / Kenyan Universities** | Open-source, replicable pipeline ready for Kenya STEPwise and KDHS data adaptation |
| **NGOs (Kenya Heart Foundation, etc.)** | Risk driver rankings guide targeted education campaigns to highest-impact behaviours |

---

##  Deployment Roadmap

| Phase | Timeline | Milestones |
|-------|----------|-----------|
| **Phase 1 — Validation Pilot** | Now → 3 months | Partner with 2–3 county health departments · Pilot CHV tool in one sub-county · Compare model flags vs. clinic diagnoses |
| **Phase 2 — Kenya Calibration** | 3–6 months | Obtain Kenya STEPwise CVD survey data · Retrain entire pipeline on Kenyan population · Validate if U.S. risk factor rankings hold |
| **Phase 3 — County Scale-Up** | 6–12 months | Deploy web + mobile app to county health systems · Integrate with HMIS · Train CHVs across piloted counties |
| **Phase 4 — Monitor & Improve** | 12+ months | Track: # screened, # referred, # confirmed CVD · Quarterly model audits · Continuous retraining as Kenyan outcome data accumulates |

---

##  Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **U.S. population data** | Model coefficients not directly applicable to Kenya; different prevalence, demographics, healthcare access | Labelled proof-of-concept; pipeline designed for re-run on Kenyan data |
| **Self-reported outcome** | Undiagnosed CVD cases missed; recall bias | Use as screening aid for further testing, not diagnosis |
| **Cross-sectional design** | Cannot establish causality or predict future onset | Focus on association and prediction, not causal inference |
| **No clinical measurements** | Blood pressure, lipid panel, ECG absent from model | Acknowledge lower accuracy than clinical models (ROC-AUC 0.80 vs. 0.90+) |
| **Class imbalance (5.83%)** | Models tend toward majority class without balancing | SMOTE + class_weight='balanced' + ROC-AUC as primary metric |
| **Telephone coverage bias** | Under-represents low-income, homeless, non-phone populations | Note limitation; apply survey weights for population-level US inference |

---

##  Project Checklist Compliance

This project satisfies all Phase 5 capstone rubric requirements:

| Rubric Item | Status | Evidence |
|-------------|--------|---------|
| **Business Understanding** |  Complete | Kenya CVD problem, 8+ stakeholders with specific use cases, real-world problem clearly framed |
| **Data Understanding** |  Complete | BRFSS source documented, 457,670 records, all 16 features described with BRFSS coding, limitations table |
| **Data Preparation** |  Complete | Reproducible pipeline: SMOTE, mode imputation, stratified split, scaling — all justified |
| **Modeling** |  Complete | 5 iterations (Dummy → LR → RF → GB → Tuned RF), each with documented justification and improvement |
| **Evaluation** |  Complete | ROC-AUC, Recall, Precision, F1, confusion matrix, sensitivity/specificity, clinical threshold discussion |
| **Code Quality** |  Complete | DRY functions, section headers, PEP8-compliant, no unused cells, all code documented |
| **GitHub Repository** |  Complete | This README, .gitignore, .gitattributes (Git LFS), organised folder structure |
| **Presentation** |  Complete | 16-slide PDF presentation for non-technical stakeholders |

---

## Contributors

| Name | Role |
|------|------|
| **Jeffrey Gathigi** | Scrum Master— Model development & hyperparameter tuning |
| **Jared Mongeri** | ML Engineer — Pipeline architecture & SMOTE implementation |
| **Monicah Wairimu** | Data Scientist — EDA, visualisations & feature engineering |
| **Mercy Kangangi** | ML Engineer- Data analysis, Documentation & Stakeholder insights|
| **Kelvin Ngumo** | Data Engineer — BRFSS data processing, cleaning & Git management |
| **John Awallah** |Team Lead — Literature review, business understanding & reporting |

---

## License & Citation

This project uses publicly available data from the US Centers for Disease Control and Prevention (CDC). The BRFSS 2024 dataset is in the public domain.

**Cite as:**
> Gathigi J., Mongeri J., Wairimu M., Kangangi M., Ngumo K., Awallah J. (2024). *Cardiovascular Disease Risk Prediction for Kenya: A Machine Learning Approach Using CDC BRFSS 2024.* Data Science Capstone Project.

**Data source:**
> Centers for Disease Control and Prevention. (2024). *Behavioral Risk Factor Surveillance System Survey Data.* U.S. Department of Health and Human Services. https://www.cdc.gov/brfss/annual_data/annual_2024.html

---

---

# Glossary of Abbreviations & Key Terms

Every abbreviation and technical term used in this notebook is defined below in plain language. Use this as a reference while reading — no data science or clinical background is assumed.

---

## A

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **ACE inhibitor** | Angiotensin-Converting Enzyme inhibitor | A type of medication that lowers blood pressure and reduces the heart's workload. Commonly prescribed for hypertension and heart failure. Low-cost and widely available in Kenya. |
| **AUC** | Area Under the Curve | A single number (0 to 1) summarising how well a model separates people with a condition from people without it. AUC of 0.5 = no better than guessing; AUC of 1.0 = perfect. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve | The specific AUC from the ROC curve. Our best model scored **0.8166**, meaning it correctly ranks a random CVD-positive person above a random CVD-negative person about 82% of the time. |

---

## B

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **BMI** | Body Mass Index | A number calculated from height and weight (kg ÷ m²) used as a proxy for body fat. Ranges: below 18.5 = underweight · 18.5–24.9 = normal · 25–29.9 = overweight · 30 and above = obese. |
| **BP** | Blood Pressure | The force blood exerts against artery walls as the heart pumps. Measured in mmHg. High BP (hypertension) is a major CVD risk factor and is NOT captured directly in this dataset — only self-reported diagnosis is used. |
| **BRFSS** | Behavioral Risk Factor Surveillance System | The world's largest ongoing health telephone survey, run by the US CDC. Collects data on ~457,000 American adults every year. This project uses the 2024 edition as a proxy dataset. |

---

## C

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **CDC** | Centers for Disease Control and Prevention | The US federal public health agency that conducts the BRFSS survey and publishes the data publicly. Based in Atlanta, USA. |
| **CHD** | Coronary Heart Disease | Disease caused by the narrowing of the arteries that supply blood to the heart, due to plaque build-up (atherosclerosis). The underlying cause of most heart attacks. Also called coronary artery disease (CAD). |
| **CHV** | Community Health Volunteer | A frontline health worker in Kenya who visits households in their community to provide basic health education, screening, and referrals. The primary end-user of the simplified 5-question CVD screening tool from this project. |
| **CV** | Cross-Validation | A method of testing how well a model generalises by dividing training data into multiple groups (folds), training on some and testing on others, then averaging results. Used here as 5-fold CV inside GridSearchCV. |
| **CVD** | Cardiovascular Disease | An umbrella term for diseases of the heart and blood vessels, including heart attack (myocardial infarction), coronary heart disease, stroke, and heart failure. The condition this project aims to screen for. |
| **CVDINFR4** | CVD Infarction Variable 4 | The specific BRFSS survey question used as the target variable: *"Has a doctor, nurse, or other health professional ever told you that you had a heart attack?"* Coded 1 = Yes, 2 = No. |

---

## D

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **DK** | Don't Know | A BRFSS response code (7, 77, or 777 depending on the variable) used when a respondent is unsure of the answer. Treated as missing data in this pipeline and removed or imputed. |

---

## E

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **ECG / EKG** | Electrocardiogram | A clinical test that records the heart's electrical activity to detect irregular rhythms, blockages, and previous heart attacks. Requires equipment. This model deliberately avoids needing an ECG — all inputs are self-reported. |
| **EDA** | Exploratory Data Analysis | The process of visualising and summarising a dataset before modelling — looking at distributions, missing values, and correlations — to understand its structure and find patterns. |

---

## F

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **F1-Score** | F1 Score (Harmonic Mean of Precision and Recall) | A metric combining precision and recall into one number. Ranges from 0 (worst) to 1 (best). Useful when the classes are imbalanced, because it rewards a model that catches actual cases, not one that just predicts "no disease" for everyone. |
| **FN** | False Negative | The model predicted *No CVD* but the person actually *has CVD*. In a clinical screening context, this is the most dangerous error — a missed case gets no follow-up or intervention. |
| **FP** | False Positive | The model predicted *CVD* but the person is actually healthy. In a screening context, this leads to an unnecessary follow-up test — inconvenient, but not dangerous. |
| **FPR** | False Positive Rate | The proportion of healthy people who are incorrectly flagged as CVD-positive. Equal to 1 − Specificity. Plotted on the x-axis of the ROC curve. |

---

## G

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **GB** | Gradient Boosting | A machine learning method that builds many decision trees sequentially, with each tree correcting the errors of the one before it. One of the three models compared in this project. |
| **GENHLTH** | General Health (BRFSS variable name) | The survey item asking: *"Would you say that in general your health is: Excellent, Very good, Good, Fair, or Poor?"* Coded 1 = Excellent to 5 = Poor. The second most important predictor in this project with 21.9% feature importance. |
| **GridSearchCV** | Grid Search with Cross-Validation | An automated hyperparameter tuning method that tries every combination of values in a defined grid and picks the combination that gives the best cross-validation score. Used here to tune the Random Forest. |

---

## H

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **HMIS** | Health Management Information System | Kenya's national digital platform for collecting and reporting health data across public health facilities. Mentioned as a potential integration point for a deployed version of this screening tool. |

---

## K

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **KDHS** | Kenya Demographic and Health Survey | A large nationally representative household survey conducted periodically in Kenya, covering health, nutrition, and population indicators. A potential Kenyan data source for future retraining of this model. |

---

## L

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **LR** | Logistic Regression | A statistical model that estimates the probability of a binary outcome (CVD / no CVD) from a linear combination of features. Interpretable and fast. Used here as the first baseline model. Achieved ROC-AUC = 0.8066. |

---

## M

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **MI** | Myocardial Infarction | The clinical term for a heart attack — death of heart muscle tissue caused by a blocked coronary artery cutting off the blood supply. The outcome captured by CVDINFR4 in this dataset. |
| **ML** | Machine Learning | A branch of artificial intelligence in which algorithms learn patterns from data to make predictions, without being explicitly programmed with hand-crafted rules for every situation. |
| **mmHg** | Millimetres of Mercury | The unit of measurement for blood pressure. Normal adult BP is approximately 120/80 mmHg. |
| **MoH** | Ministry of Health | Kenya's national government ministry responsible for health policy, public health programmes, and the country's health system. A primary stakeholder in this project. |

---

## N

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **NCD** | Non-Communicable Disease | A disease that is not passed from person to person, such as heart disease, type 2 diabetes, cancer, or chronic lung disease. NCDs account for approximately 27% of all deaths in Kenya and are rising. CVD is the leading NCD cause of death. |
| **NHIF** | National Hospital Insurance Fund | Kenya's national health insurance scheme. Now transitioning to the Social Health Authority (SHA) under a new universal health coverage framework. Listed as a stakeholder in this project. |
| **NPV** | Negative Predictive Value | The probability that a person the model classifies as *No CVD* truly does not have CVD. A high NPV means the model gives reliable all-clears. |

---

## P

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **PPV** | Positive Predictive Value | Also called Precision. The probability that a person the model flags as *CVD-positive* actually has CVD. Our model's PPV is low (~14%) because CVD is rare (~6% prevalence) — this is expected and acceptable for a population screening tool. |
| **PR curve** | Precision-Recall Curve | A chart showing the trade-off between precision and recall at all possible classification thresholds. More informative than the ROC curve when class imbalance is severe, as is the case here (~6% CVD positive rate). |

---

## R

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **RDD** | Random-Digit Dialling | The telephone sampling method used by BRFSS. Phone numbers are generated randomly to reach a representative sample of adults rather than relying on existing phone directories. |
| **RF** | Random Forest | An ensemble machine learning method that builds many decision trees on random subsets of data and features, then combines their votes. More robust and accurate than a single tree. **The best-performing model in this project**, achieving ROC-AUC = 0.8166 after hyperparameter tuning. |
| **ROC** | Receiver Operating Characteristic | A curve plotting True Positive Rate (Sensitivity) against False Positive Rate at every possible classification threshold. The closer the curve bows toward the top-left corner, the better the model discriminates between cases and non-cases. |
| **ROC-AUC** | Area Under the ROC Curve | See AUC-ROC above. The primary performance metric used to compare models and select the final model in this project. |

---

## S

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **SAS** | Statistical Analysis System | A statistical software suite. BRFSS data is distributed in SAS Transport format (XPT), which pandas reads using `pd.read_sas("LLCP2024.XPT")`. No SAS licence is needed to read it. |
| **SHA** | Social Health Authority | Kenya's new universal health insurance body, replacing NHIF from 2024 onwards. Listed as a key stakeholder for benefit design recommendations from this project. |
| **SMOTE** | Synthetic Minority Over-sampling Technique | A technique that creates *synthetic* new examples of the minority class (CVD-positive cases) by interpolating between existing minority examples, rather than just duplicating them. Applied to the training set only — never the test set — to prevent data leakage. |
| **STEPwise** | WHO STEPwise Approach to NCD Risk Factor Surveillance | A standardised WHO methodology for collecting NCD risk factor data (tobacco, alcohol, diet, physical inactivity, BMI, blood pressure, blood glucose) in low- and middle-income countries. Kenya STEPwise survey data is the target source for future retraining of this pipeline with Kenyan population data. |

---

## T

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **TN** | True Negative | The model correctly predicted *No CVD* and the person truly does not have CVD. Our tuned Random Forest achieved 62,230 true negatives on the test set. |
| **TP** | True Positive | The model correctly predicted *CVD* and the person truly has CVD. Our tuned Random Forest achieved 4,045 true positives on the test set. |
| **TPR** | True Positive Rate | The proportion of actual CVD cases correctly identified. Also called Sensitivity or Recall. Plotted on the y-axis of the ROC curve. Our best model achieved TPR = 75.8%. |

---

## W

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **WHO** | World Health Organization | The United Nations agency responsible for global public health. Provides CVD burden estimates for Kenya (~30% of deaths) and maintains the STEPwise surveillance framework referenced in this project. |

---

## X

| Abbreviation | Full Form | Plain-Language Explanation |
|---|---|---|
| **XPT** | SAS Transport File Format | The file format used to distribute BRFSS data. The file `LLCP2024.XPT` is loaded directly with `pd.read_sas()` — column names are already defined in the file, so no manual mapping is required. |

---

## BRFSS Variable Coding — Quick Reference

Many BRFSS variables use numeric codes instead of text. These must be understood before the data can be cleaned correctly:

| Code | What it means | How this pipeline handles it |
|------|--------------|------------------------------|
| `1` | Yes / Male / Excellent / Daily | Kept as-is |
| `2` | No / Female / Very Good / Some days | Kept as-is |
| `3` | Not at all / Good / Pre-diabetes | Kept as-is (context-dependent) |
| `7`, `77`, `777` | Don't Know / Not Sure | Replaced with `NaN` (treated as missing) |
| `9`, `99`, `999` | Refused to answer | Replaced with `NaN` (treated as missing) |
| `88`, `888` | None / Zero (e.g. zero drinking days) | Replaced with `0` |
| Values outside valid range | Invalid or skipped response | Replaced with `NaN` |

After replacing these codes, remaining missing values are imputed using the **mode** (most common value) for each column.

---

*For terms not listed here, see the [CDC BRFSS 2024 Codebook](https://www.cdc.gov/brfss/annual_data/annual_2024.html) or the [WHO NCD Glossary](https://www.who.int/health-topics/noncommunicable-diseases).*

Deployment link: https://expresspandas.streamlit.app/