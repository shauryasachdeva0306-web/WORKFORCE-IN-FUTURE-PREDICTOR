import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Load model + expected column order (cached so it only runs once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, columns


model, columns = load_artifacts()

# ----------------------------------------------------------------------------
# These mappings must match the LabelEncoder used during training
# (fit on the original Employee.csv, alphabetical order -> 0,1,2,...)
# ----------------------------------------------------------------------------
EDUCATION_MAP = {"Bachelors": 0, "Masters": 1, "PHD": 2}
CITY_MAP = {"Bangalore": 0, "New Delhi": 1, "Pune": 2}
GENDER_MAP = {"Female": 0, "Male": 1}
EVER_BENCHED_MAP = {"No": 0, "Yes": 1}

st.title("🧑‍💼 Employee Attrition Predictor")
st.write(
    "Fill in the employee details below and click **Predict** to estimate "
    "whether the employee is likely to leave the company."
)

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("Education", list(EDUCATION_MAP.keys()))
        city = st.selectbox("City", list(CITY_MAP.keys()))
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        ever_benched = st.selectbox("Ever Benched", list(EVER_BENCHED_MAP.keys()))

    with col2:
        joining_year = st.number_input(
            "Joining Year", min_value=2000, max_value=2035, value=2015, step=1
        )
        payment_tier = st.selectbox("Payment Tier", [1, 2, 3], index=2)
        age = st.number_input("Age", min_value=18, max_value=65, value=25, step=1)
        experience = st.number_input(
            "Experience In Current Domain (years)",
            min_value=0,
            max_value=40,
            value=2,
            step=1,
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    # Build a single-row dataframe with the SAME column order the model
    # was trained on (loaded from columns.pkl), using the label-encoded values.
    input_dict = {
        "Education": EDUCATION_MAP[education],
        "JoiningYear": joining_year,
        "City": CITY_MAP[city],
        "PaymentTier": payment_tier,
        "Age": age,
        "Gender": GENDER_MAP[gender],
        "EverBenched": EVER_BENCHED_MAP[ever_benched],
        "ExperienceInCurrentDomain": experience,
    }

    input_df = pd.DataFrame([input_dict])[columns]  # enforce training column order

    prediction = model.predict(input_df)[0]

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ This employee is likely to **LEAVE** the company.")
    else:
        st.success("✅ This employee is likely to **STAY** with the company.")

    # Show probability if the underlying model supports it
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        st.write("**Prediction confidence:**")
        st.progress(float(proba[int(prediction)]))
        prob_df = pd.DataFrame(
            {"Outcome": ["Stay", "Leave"], "Probability": [proba[0], proba[1]]}
        )
        st.bar_chart(prob_df.set_index("Outcome"))

    with st.expander("See input passed to the model"):
        st.dataframe(input_df)

st.divider()
st.caption(
    "Model: VotingClassifier (RandomForest, SVC, LogisticRegression, KNN, GaussianNB) "
    "trained on Employee.csv"
)
