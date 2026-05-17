import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Repurchase Prediction System",
    page_icon="🛒",
    layout="centered"
)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(
    'xgboost_repurchase_model.pkl'
)

# =========================================================
# TITLE
# =========================================================

st.title("🛒 E-commerce Repurchase Prediction System")

st.markdown("""
This application predicts customer repurchase probability
using Machine Learning techniques.
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Customer Information")

# Strategic features
satis = st.sidebar.slider(
    "Customer Satisfaction",
    1.0,
    10.0,
    7.5
)

recomm = st.sidebar.slider(
    "Recommendation Intention",
    1.0,
    10.0,
    7.5
)

poverq = st.sidebar.slider(
    "Perceived Overall Quality",
    1.0,
    10.0,
    7.5
)

pq = st.sidebar.slider(
    "Perceived Product Quality",
    1.0,
    10.0,
    7.5
)

# =========================================================
# INPUT DATA
# =========================================================

input_data = pd.DataFrame({
    'poverq': [poverq],
    'soverq': [7.5],
    'pq': [pq],
    'satis': [satis],
    'repur': [7.5],
    'recomm': [recomm],
    'Q19': [1],
    'VN_1009_Q20A': [satis],
    'VN_1009_TP01': [7],
    'VN_1009_TP02': [7],
    'VN_1009_TP03': [7],
    'VN_1009_TP04': [7],
    'VN_1009_TP05': [7],
    'VN_1009_TP06': [7],
    'VN_1009_TP07': [7],
    'VN_1009_TP08': [7],
    'VN_1009_TP09': [0],
    'VN_1009_TP10': [7],
    'VN_1009_TP11': [7],
    'VN_1009_TP12': [7],
    'VN_1009_TP13': [0],
    'VN_1009_TP14': [0],
    'VN_1009_TP15': [0],
    'VN_1009_TP16': [0],
    'VN_1009_TP17': [7],
    'VN_1009_TP18': [0],
    'VN_1009_TP19': [0],
    'VN_1009_TP20': [1],
    'VN_1009_TP21': [1],
    'VN_1009_TP24_1': [1],
    'VN_1009_TP24_2': [1],
    'Q9C_P': [1],
    'Q9D': [100],
    'VN_1009_TP25A': [2],
    'age': [35],
    'race': [1],
    'work': [1],
    'income': [50],
    'educat': [7],
    'childsupp': [0],
    'marital': [2],
    'gender': [1],
    'house': [3],
    'customer_experience_index': [7.5],
    'ux_score': [7.5],
    'service_quality_score': [7.5],
    'promotion_sensitivity_score': [7.5],
    'company_v': ['FAVE'],
    'VN_1009_TP21_6specify': ['none'],
    'VN_1009_TP22': ['none'],
    'VN_1009_TP23': ['none'],
    'pincome': ['5'],
    'DOI': ['2025-01-01']
})

# =========================================================
# PREDICT
# =========================================================

if st.button("Predict Repurchase Probability"):

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        st.subheader("Prediction Results")

        st.metric(
            label="Repurchase Probability",
            value=f"{probability*100:.2f}%"
        )

        if prediction == 1:

            st.success(
                "High Repurchase Probability"
            )

        else:

            st.error(
                "Low Repurchase Probability"
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Machine Learning II - Universidad Externado de Colombia"
)