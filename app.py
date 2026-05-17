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

model = joblib.load("modelo_final.pkl")

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

customer_experience_index = st.sidebar.slider(
    "Customer Experience",
    1.0,
    10.0,
    7.5
)

service_quality_score = st.sidebar.slider(
    "Service Quality",
    1.0,
    10.0,
    7.5
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Repurchase Probability"):

    try:

        input_data = pd.DataFrame({

            'poverq': [float(poverq)],
            'soverq': [7.5],
            'pq': [float(pq)],
            'satis': [float(satis)],
            'repur': [7.5],
            'recomm': [float(recomm)],
            'Q19': [1],
            'VN_1009_Q20A': [float(satis)],
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

            'customer_experience_index': [float(customer_experience_index)],
            'ux_score': [float(customer_experience_index)],
            'service_quality_score': [float(service_quality_score)],
            'promotion_sensitivity_score': [7.5],

            # IMPORTANTÍSIMO:
            # categorías como category y NO str

            'company_v': pd.Categorical(['FAVE']),
            'VN_1009_TP21_6specify': pd.Categorical(['none']),
            'VN_1009_TP22': pd.Categorical(['none']),
            'VN_1009_TP23': pd.Categorical(['none']),
            'pincome': pd.Categorical(['5']),
            'DOI': pd.Categorical(['2025-01-01'])

        })

        # =====================================================
        # PREDICTION
        # =====================================================

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        # =====================================================
        # RESULTS
        # =====================================================

        st.subheader("Prediction Results")

        st.metric(
            "Repurchase Probability",
            f"{probability:.2%}"
        )

        if prediction == 1:

            st.success(
                "High probability of customer repurchase"
            )

        else:

            st.error(
                "Low probability of customer repurchase"
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