import streamlit as st
import pandas as pd
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

model = joblib.load("xgb_clean.pkl")

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
# PREDICT
# =========================================================

if st.button("Predict Repurchase Probability"):

    try:

        # =====================================================
        # INPUT DATA
        # =====================================================

        input_data = pd.DataFrame({

            'satis': [satis],
            'recomm': [recomm],
            'customer_experience_index': [customer_experience_index],
            'service_quality_score': [service_quality_score]

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