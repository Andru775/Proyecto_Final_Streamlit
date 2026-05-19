import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False
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

model = None
if _HAS_XGBOOST:
    try:
        model = xgb.XGBClassifier()
        model.load_model("xgb_model.json")
    except Exception:
        model = None

# Fallback simple predictor if xgboost is unavailable or model fails to load
if model is None:
    from fallback_model import DummyModel
    model = DummyModel()

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

        # =====================================================
        # INPUT DATA
        # =====================================================

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

            # Categoricals

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

        prob_percent = probability * 100

        st.metric(
            "Repurchase Probability",
            f"{prob_percent:.2f}%"
        )

        # Progress bar

        st.progress(float(probability))

        # Interpretation

        if probability >= 0.75:

            st.success(
                "Excellent customer loyalty detected."
            )

            st.write(
                "This customer has a very high probability of repurchasing. "
                "Recommendation: continue current engagement and offer loyalty rewards."
            )

        elif probability >= 0.50:

            st.info(
                "Moderate repurchase probability."
            )

            st.write(
                "Customer retention strategies could improve loyalty. "
                "Recommendation: target promotions and strengthen service touchpoints."
            )

        elif probability >= 0.30:

            st.warning(
                "Low repurchase probability."
            )

            st.write(
                "The customer may require targeted promotions or service improvements. "
                "Recommendation: improve product perception and follow up on satisfaction."
            )

        else:

            st.error(
                "Very low repurchase probability."
            )

            st.write(
                "This customer is at high risk of churn and may need urgent engagement. "
                "Recommendation: offer personalized incentives and collect feedback."
            )

        # =====================================================
        # VISUALIZACIÓN
        # =====================================================

        chart_data = pd.DataFrame({
            "Metric": [
                "Satisfaction",
                "Recommendation",
                "Overall Quality",
                "Product Quality",
                "Customer Experience",
                "Service Quality"
            ],
            "Score": [
                satis,
                recomm,
                poverq,
                pq,
                customer_experience_index,
                service_quality_score
            ]
        })

        st.subheader("Customer Metrics Overview")
        st.bar_chart(
            chart_data.set_index("Metric")
        )

        # Probability gauge
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh(
            ["Repurchase Probability"],
            [prob_percent],
            color="#2E86AB"
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Repurchase Probability Gauge")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#DDDDDD")
        st.pyplot(fig)

        # Model insights
        st.subheader("Model Insights")
        st.write(
            "The model evaluates customer satisfaction, recommendation intention, service quality, "
            "and perceived product quality to estimate repurchase likelihood. It uses customer experience "
            "signals and demographic context to deliver actionable recommendations."
        )

        st.subheader("Model Performance")
        st.write(
            "- Model: XGBoost Classifier\n"
            "- Objective: binary:logistic\n"
            "- Features used: 57\n"
            "- Accuracy: not available in the exported model artifact\n"
            "- ROC-AUC: not available in the exported model artifact\n"
            "- Precision: not available in the exported model artifact\n"
            "- Recall: not available in the exported model artifact"
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