import pickle
import numpy as np
import streamlit as st
import pandas as pd 

st.set_page_config(
    page_title="Customer Satisfaction Predictor",
    layout="centered",
)

# HEADER
st.title("E-commerce Satisfaction Predictor")
st.caption("ML-powered e-commerce customer satisfaction prediction")

# LOAD ARTEFACTS
@st.cache_resource
def load_artefacts():
    with open("lr_model.pkl",  "rb") as f: lr_model  = pickle.load(f)
    with open("dt_model.pkl",  "rb") as f: dt_model  = pickle.load(f)
    with open("scaler.pkl",    "rb") as f: scaler    = pickle.load(f)
    with open("features.pkl",  "rb") as f: features  = pickle.load(f)
    return lr_model, dt_model, scaler, features

try:
    lr_model, dt_model, scaler, features = load_artefacts()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    st.error(
        "Trained model files not found. "
        "Please run **`python train.py`** first to generate "
        "`lr_model.pkl`, `dt_model.pkl`, `scaler.pkl`, and `features.pkl`."
    )

# LABEL MAP
LABEL_MAP = {0: "Neutral", 1: "Satisfied", 2: "Unsatisfied"}

EMOJI_MAP = {
    "Satisfied":   "Satisfied",
    "Neutral":     "Neutral",
    "Unsatisfied": "Unsatisfied",
}

# INPUT FORM
if models_loaded:
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    <style>
    div[data-baseweb="select"] * {
        cursor: default !important;
    }
    </style>
""", unsafe_allow_html=True)
        discount_applied = st.selectbox(
            "Discount Applied?",
            options=["No (0)", "Yes (1)"],
            help="Was a discount applied to the customer's order?",
        )

        discount_value = int(discount_applied.split("(")[1].rstrip(")"))

    with col2:
        days_since_last = st.slider(
            "Days Since Last Purchase",
            min_value=0,
            max_value=365,
            value=30,
            help="Number of days since the customer last made a purchase.",
        )

    st.write("")

    input_map = {
        "Discount Applied": discount_value,
        "Days Since Last Purchase": days_since_last,
    }
    input_array = pd.DataFrame([[input_map[f] for f in features]], columns=features)

    if st.button("Predict Satisfaction Level", use_container_width=True):

        # --- Logistic Regression (needs scaling) ---
        input_scaled = scaler.transform(input_array)
        lr_pred      = lr_model.predict(input_scaled)[0]
        lr_proba     = lr_model.predict_proba(input_scaled)[0]
        lr_verdict   = LABEL_MAP.get(int(lr_pred), str(lr_pred))
        lr_conf      = round(max(lr_proba) * 100, 1)

        # --- Decision Tree (no scaling needed) ---
        dt_pred    = dt_model.predict(input_array)[0]
        dt_proba   = dt_model.predict_proba(input_array)[0]
        dt_verdict = LABEL_MAP.get(int(dt_pred), str(dt_pred))
        dt_conf    = round(max(dt_proba) * 100, 1)

        st.divider()
        st.subheader("Prediction Results")

        col_lr, col_dt = st.columns(2)

        # ── Logistic Regression ──
        with col_lr:
            st.markdown("**Logistic Regression**")
            st.metric(
                label="Predicted Satisfaction",
                value=EMOJI_MAP.get(lr_verdict, lr_verdict),
                delta=f"Confidence: {lr_conf}%",
            )
            with st.expander("Probability breakdown", expanded=True):
                for c, p in zip(lr_model.classes_, lr_proba):
                    lbl = LABEL_MAP.get(int(c), str(c))
                    pct = round(float(p) * 100, 2)
                    st.progress(int(pct), text=f"{lbl}: {pct}%")

        # ── Decision Tree ──
        with col_dt:
            st.markdown("**Decision Tree**")
            st.metric(
                label="Predicted Satisfaction",
                value=EMOJI_MAP.get(dt_verdict, dt_verdict),
                delta=f"Confidence: {dt_conf}%",
            )
            with st.expander("Probability breakdown", expanded=True):
                for c, p in zip(dt_model.classes_, dt_proba):
                    lbl = LABEL_MAP.get(int(c), str(c))
                    pct = round(float(p) * 100, 2)
                    st.progress(int(pct), text=f"{lbl}: {pct}%")

# FOOTER
st.divider()
st.caption("E-commerce ML Project · Logistic Regression & Decision Tree")