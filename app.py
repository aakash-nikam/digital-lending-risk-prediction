import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Dublin Loan Risk Early-Warning System",
    layout="wide"
)


# --------------------------------------------------
# Constants
# --------------------------------------------------
MODEL_FILE = "xgb_model.pkl"
SIGNALS_FILE = "signals_dublin_monthly.csv"
SENTIMENT_FILE = "reviews_sentiment_dublin_app_monthly.csv"
MACRO_FILE = "macro_financial_pressure_yearly.csv"

AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
APPS = ["Revolut", "Klarna", "Avant", "Humm", "FlexiFi", "AIB App"]

AGE_MAP = {v: i for i, v in enumerate(AGE_GROUPS)}
APP_MAP = {v: i for i, v in enumerate(APPS)}


# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


def safe_read_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)
    return None


def lookup_signals(df, year_month):
    defaults = {
        "consumer_sentiment_index": 75.0,
        "bnpl_search_index": 55.0,
        "debt_to_income_ratio": 0.65,
        "avg_savings_rate": 0.12,
    }
    if df is None:
        return defaults
    row = df[df["year_month"] == year_month]
    if row.empty:
        return defaults
    return row.iloc[0].to_dict()


def lookup_sentiment(df, year_month, app):
    if df is None:
        return 0.10
    row = df[
        (df["year_month"] == year_month) &
        (df["app_used"] == app)
    ]
    if row.empty:
        return 0.10
    return float(row.iloc[0]["sentiment_score"])


def lookup_macro(df, year):
    if df is None:
        return 25.0
    row = df[df["year"] == year]
    if row.empty:
        return 25.0
    return float(row.iloc[0]["financial_pressure_index"])


def build_features(payload, model_features):
    df = pd.DataFrame([payload])

    dt = pd.to_datetime(df["application_date"])
    df["month"] = dt.dt.month
    df["year_month_num"] = dt.dt.year * 100 + dt.dt.month

    df["loan_per_month"] = df["loan_amount"] / df["loan_term_months"]
    df["missed_payment_ratio"] = df["missed_payments"] / df["loan_term_months"]

    df["age_group"] = df["age_group"].map(AGE_MAP)
    df["app_used"] = df["app_used"].map(APP_MAP)

    df = df.drop(columns=["application_date"])

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    return df[model_features]


# --------------------------------------------------
# Load model
# --------------------------------------------------
if not Path(MODEL_FILE).exists():
    st.error("Model file xgb_model.pkl not found in the app directory.")
    st.stop()

model = load_model()
model_features = model.get_booster().feature_names


# --------------------------------------------------
# Load optional lookup tables
# --------------------------------------------------
signals_df = safe_read_csv(SIGNALS_FILE)
sentiment_df = safe_read_csv(SENTIMENT_FILE)
macro_df = safe_read_csv(MACRO_FILE)

if signals_df is not None:
    signals_df["year_month"] = signals_df["year_month"].astype(str)

if sentiment_df is not None:
    sentiment_df["year_month"] = sentiment_df["year_month"].astype(str)
    sentiment_df["app_used"] = sentiment_df["app_used"].astype(str)

if macro_df is not None:
    macro_df["year"] = macro_df["year"].astype(int)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("Dublin Loan Risk Early-Warning System")

st.write(
    """
    This system provides an early warning assessment of loan default risk
    for digital lending in Dublin.

    Applicant inputs are intentionally minimal.
    Market, sentiment and macroeconomic indicators are automatically derived
    from Dublin-specific datasets where available.
    """
)

st.divider()


# --------------------------------------------------
# Layout
# --------------------------------------------------
left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("Applicant information (Dublin only)")

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            application_date = st.date_input(
                "Application date",
                value=pd.to_datetime("2024-06-01")
            )
            age_group = st.selectbox("Age group", AGE_GROUPS)

        with c2:
            loan_amount = st.number_input(
                "Loan amount (€)",
                min_value=100.0,
                max_value=5000.0,
                value=1000.0,
                step=50.0,
            )
            loan_term = st.selectbox("Loan term (months)", [3, 6, 12, 24])

        with c3:
            missed_payments = st.slider("Missed payments", 0, 10, 0)
            app_used = st.selectbox("Digital lending app", APPS)

        threshold = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.01,
        )

        run = st.form_submit_button("Run risk assessment")


with right:
    st.subheader("Risk assessment output")

    if not run:
        st.info("Run an assessment to see the risk decision.")
    else:
        dt = pd.to_datetime(application_date)
        year = int(dt.year)
        year_month = f"{year}-{dt.month:02d}"

        signals = lookup_signals(signals_df, year_month)
        sentiment = lookup_sentiment(sentiment_df, year_month, app_used)
        macro = lookup_macro(macro_df, year)

        payload = {
            "application_date": str(application_date),
            "loan_amount": loan_amount,
            "loan_term_months": loan_term,
            "missed_payments": missed_payments,
            "age_group": age_group,
            "app_used": app_used,
            "consumer_sentiment_index": signals["consumer_sentiment_index"],
            "bnpl_search_index": signals["bnpl_search_index"],
            "debt_to_income_ratio": signals["debt_to_income_ratio"],
            "avg_savings_rate": signals["avg_savings_rate"],
            "sentiment_score": sentiment,
            "financial_pressure_index": macro,
        }

        X = build_features(payload, model_features)
        prob = float(model.predict_proba(X)[:, 1][0])
        label = int(prob >= threshold)

        st.metric("Probability of default", f"{prob:.4f}")
        st.metric("Decision threshold", f"{threshold:.2f}")

        if label == 1:
            st.error("Risk classification: HIGH RISK")
            st.write("Recommended action: manual review or tighter underwriting.")
        else:
            st.success("Risk classification: LOW RISK")
            st.write("Recommended action: standard processing.")

        st.divider()

        st.subheader("Auto-filled Dublin context")

        context_df = pd.DataFrame([{
            "Year-Month": year_month,
            "Consumer sentiment index": payload["consumer_sentiment_index"],
            "BNPL search index": payload["bnpl_search_index"],
            "Debt-to-income ratio": payload["debt_to_income_ratio"],
            "Savings rate": payload["avg_savings_rate"],
            "App sentiment score": payload["sentiment_score"],
            "Financial pressure index": payload["financial_pressure_index"],
        }])

        st.dataframe(context_df, use_container_width=True)
