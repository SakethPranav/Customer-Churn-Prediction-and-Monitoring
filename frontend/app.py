"""Streamlit frontend for the churn prediction system.

Talks to the backend only through REST (see api_client.py).
Run with:  streamlit run frontend/app.py
"""
import os

import pandas as pd
import streamlit as st

import api_client as api

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://127.0.0.1:5000")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000")

FEATURES = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"]
ADDONS = [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
          ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
          ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]
TIER_LABEL = {"low": "Low risk", "medium": "Medium risk", "high": "High risk"}
TIER_BOX = {"low": st.success, "medium": st.warning, "high": st.error}

st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")


def show_result(pred: dict, version: str) -> None:
    prob, tier = pred["churn_probability"], pred["risk_tier"]
    TIER_BOX[tier](f"**{TIER_LABEL[tier]}**: estimated churn probability {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{prob:.1%}")
    c2.metric("Prediction", "Likely to leave" if pred["churn_prediction"] else "Likely to stay")
    c3.metric("Model version", version)


def page_single() -> None:
    st.header("Single customer prediction")
    st.caption("Fill in the customer's details, then press the button at the bottom.")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has a partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])

    with c2:
        st.subheader("Account")
        tenure = st.number_input("Tenure (months)", 0, 120, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges", 0.0, 500.0, 70.0, step=1.0)
        auto_total = st.checkbox("Estimate total charges (tenure x monthly)", value=True)
        if auto_total:
            total = float(tenure) * monthly
            st.caption(f"Total charges: {total:,.2f}")
        else:
            total = st.number_input("Total charges", 0.0, 100000.0, 840.0, step=10.0)

    with c3:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        if phone == "Yes":
            lines = st.selectbox("Multiple lines", ["No", "Yes"])
        else:
            lines = "No phone service"
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addons = {}
        for field, label in ADDONS:
            if internet == "No":
                addons[field] = "No internet service"
            else:
                addons[field] = st.selectbox(label, ["No", "Yes"])

    customer = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": int(tenure),
        "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        **addons, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": float(monthly),
        "TotalCharges": float(total)}

    st.divider()
    if st.button("Predict churn risk", type="primary"):
        try:
            with st.spinner("Contacting the prediction service..."):
                data = api.predict(customer)
            show_result(data["prediction"], data["model_version"])
        except api.ApiError as err:
            st.error(str(err))


def load_batch_csv(upload) -> pd.DataFrame:
    """Read the uploaded CSV and check it has what the model needs."""
    df = pd.read_csv(upload)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("The file is missing these columns: " + ", ".join(missing))
    # Same rule as the cleaning stage: blank TotalCharges (tenure 0) become 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if df[FEATURES].isna().any().any():
        bad = df[FEATURES].columns[df[FEATURES].isna().any()].tolist()
        raise ValueError("Empty values found in: " + ", ".join(bad))
    df["tenure"] = df["tenure"].astype(int)
    return df


def page_batch() -> None:
    st.header("Batch prediction")
    st.caption("Upload a CSV with one customer per row. Extra columns "
               "(such as customerID) are kept in the results.")
    upload = st.file_uploader("Customer CSV file", type="csv")
    if upload is not None and st.button("Score all customers", type="primary"):
        try:
            df = load_batch_csv(upload)
            with st.spinner(f"Scoring {len(df):,} customers..."):
                preds, version = api.predict_batch(df[FEATURES].to_dict("records"))
            out = df.copy()
            out["churn_probability"] = [p["churn_probability"] for p in preds]
            out["churn_prediction"] = [p["churn_prediction"] for p in preds]
            out["risk_tier"] = [p["risk_tier"] for p in preds]
            out = out.sort_values("churn_probability", ascending=False)
            st.session_state["batch"] = (out, version)
        except ValueError as err:
            st.session_state.pop("batch", None)
            st.error(str(err))
        except api.ApiError as err:
            st.session_state.pop("batch", None)
            st.error(str(err))

    if "batch" in st.session_state:
        out, version = st.session_state["batch"]
        counts = out["risk_tier"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers scored", f"{len(out):,}")
        c2.metric("High risk", int(counts.get("high", 0)))
        c3.metric("Medium risk", int(counts.get("medium", 0)))
        c4.metric("Low risk", int(counts.get("low", 0)))
        shown = [c for c in ["customerID"] if c in out.columns] + [
            "churn_probability", "churn_prediction", "risk_tier",
            "tenure", "Contract", "MonthlyCharges"]
        st.dataframe(out[shown], width="stretch", hide_index=True)
        st.download_button("Download full results (CSV)",
                           out.to_csv(index=False).encode("utf-8"),
                           file_name="churn_predictions.csv", mime="text/csv")
        st.caption(f"Scored with model version {version}, highest risk first.")


def page_pipeline() -> None:
    st.header("Pipeline and monitoring")
    st.subheader("Service status")
    try:
        info = api.health()
        c1, c2, c3 = st.columns(3)
        c1.metric("API status", info["status"])
        c2.metric("Model", info["model_name"])
        c3.metric("Model version", info.get("model_version") or "none")
    except api.ApiError as err:
        st.error(str(err))

    st.subheader("Training pipeline (DVC)")
    st.table(pd.DataFrame({
        "Stage": ["clean_data", "featurize", "split", "train"],
        "What it does": ["Validate and clean the raw CSV",
                         "Create features in parallel with Ray",
                         "Stratified 80/20 train/test split",
                         "Train 3 models, track in MLflow, register champion"]}))

    st.subheader("Tools")
    c1, c2 = st.columns(2)
    c1.link_button("Open MLflow (experiments and registry)", MLFLOW_URL)
    c2.link_button("Open Grafana (monitoring)", GRAFANA_URL)
    st.info("Live monitoring panels will appear here once Prometheus and "
            "Grafana are added.")


PAGES = {"Single customer": page_single, "Batch upload": page_batch,
         "Pipeline and monitoring": page_pipeline}
st.sidebar.title("Churn Risk Predictor")
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.caption(f"API: {api.API_URL}")
PAGES[choice]()
