"""
Streamlit Dashboard — Customer Churn Intelligence System
----------------------------------------------------------
Run with:
    streamlit run dashboard/app.py

Provides a non-technical, business-facing view of the churn model:
overview KPIs, a live prediction form, a retention priority list,
and EDA insights.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.models.evaluate_model import business_retention_report  # noqa: E402

st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="📊",
    layout="wide",
)

config = load_config()


@st.cache_resource
def load_model_bundle():
    bundle = joblib.load(ROOT / config["model"]["saved_model_path"])
    return bundle["model"], bundle["features"], bundle["name"]


@st.cache_data
def load_processed_data():
    return pd.read_csv(ROOT / config["data"]["processed_path"])


@st.cache_resource
def load_encoders():
    return joblib.load(ROOT / config["model"]["encoder_path"])


model, feature_cols, model_name = load_model_bundle()
df = load_processed_data()
encoders = load_encoders()

st.sidebar.title("📊 Churn Intelligence")
st.sidebar.caption("Built by Youssef Lasheen — AI & ML Engineer")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict a Customer", "Retention Priority List", "Data Insights"],
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Customer Churn Intelligence System")
    st.caption(
        "An end-to-end ML system that predicts churn using tabular customer data + "
        "NLP support-ticket analysis, and translates predictions into business actions."
    )

    retention_df = business_retention_report(df, model, feature_cols, config)
    total_savings = retention_df[retention_df["campaign_expected_savings"] > 0][
        "campaign_expected_savings"
    ].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers analyzed", f"{len(df):,}")
    col2.metric("Churn rate", f"{df['churn'].mean():.1%}")
    col3.metric("Model in use", model_name.replace("_", " ").title())
    col4.metric("Est. annual savings from retention", f"${total_savings:,.0f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Churn distribution")
        fig = px.pie(
            df, names=df["churn"].map({0: "Stayed", 1: "Churned"}),
            title="Customer Churn Split", hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Churn probability distribution")
        probs = model.predict_proba(df[feature_cols])[:, 1]
        fig = px.histogram(probs, nbins=30, title="Predicted Churn Probability")
        fig.update_layout(showlegend=False, xaxis_title="Churn probability")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Predict a Customer
# ---------------------------------------------------------------------------
elif page == "Predict a Customer":
    st.title("🔮 Predict Churn for a Customer")
    st.caption("Enter customer details to get a live churn prediction from the trained model.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
            total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, monthly_charges * tenure)
            num_support_calls = st.slider("Support calls (last period)", 0, 10, 1)

        with c2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox(
                "Payment method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

        with c3:
            tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])

        c4, c5 = st.columns(2)
        with c4:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior citizen", [0, 1])
        with c5:
            sentiment_polarity = st.slider("Latest ticket sentiment (-1 angry, +1 happy)", -1.0, 1.0, 0.0)
            complaint_topic = st.selectbox(
                "Latest ticket topic", ["None / satisfied", "Billing complaint", "Service quality", "Considering leaving"]
            )

        submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

    if submitted:
        frustration_score = (1 - sentiment_polarity) / 2
        topic_billing = int(complaint_topic == "Billing complaint")
        topic_service = int(complaint_topic == "Service quality")
        topic_leaving = int(complaint_topic == "Considering leaving")

        raw_row = {
            "gender": gender,
            "partner": partner,
            "dependents": dependents,
            "contract": contract,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
        }
        encoded_row = {col: encoders[col].transform([val])[0] for col, val in raw_row.items()}

        row = pd.DataFrame([{
            **encoded_row,
            "senior_citizen": senior_citizen,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
            "sentiment_polarity": sentiment_polarity,
            "frustration_score": frustration_score,
            "topic_billing": topic_billing,
            "topic_service": topic_service,
            "topic_leaving": topic_leaving,
        }])[feature_cols]

        proba = float(model.predict_proba(row)[0, 1])
        risk = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"
        color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk]

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Churn probability", f"{proba:.1%}")
        col2.metric("Risk level", f"{color} {risk}")
        col3.metric("Recommended action", "Contact now" if risk == "High" else "Monitor" if risk == "Medium" else "No action needed")

        expected_revenue = proba * monthly_charges * 12
        st.info(f"Estimated annual revenue at risk for this customer: **${expected_revenue:,.2f}**")

# ---------------------------------------------------------------------------
# Retention Priority List
# ---------------------------------------------------------------------------
elif page == "Retention Priority List":
    st.title("📋 Retention Priority List")
    st.caption("Customers ranked by churn risk, with estimated business impact of targeting them.")

    retention_df = business_retention_report(df, model, feature_cols, config)
    top_n = st.slider("Show top N at-risk customers", 5, 100, 20)
    ranked = retention_df.sort_values("churn_probability", ascending=False).head(top_n)

    st.dataframe(
        ranked.style.format({
            "churn_probability": "{:.1%}",
            "expected_revenue_at_risk": "${:,.2f}",
            "campaign_expected_savings": "${:,.2f}",
        }),
        use_container_width=True,
    )

    csv = ranked.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", csv, "retention_priority_list.csv", "text/csv")

# ---------------------------------------------------------------------------
# Data Insights
# ---------------------------------------------------------------------------
elif page == "Data Insights":
    st.title("📈 Data Insights")

    col1, col2 = st.columns(2)
    with col1:
        contract_labels = encoders["contract"].inverse_transform(df["contract"])
        fig = px.histogram(
            x=contract_labels, color=df["churn"].map({0: "Stayed", 1: "Churned"}),
            barmode="group", title="Contract Type vs Churn",
        )
        fig.update_layout(xaxis_title="Contract type", legend_title="Outcome")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x=df["churn"].map({0: "Stayed", 1: "Churned"}), y="tenure",
            title="Tenure vs Churn",
        )
        fig.update_layout(xaxis_title="Outcome")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(
            df, x=df["churn"].map({0: "Stayed", 1: "Churned"}), y="monthly_charges",
            title="Monthly Charges vs Churn",
        )
        fig.update_layout(xaxis_title="Outcome")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            df, x=df["churn"].map({0: "Stayed", 1: "Churned"}), y="frustration_score",
            title="NLP Frustration Score vs Churn",
        )
        fig.update_layout(xaxis_title="Outcome")
        st.plotly_chart(fig, use_container_width=True)
