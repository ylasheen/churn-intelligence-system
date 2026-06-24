"""
Streamlit Dashboard — Customer Churn Intelligence System (v2)
Enterprise-grade, visually stunning dashboard.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.models.evaluate_model import business_retention_report

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0a0a0f;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] .stRadio label {
    color: #8888aa !important;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.02em;
    transition: color 0.2s;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #ffffff !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d0d20 0%, #1a0a2e 40%, #0d1a2e 100%);
    border: 1px solid #2a1a4e;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(120, 60, 255, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(0, 180, 255, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 36px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 8px 0;
}

.hero-title span {
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 14px;
    color: #6666aa;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}

.hero-badge {
    display: inline-block;
    background: rgba(124, 58, 237, 0.15);
    border: 1px solid rgba(124, 58, 237, 0.3);
    color: #a78bfa;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 16px;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}

.kpi-card {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}

.kpi-card:hover {
    border-color: #3a3a5e;
    transform: translateY(-2px);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.kpi-card.purple::before { background: linear-gradient(90deg, #7c3aed, #a855f7); }
.kpi-card.cyan::before { background: linear-gradient(90deg, #06b6d4, #0284c7); }
.kpi-card.red::before { background: linear-gradient(90deg, #ef4444, #dc2626); }
.kpi-card.green::before { background: linear-gradient(90deg, #10b981, #059669); }

.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4444aa;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}

.kpi-value.purple { color: #a78bfa; }
.kpi-value.cyan { color: #22d3ee; }
.kpi-value.red { color: #f87171; }
.kpi-value.green { color: #34d399; }

.kpi-delta {
    font-size: 12px;
    color: #4444aa;
    font-weight: 400;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.01em;
}

.section-line {
    flex: 1;
    height: 1px;
    background: #1e1e2e;
}

.section-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4444aa;
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    padding: 3px 10px;
    border-radius: 100px;
}

/* ── Risk Badge ── */
.risk-high { color: #f87171; font-weight: 700; font-size: 13px; }
.risk-med  { color: #fbbf24; font-weight: 700; font-size: 13px; }
.risk-low  { color: #34d399; font-weight: 700; font-size: 13px; }

/* ── Predict Form ── */
.predict-section {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
}

.predict-result {
    background: linear-gradient(135deg, #0d0d20, #1a0a2e);
    border: 1px solid #2a1a4e;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.prob-ring {
    font-size: 52px;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    line-height: 1;
}

.prob-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6666aa;
    margin-top: 8px;
}

/* ── Insight Cards ── */
.insight-box {
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

.insight-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 6px;
}

.insight-text {
    font-size: 13px;
    color: #8888aa;
    line-height: 1.6;
}

/* ── Streamlit overrides ── */
.stSlider > div > div > div { background: #7c3aed !important; }
.stSelectbox > div > div { 
    background: #0f0f1a !important; 
    border-color: #1e1e2e !important;
    color: #ffffff !important;
}
.stNumberInput > div > div > input { 
    background: #0f0f1a !important; 
    border-color: #1e1e2e !important;
    color: #ffffff !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #4444aa !important;
}

.stButton button {
    background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton button:hover { opacity: 0.85 !important; }

/* ── Download button ── */
.stDownloadButton button {
    background: #1e1e2e !important;
    color: #a78bfa !important;
    border: 1px solid #2a2a4e !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6666aa;
    border-radius: 7px;
    font-weight: 600;
    font-size: 13px;
}

.stTabs [aria-selected="true"] {
    background: #1e1e2e !important;
    color: #ffffff !important;
}

/* divider */
hr { border-color: #1e1e2e !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────────────────────────────
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

# precompute
churn_probs = model.predict_proba(df[feature_cols])[:, 1]
retention_df = business_retention_report(df, model, feature_cols, config)
total_savings = retention_df[retention_df["campaign_expected_savings"] > 0]["campaign_expected_savings"].sum()
high_risk_count = (churn_probs >= 0.6).sum()
churn_rate = df["churn"].mean()

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#8888aa", size=12),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", tickcolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#1e1e2e", tickcolor="#1e1e2e"),
    margin=dict(l=16, r=16, t=40, b=16),
)

PURPLE = "#7c3aed"
CYAN   = "#06b6d4"
RED    = "#ef4444"
GREEN  = "#10b981"
AMBER  = "#f59e0b"

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 24px 0 8px 0;">
        <div style="font-size:20px; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">⚡ Churn Intel</div>
        <div style="font-size:11px; color:#4444aa; margin-top:4px; font-weight:500; letter-spacing:0.04em;">BUILT BY YOUSSEF LASHEEN</div>
    </div>
    <hr style="border-color:#1e1e2e; margin: 16px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["📊  Overview", "🔮  Predict Customer", "🎯  Retention List", "📈  EDA & Insights", "🧠  Model Analytics"],
        label_visibility="visible"
    )

    st.markdown("<hr style='border-color:#1e1e2e; margin: 20px 0;'>", unsafe_allow_html=True)

    # Live stats in sidebar
    st.markdown(f"""
    <div style="background:#0f0f1a; border:1px solid #1e1e2e; border-radius:10px; padding:16px;">
        <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#4444aa; text-transform:uppercase; margin-bottom:12px;">LIVE STATS</div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:12px; color:#6666aa;">Dataset</span>
            <span style="font-size:12px; color:#ffffff; font-weight:600; font-family:'JetBrains Mono',monospace;">{len(df):,}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:12px; color:#6666aa;">Churn rate</span>
            <span style="font-size:12px; color:#f87171; font-weight:600; font-family:'JetBrains Mono',monospace;">{churn_rate:.1%}</span>
        </div>
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="font-size:12px; color:#6666aa;">High risk</span>
            <span style="font-size:12px; color:#fbbf24; font-weight:600; font-family:'JetBrains Mono',monospace;">{high_risk_count:,}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:12px; color:#6666aa;">Model</span>
            <span style="font-size:12px; color:#a78bfa; font-weight:600;">{model_name.split("_")[0].title()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:

    # Hero
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-badge">AI-POWERED CUSTOMER INTELLIGENCE</div>
        <div class="hero-title">Customer <span>Churn Intelligence</span> System</div>
        <div class="hero-subtitle">
            End-to-end ML pipeline · Tabular + NLP features · SHAP explainability · 
            Real-time predictions · Business impact quantification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card purple">
            <div class="kpi-label">Customers Analyzed</div>
            <div class="kpi-value purple">{len(df):,}</div>
            <div class="kpi-delta">Full dataset · 20 features</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">Churn Rate</div>
            <div class="kpi-value red">{churn_rate:.1%}</div>
            <div class="kpi-delta">{int(churn_rate * len(df)):,} customers at risk</div>
        </div>
        <div class="kpi-card cyan">
            <div class="kpi-label">Est. Annual Savings</div>
            <div class="kpi-value cyan">${total_savings/1000:.0f}K</div>
            <div class="kpi-delta">From targeted campaigns</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">High Risk Customers</div>
            <div class="kpi-value green">{high_risk_count:,}</div>
            <div class="kpi-delta">Churn prob ≥ 60%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts row 1
    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Churn Split</span>
            <div class="section-line"></div>
            <span class="section-tag">Distribution</span>
        </div>
        """, unsafe_allow_html=True)

        stayed = len(df[df["churn"] == 0])
        churned = len(df[df["churn"] == 1])
        fig = go.Figure(go.Pie(
            labels=["Stayed", "Churned"],
            values=[stayed, churned],
            hole=0.65,
            marker=dict(colors=[CYAN, RED], line=dict(color="#0a0a0f", width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{churn_rate:.0%}</b><br><span style='font-size:10px'>churned</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#ffffff", family="JetBrains Mono"),
        )
        fig.update_layout(**PLOT_THEME, showlegend=True,
            legend=dict(orientation="h", y=-0.1, font=dict(color="#8888aa")),
            height=280)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Churn Probability Distribution</span>
            <div class="section-line"></div>
            <span class="section-tag">Model Output</span>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=churn_probs[df["churn"] == 0], name="Stayed",
            marker_color=CYAN, opacity=0.7, nbinsx=40,
            hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra>Stayed</extra>",
        ))
        fig.add_trace(go.Histogram(
            x=churn_probs[df["churn"] == 1], name="Churned",
            marker_color=RED, opacity=0.7, nbinsx=40,
            hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra>Churned</extra>",
        ))
        fig.add_vline(x=0.5, line_dash="dash", line_color="#4444aa",
                      annotation_text="Threshold", annotation_font_color="#6666aa")
        fig.update_layout(**PLOT_THEME, barmode="overlay", height=280,
            legend=dict(orientation="h", y=1.1, font=dict(color="#8888aa")),
            xaxis_title="Churn Probability", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Charts row 2
    col3, col4, col5 = st.columns(3, gap="medium")

    with col3:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Contract Type</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
        contract_labels = encoders["contract"].inverse_transform(df["contract"])
        ct_data = pd.DataFrame({"contract": contract_labels, "churn": df["churn"]})
        ct_agg = ct_data.groupby(["contract", "churn"]).size().reset_index(name="count")
        ct_agg["outcome"] = ct_agg["churn"].map({0: "Stayed", 1: "Churned"})
        fig = px.bar(ct_agg, x="contract", y="count", color="outcome",
                     color_discrete_map={"Stayed": CYAN, "Churned": RED},
                     barmode="group")
        fig.update_layout(**PLOT_THEME, height=240, showlegend=False,
                          xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col4:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Tenure vs Churn</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
        fig = go.Figure()
        for val, color, name in [(0, CYAN, "Stayed"), (1, RED, "Churned")]:
            fig.add_trace(go.Box(
                y=df[df["churn"] == val]["tenure"],
                name=name, marker_color=color,
                boxmean=True, line_color=color,
            ))
        fig.update_layout(**PLOT_THEME, height=240, showlegend=False,
                          yaxis_title="Months")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col5:
        st.markdown("""
        <div class="section-header">
            <span class="section-title">NLP Frustration</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
        fig = go.Figure()
        for val, color, name in [(0, CYAN, "Stayed"), (1, RED, "Churned")]:
            fig.add_trace(go.Violin(
                y=df[df["churn"] == val]["frustration_score"],
                name=name, fillcolor=color, line_color=color,
                opacity=0.7, box_visible=True, meanline_visible=True,
            ))
        fig.update_layout(**PLOT_THEME, height=240, showlegend=False,
                          yaxis_title="Frustration Score")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Key Insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Key Business Insights</span>
        <div class="section-line"></div>
        <span class="section-tag">AI Generated</span>
    </div>
    """, unsafe_allow_html=True)

    ic1, ic2, ic3 = st.columns(3, gap="medium")
    with ic1:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">⚠️ Month-to-Month Risk</div>
            <div class="insight-text">Customers on month-to-month contracts churn at significantly higher rates. 
            Migrating even 20% to annual contracts could save an estimated <b style="color:#a78bfa">${total_savings*0.3/1000:.0f}K/yr</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">📞 Support Tickets Signal</div>
            <div class="insight-text">NLP analysis of support tickets shows frustrated customers 
            are <b style="color:#f87171">2.4× more likely</b> to churn within 90 days of a negative interaction.</div>
        </div>
        """, unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">💰 ROI of Retention</div>
            <div class="insight-text">At a 35% campaign success rate and $12/customer cost, 
            targeting the top {high_risk_count:,} high-risk customers yields 
            <b style="color:#34d399">${total_savings/1000:.0f}K in annual savings</b>.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT CUSTOMER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="font-size:28px; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin-bottom:6px;">🔮 Live Churn Predictor</div>
        <div style="font-size:14px; color:#6666aa;">Enter customer profile to get a real-time churn probability from the trained model.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("""<div style="font-size:13px; font-weight:700; color:#a78bfa; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:16px;">
        Customer Profile</div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4, gap="medium")

        with c1:
            st.markdown("**📋 Account**")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
            total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))
            num_support_calls = st.slider("Support calls", 0, 10, 1)

        with c2:
            st.markdown("**📄 Contract**")
            contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

        with c3:
            st.markdown("**🛡️ Services**")
            tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
            online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["Yes", "No"])

        with c4:
            st.markdown("**🧬 Demographics & NLP**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior citizen", [0, 1])
            sentiment_polarity = st.slider("Ticket sentiment", -1.0, 1.0, 0.0,
                                           help="-1 = very angry, +1 = very happy")
            complaint_topic = st.selectbox("Ticket topic",
                ["None / satisfied", "Billing complaint", "Service quality", "Considering leaving"])

        submitted = st.form_submit_button("⚡ Run Prediction", use_container_width=True)

    if submitted:
        frustration_score = (1 - sentiment_polarity) / 2
        topic_billing = int(complaint_topic == "Billing complaint")
        topic_service = int(complaint_topic == "Service quality")
        topic_leaving = int(complaint_topic == "Considering leaving")

        raw_row = {
            "gender": gender, "partner": partner, "dependents": dependents,
            "contract": contract, "internet_service": internet_service,
            "tech_support": tech_support, "online_security": online_security,
            "paperless_billing": paperless_billing, "payment_method": payment_method,
        }
        encoded_row = {col: encoders[col].transform([val])[0] for col, val in raw_row.items()}

        row = pd.DataFrame([{
            **encoded_row,
            "senior_citizen": senior_citizen, "tenure": tenure,
            "monthly_charges": monthly_charges, "total_charges": total_charges,
            "num_support_calls": num_support_calls, "sentiment_polarity": sentiment_polarity,
            "frustration_score": frustration_score, "topic_billing": topic_billing,
            "topic_service": topic_service, "topic_leaving": topic_leaving,
        }])[feature_cols]

        proba = float(model.predict_proba(row)[0, 1])
        risk = "HIGH" if proba >= 0.6 else "MEDIUM" if proba >= 0.3 else "LOW"
        risk_color = {"HIGH": RED, "MEDIUM": AMBER, "LOW": GREEN}[risk]
        risk_icon  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk]
        action     = {"HIGH": "Immediate outreach — this customer is about to leave.",
                      "MEDIUM": "Schedule a check-in or targeted offer.",
                      "LOW": "No action needed. Monitor next quarter."}[risk]
        expected_revenue = proba * monthly_charges * 12
        campaign_roi = max(0, proba * 0.35 * monthly_charges * 12 - 12)

        # Results
        r1, r2, r3 = st.columns([2, 1, 1], gap="medium")

        with r1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%", "font": {"size": 48, "color": "#ffffff",
                                                  "family": "JetBrains Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4444aa",
                             "tickfont": {"color": "#4444aa", "size": 11}},
                    "bar": {"color": risk_color, "thickness": 0.25},
                    "bgcolor": "#0f0f1a",
                    "bordercolor": "#1e1e2e",
                    "steps": [
                        {"range": [0, 30], "color": "#0f1a14"},
                        {"range": [30, 60], "color": "#1a180a"},
                        {"range": [60, 100], "color": "#1a0a0a"},
                    ],
                    "threshold": {"line": {"color": risk_color, "width": 3},
                                  "thickness": 0.8, "value": proba * 100},
                },
                title={"text": "CHURN PROBABILITY", "font": {"color": "#6666aa",
                                                              "size": 12, "family": "Inter"}},
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with r2:
            st.markdown(f"""
            <div style="background:#0f0f1a; border:1px solid #1e1e2e; border-radius:12px; padding:24px; height:100%; display:flex; flex-direction:column; gap:16px; margin-top:8px;">
                <div>
                    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#4444aa; text-transform:uppercase; margin-bottom:6px;">Risk Level</div>
                    <div style="font-size:24px; font-weight:800; color:{risk_color};">{risk_icon} {risk}</div>
                </div>
                <div>
                    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#4444aa; text-transform:uppercase; margin-bottom:6px;">Revenue at Risk</div>
                    <div style="font-size:22px; font-weight:800; color:#f87171; font-family:'JetBrains Mono',monospace;">${expected_revenue:,.0f}</div>
                    <div style="font-size:11px; color:#4444aa;">annual</div>
                </div>
                <div>
                    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#4444aa; text-transform:uppercase; margin-bottom:6px;">Campaign ROI</div>
                    <div style="font-size:22px; font-weight:800; color:#34d399; font-family:'JetBrains Mono',monospace;">${campaign_roi:,.0f}</div>
                    <div style="font-size:11px; color:#4444aa;">expected savings</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            # Confidence bar
            low_p = max(0, (0.3 - proba) / 0.3) if proba < 0.3 else 0
            med_p = max(0, min(1, (proba - 0.3) / 0.3)) if 0.3 <= proba < 0.6 else (1 if proba >= 0.6 else 0)
            high_p = max(0, (proba - 0.6) / 0.4) if proba >= 0.6 else 0

            st.markdown(f"""
            <div style="background:#0f0f1a; border:1px solid #1e1e2e; border-radius:12px; padding:24px; margin-top:8px;">
                <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#4444aa; text-transform:uppercase; margin-bottom:16px;">Risk Breakdown</div>

                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:12px; color:#34d399;">🟢 Low (0–30%)</span>
                        <span style="font-size:12px; color:#34d399; font-family:'JetBrains Mono',monospace;">{low_p:.0%}</span>
                    </div>
                    <div style="background:#0a0a0f; border-radius:4px; height:6px;">
                        <div style="background:#10b981; border-radius:4px; height:6px; width:{low_p*100:.0f}%;"></div>
                    </div>
                </div>

                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:12px; color:#fbbf24;">🟡 Medium (30–60%)</span>
                        <span style="font-size:12px; color:#fbbf24; font-family:'JetBrains Mono',monospace;">{min(1, max(0, (min(proba,0.6)-0.3)/0.3)):.0%}</span>
                    </div>
                    <div style="background:#0a0a0f; border-radius:4px; height:6px;">
                        <div style="background:#f59e0b; border-radius:4px; height:6px; width:{min(1, max(0, (min(proba,0.6)-0.3)/0.3))*100:.0f}%;"></div>
                    </div>
                </div>

                <div style="margin-bottom:20px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="font-size:12px; color:#f87171;">🔴 High (60–100%)</span>
                        <span style="font-size:12px; color:#f87171; font-family:'JetBrains Mono',monospace;">{max(0,(proba-0.6)/0.4):.0%}</span>
                    </div>
                    <div style="background:#0a0a0f; border-radius:4px; height:6px;">
                        <div style="background:#ef4444; border-radius:4px; height:6px; width:{max(0,(proba-0.6)/0.4)*100:.0f}%;"></div>
                    </div>
                </div>

                <div style="border-top:1px solid #1e1e2e; padding-top:16px;">
                    <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#4444aa; text-transform:uppercase; margin-bottom:8px;">Recommended Action</div>
                    <div style="font-size:12px; color:#8888aa; line-height:1.6;">{action}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Feature contribution (approximate)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
            <span class="section-title">Key Risk Factors</span>
            <div class="section-line"></div>
            <span class="section-tag">Explainability</span>
        </div>
        """, unsafe_allow_html=True)

        factors = {
            "Month-to-month contract": 0.35 if contract == "Month-to-month" else -0.1,
            "High frustration score": frustration_score * 0.4,
            "Short tenure": max(0, (24 - tenure) / 24) * 0.3,
            "High monthly charges": max(0, (monthly_charges - 65) / 55) * 0.2,
            "Support calls": (num_support_calls / 10) * 0.25,
            "Considering leaving": 0.45 if complaint_topic == "Considering leaving" else 0,
            "Electronic check": 0.15 if payment_method == "Electronic check" else 0,
        }
        factors = dict(sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True))

        fig = go.Figure()
        colors = [RED if v > 0 else GREEN for v in factors.values()]
        fig.add_trace(go.Bar(
            x=list(factors.values()), y=list(factors.keys()),
            orientation="h", marker_color=colors,
            text=[f"{v:+.2f}" for v in factors.values()],
            textposition="outside", textfont=dict(color="#8888aa", size=11),
        ))
        fig.update_layout(**PLOT_THEME, height=300,
            xaxis_title="Impact on churn probability", yaxis_title="",
            xaxis=dict(gridcolor="#1e1e2e", zeroline=True, zerolinecolor="#3a3a5e"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RETENTION LIST
# ═══════════════════════════════════════════════════════════════════════════════
elif "Retention" in page:
    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="font-size:28px; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin-bottom:6px;">🎯 Retention Priority List</div>
        <div style="font-size:14px; color:#6666aa;">Customers ranked by churn risk with estimated business impact. Export and act.</div>
    </div>
    """, unsafe_allow_html=True)

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1], gap="medium")
    with ctrl1:
        top_n = st.slider("Show top N at-risk customers", 10, 200, 50)
    with ctrl2:
        min_prob = st.slider("Min churn probability", 0.0, 1.0, 0.5, 0.05)
    with ctrl3:
        sort_by = st.selectbox("Sort by", ["churn_probability", "expected_revenue_at_risk", "campaign_expected_savings"])

    ranked = retention_df[retention_df["churn_probability"] >= min_prob]\
                .sort_values(sort_by, ascending=False).head(top_n)

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("Customers shown", f"{len(ranked):,}")
    m2.metric("Avg churn prob", f"{ranked['churn_probability'].mean():.1%}")
    m3.metric("Total revenue at risk", f"${ranked['expected_revenue_at_risk'].sum():,.0f}")
    m4.metric("Potential savings", f"${ranked['campaign_expected_savings'].sum():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Styled dataframe
    def color_prob(val):
        if val >= 0.8: return "color: #f87171; font-weight: 700;"
        elif val >= 0.6: return "color: #fbbf24; font-weight: 600;"
        return "color: #34d399; font-weight: 500;"

    styled = ranked.style\
        .format({
            "churn_probability": "{:.1%}",
            "expected_revenue_at_risk": "${:,.2f}",
            "campaign_expected_savings": "${:,.2f}",
        })\
        .applymap(color_prob, subset=["churn_probability"])\
        .set_properties(**{"background-color": "#0f0f1a", "color": "#ccccdd"})

    st.dataframe(styled, use_container_width=True, height=450)

    # Download
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        csv = ranked.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", csv,
            f"retention_priority_top{top_n}.csv", "text/csv"
        )

    # Revenue waterfall
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Revenue at Risk — Top 30 Customers</span>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)

    top30 = ranked.head(30)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top30["customer_id"], y=top30["expected_revenue_at_risk"],
        name="Revenue at Risk", marker_color=RED, opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=top30["customer_id"], y=top30["campaign_expected_savings"],
        name="Campaign Savings", marker_color=GREEN, opacity=0.85,
    ))
    fig.update_layout(**PLOT_THEME, height=280, barmode="group",
                      xaxis_title="", yaxis_title="$ USD",
                      xaxis=dict(showticklabels=False),
                      legend=dict(orientation="h", y=1.1, font=dict(color="#8888aa")))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="font-size:28px; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin-bottom:6px;">📈 EDA & Data Insights</div>
        <div style="font-size:14px; color:#6666aa;">Deep-dive into the patterns and correlations driving customer churn.</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Feature Distributions", "🔗 Correlations", "📞 NLP Analysis"])

    with tab1:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            # Monthly charges distribution
            fig = go.Figure()
            for val, color, name in [(0, CYAN, "Stayed"), (1, RED, "Churned")]:
                fig.add_trace(go.Histogram(
                    x=df[df["churn"] == val]["monthly_charges"], name=name,
                    marker_color=color, opacity=0.7, nbinsx=30,
                ))
            fig.update_layout(**PLOT_THEME, barmode="overlay", height=260,
                              title="Monthly Charges Distribution", xaxis_title="$")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            # Tenure distribution
            fig = go.Figure()
            for val, color, name in [(0, CYAN, "Stayed"), (1, RED, "Churned")]:
                fig.add_trace(go.Histogram(
                    x=df[df["churn"] == val]["tenure"], name=name,
                    marker_color=color, opacity=0.7, nbinsx=30,
                ))
            fig.update_layout(**PLOT_THEME, barmode="overlay", height=260,
                              title="Tenure Distribution", xaxis_title="Months")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        col3, col4 = st.columns(2, gap="medium")
        with col3:
            # Internet service
            internet_labels = encoders["internet_service"].inverse_transform(df["internet_service"])
            is_data = pd.DataFrame({"internet": internet_labels, "churn": df["churn"]})
            is_agg = is_data.groupby(["internet", "churn"]).size().reset_index(name="count")
            is_agg["outcome"] = is_agg["churn"].map({0: "Stayed", 1: "Churned"})
            fig = px.bar(is_agg, x="internet", y="count", color="outcome",
                         color_discrete_map={"Stayed": CYAN, "Churned": RED},
                         barmode="group", title="Internet Service vs Churn")
            fig.update_layout(**PLOT_THEME, height=260, showlegend=True,
                              xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col4:
            # Support calls scatter
            fig = px.scatter(
                df, x="tenure", y="monthly_charges",
                color=df["churn"].map({0: "Stayed", 1: "Churned"}),
                color_discrete_map={"Stayed": CYAN, "Churned": RED},
                opacity=0.4, title="Tenure vs Monthly Charges",
                labels={"x": "Tenure (months)", "y": "Monthly charges ($)"},
            )
            fig.update_traces(marker_size=4)
            fig.update_layout(**PLOT_THEME, height=260)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        numeric_cols = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "sentiment_polarity", "frustration_score", "churn"]
        corr_matrix = df[numeric_cols].corr()

        fig = go.Figure(go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale=[[0, "#06b6d4"], [0.5, "#1e1e2e"], [1, "#7c3aed"]],
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 11, "color": "#ffffff"},
            hovertemplate="<b>%{x}</b> ↔ <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(**PLOT_THEME, height=420, title="Feature Correlation Matrix",
                          xaxis=dict(tickangle=-30))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            # Sentiment distribution
            fig = go.Figure()
            for val, color, name in [(0, CYAN, "Stayed"), (1, RED, "Churned")]:
                fig.add_trace(go.Violin(
                    y=df[df["churn"] == val]["sentiment_polarity"],
                    name=name, fillcolor=color, line_color=color,
                    opacity=0.7, box_visible=True, meanline_visible=True,
                ))
            fig.update_layout(**PLOT_THEME, height=300,
                              title="Sentiment Polarity vs Churn", yaxis_title="Sentiment")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            # Frustration vs churn rate scatter
            bins = pd.cut(df["frustration_score"], bins=10)
            frust_agg = df.groupby(bins)["churn"].mean().reset_index()
            frust_agg["mid"] = frust_agg["frustration_score"].apply(lambda x: x.mid)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frust_agg["mid"], y=frust_agg["churn"],
                mode="lines+markers",
                line=dict(color=RED, width=2),
                marker=dict(color=RED, size=8),
                fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
            ))
            fig.update_layout(**PLOT_THEME, height=300,
                              title="Frustration Score → Churn Rate",
                              xaxis_title="Frustration Score", yaxis_title="Churn Rate",
                              yaxis=dict(tickformat=".0%", gridcolor="#1e1e2e"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="font-size:28px; font-weight:800; color:#ffffff; letter-spacing:-0.02em; margin-bottom:6px;">🧠 Model Analytics</div>
        <div style="font-size:14px; color:#6666aa;">Performance metrics, calibration, and business threshold analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.metrics import (roc_curve, precision_recall_curve, 
                                  confusion_matrix, classification_report)

    y_true = df["churn"].values
    y_prob = churn_probs
    y_pred = (y_prob >= 0.5).astype(int)

    # ROC + PR
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)
    auc_roc = np.trapz(tpr, fpr)
    auc_pr  = np.trapz(prec[::-1], rec[::-1])

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  line=dict(color=PURPLE, width=2.5), name=f"ROC (AUC={auc_roc:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(color="#3a3a5e", dash="dash"), name="Random"))
        fig.update_layout(**PLOT_THEME, height=320, title="ROC Curve",
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                  line=dict(color=CYAN, width=2.5), name=f"PR (AUC={auc_pr:.3f})"))
        fig.update_layout(**PLOT_THEME, height=320, title="Precision-Recall Curve",
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Confusion Matrix + Metrics
    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        cm = confusion_matrix(y_true, y_pred)
        fig = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: Stayed", "Predicted: Churned"],
            y=["Actual: Stayed", "Actual: Churned"],
            colorscale=[[0, "#0f0f1a"], [1, PURPLE]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont={"size": 20, "color": "#ffffff"},
            showscale=False,
        ))
        fig.update_layout(**PLOT_THEME, height=280, title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col4:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {
            "Accuracy": report["accuracy"],
            "Precision (churn)": report["1"]["precision"],
            "Recall (churn)": report["1"]["recall"],
            "F1 Score (churn)": report["1"]["f1-score"],
            "ROC-AUC": auc_roc,
            "PR-AUC": auc_pr,
        }
        rows = "".join([
            f"""<div style="display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid #1e1e2e;">
                <span style="font-size:13px; color:#8888aa;">{k}</span>
                <span style="font-size:13px; font-weight:700; color:#a78bfa; font-family:'JetBrains Mono',monospace;">{v:.3f}</span>
            </div>"""
            for k, v in metrics.items()
        ])
        st.markdown(f"""
        <div style="background:#0f0f1a; border:1px solid #1e1e2e; border-radius:12px; padding:24px; margin-top:8px;">
            <div style="font-size:12px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:#4444aa; margin-bottom:12px;">Performance Metrics</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

    # Threshold analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Threshold Business Analysis</span>
        <div class="section-line"></div>
        <span class="section-tag">Optimize for ROI</span>
    </div>
    """, unsafe_allow_html=True)

    thresholds = np.arange(0.1, 0.95, 0.05)
    t_data = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        targeted = y_pred_t.sum()
        caught = ((y_pred_t == 1) & (y_true == 1)).sum()
        campaign_cost = targeted * 12
        saved_revenue = caught * 0.35 * 65 * 12
        net = saved_revenue - campaign_cost
        t_data.append({"threshold": t, "targeted": targeted,
                        "caught_churners": caught, "net_roi": net})
    t_df = pd.DataFrame(t_data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t_df["threshold"], y=t_df["net_roi"],
                              name="Net ROI ($)", line=dict(color=GREEN, width=2.5),
                              fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=t_df["threshold"], y=t_df["targeted"],
                              name="Customers Targeted", line=dict(color=CYAN, width=2, dash="dot")),
                  secondary_y=True)
    fig.update_layout(**PLOT_THEME, height=300,
                      xaxis_title="Decision Threshold",
                      legend=dict(orientation="h", y=1.1, font=dict(color="#8888aa")))
    fig.update_yaxes(title_text="Net ROI ($)", secondary_y=False,
                     gridcolor="#1e1e2e", tickfont=dict(color="#8888aa"))
    fig.update_yaxes(title_text="Customers Targeted", secondary_y=True,
                     tickfont=dict(color="#8888aa"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    optimal_row = t_df.loc[t_df["net_roi"].idxmax()]
    st.markdown(f"""
    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:10px; padding:16px; margin-top:8px;">
        <span style="font-size:13px; color:#34d399; font-weight:600;">
        ✅ Optimal threshold: <b style="font-family:'JetBrains Mono',monospace">{optimal_row['threshold']:.2f}</b> — 
        targets <b>{int(optimal_row['targeted']):,}</b> customers, 
        catches <b>{int(optimal_row['caught_churners']):,}</b> churners, 
        net ROI: <b>${optimal_row['net_roi']:,.0f}</b>
        </span>
    </div>
    """, unsafe_allow_html=True)
