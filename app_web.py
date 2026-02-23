"""
T2 Diabetes Predictor — Streamlit UI
======================================
Run with:  streamlit run app_web.py

UI layer only. All prediction/preprocessing logic lives in app.py.
"""

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV  # needed for unpickling

# ── Single source of truth: all logic from app.py ────────────────────────────
from app import load_model_bundle, preprocess_input, predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="T2 Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background-color: #060d18; color: #cdd8e8; }
[data-testid="stSidebar"] { background-color: #0c1827; border-right: 1px solid #1a2e44; }
.app-header {
    background: linear-gradient(135deg, #0c1827 0%, #111f30 100%);
    border: 1px solid #1a2e44; border-radius: 16px; padding: 28px 36px; margin-bottom: 28px;
}
.app-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; }
.app-title em { color: #00d4a0; font-style: normal; }
.app-subtitle { color: #4a6080; font-size: 0.9rem; margin-top: 6px; }
.badge {
    display: inline-block; background: #111f30; border: 1px solid #1a2e44;
    color: #00d4a0; font-size: 0.7rem; padding: 3px 10px;
    border-radius: 100px; margin-right: 6px; font-family: monospace;
}
.metric-card { background: #0c1827; border: 1px solid #1a2e44; border-radius: 14px; padding: 22px 26px; text-align: center; }
.metric-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #4a6080; }
.metric-value { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1.1; }
.risk-banner { border-radius: 14px; padding: 20px 28px; margin-bottom: 20px; display: flex; align-items: center; gap: 16px; }
.risk-low   { background: rgba(0,212,160,0.08);  border: 1px solid rgba(0,212,160,0.25); }
.risk-mod   { background: rgba(240,192,64,0.08); border: 1px solid rgba(240,192,64,0.25); }
.risk-high  { background: rgba(240,128,64,0.08); border: 1px solid rgba(240,128,64,0.25); }
.risk-vhigh { background: rgba(224,64,96,0.08);  border: 1px solid rgba(224,64,96,0.25); }
.disclaimer {
    background: rgba(74,96,128,0.12); border: 1px solid #1a2e44;
    border-radius: 10px; padding: 14px 18px; font-size: 0.78rem; color: #4a6080; margin-top: 20px; line-height: 1.6;
}
.sidebar-section {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;
    color: #00d4a0; margin: 20px 0 8px 0; padding-bottom: 6px; border-bottom: 1px solid #1a2e44;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select { background-color: #111f30 !important; border-color: #1a2e44 !important; color: #cdd8e8 !important; }
div[data-testid="metric-container"] { background: #0c1827; border: 1px solid #1a2e44; border-radius: 12px; padding: 14px; }
</style>
""", unsafe_allow_html=True)

# ── UI-only constants ─────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "age_bmi_interaction":  "Age × BMI Interaction",
    "ethnicity":            "Ethnicity",
    "age_years":            "Age (years)",
    "waist_height_ratio":   "Waist-Height Ratio",
    "triglyceride_ratio":   "Triglyceride Ratio",
    "cholesterol_ratio":    "Cholesterol Ratio",
    "hypertension":         "Hypertension",
    "hdl_cholesterol":      "HDL Cholesterol (mg/dL)",
    "systolic_bp":          "Systolic Blood Pressure",
    "income_poverty_ratio": "Income / Poverty Ratio",
    "total_cholesterol":    "Total Cholesterol (mg/dL)",
    "creatinine":           "Creatinine (mg/dL)",
    "waist_cm":             "Waist Circumference",
    "diastolic_bp":         "Diastolic Blood Pressure",
    "weight_kg":            "Weight",
    "sleep_hours":          "Sleep Hours",
    "height_cm":            "Height",
    "bmi":                  "BMI",
    "triglycerides":        "Triglycerides (mg/dL)",
    "age_group":            "Age Group",
}

ETHNICITY_MAP = {
    "Mexican American": 0, "Other Hispanic": 1, "Non-Hispanic White": 2,
    "Non-Hispanic Black": 3, "Non-Hispanic Asian": 4, "Other / Mixed": 5,
}

RISK_CONFIG = {
    "Low":       {"color": "#00d4a0", "emoji": "✅", "css": "risk-low",   "desc": "Your predicted risk is low. Maintain a healthy lifestyle."},
    "Moderate":  {"color": "#f0c040", "emoji": "⚠️",  "css": "risk-mod",   "desc": "Moderate risk detected. Consider consulting your doctor for preventive screening."},
    "High":      {"color": "#f08040", "emoji": "🔶", "css": "risk-high",  "desc": "High risk. Clinical evaluation and lifestyle interventions are strongly recommended."},
    "Very High": {"color": "#e04060", "emoji": "🔴", "css": "risk-vhigh", "desc": "Very high risk. Please seek medical attention promptly."},
}

# ── Load model via app.py + SHAP explainer ────────────────────────────────────
@st.cache_resource
def load_resources():
    model, threshold, _ = load_model_bundle()
    explainer = None
    explainer_path = Path("models/shap_explainer.pkl")
    if explainer_path.exists():
        explainer = joblib.load(explainer_path)
    return model, explainer, threshold

model, explainer, THRESHOLD = load_resources()

# ── Plotly gauge ──────────────────────────────────────────────────────────────
def make_gauge(probability: float, risk_level: str) -> go.Figure:
    color = RISK_CONFIG[risk_level]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 48, "color": color, "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a6080", "tickfont": {"color": "#4a6080", "size": 11}},
            "bar": {"color": color, "thickness": 0.22},
            "bgcolor": "#060d18", "borderwidth": 0,
            "steps": [
                {"range": [0,  25],  "color": "rgba(0,212,160,0.08)"},
                {"range": [25, 50],  "color": "rgba(240,192,64,0.08)"},
                {"range": [50, 70],  "color": "rgba(240,128,64,0.08)"},
                {"range": [70, 100], "color": "rgba(224,64,96,0.08)"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.75, "value": THRESHOLD * 100},
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=30, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"family": "Syne"},
    )
    return fig

# ── Plotly SHAP chart ─────────────────────────────────────────────────────────
def make_shap_chart(shap_df: pd.DataFrame) -> go.Figure:
    top = shap_df.head(10).sort_values("shap_value")
    fig = go.Figure(go.Bar(
        x=top["shap_value"], y=top["feature_label"], orientation="h",
        marker_color=["#f07060" if v > 0 else "#00d4a0" for v in top["shap_value"]],
        text=[f"{v:+.3f}" for v in top["shap_value"]],
        textposition="outside",
        textfont={"size": 11, "color": "#cdd8e8"},
    ))
    fig.add_vline(x=0, line_color="#4a6080", line_width=1)
    fig.update_layout(
        height=380, margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#1a2e44", zeroline=False, tickfont={"color": "#4a6080", "size": 10}),
        yaxis=dict(showgrid=False, tickfont={"color": "#cdd8e8", "size": 12}),
        font={"family": "Syne"}, showlegend=False,
    )
    return fig

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🩺 T2 Diabetes<em>Predictor</em></p>
    <p class="app-subtitle">Machine learning risk assessment · NHANES dataset (CDC) · HistGradientBoosting</p>
    <div style="margin-top: 12px;">
        <span class="badge">AUC 0.854</span><span class="badge">Recall 86%</span>
        <span class="badge">SHAP Explainability</span><span class="badge">28,452 samples</span>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Model not found at `models/diabetes_final_model.pkl`. Run the training pipeline first.", icon="⚠️")

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Patient Data")
    st.caption("Fill in the clinical measurements below.")

    st.markdown('<div class="sidebar-section">⚖️ Anthropometric</div>', unsafe_allow_html=True)
    age    = st.number_input("Age (years)",      min_value=18,  max_value=120,  value=52,  step=1)
    height = st.number_input("Height (cm)",      min_value=100, max_value=250,  value=170, step=1)
    weight = st.number_input("Weight (kg)",      min_value=20,  max_value=300,  value=88,  step=1)
    waist  = st.number_input("Waist circ. (cm)", min_value=40,  max_value=200,  value=102, step=1)
    bmi    = st.number_input("BMI (kg/m²)",      min_value=10.0, max_value=80.0,
                              value=round(weight / (height / 100) ** 2, 1), step=0.1,
                              help="Auto-calculated from weight & height, or enter manually.")

    st.markdown('<div class="sidebar-section">🩸 Blood Markers</div>', unsafe_allow_html=True)
    hdl   = st.number_input("HDL Cholesterol (mg/dL)",   min_value=1.0,  max_value=200.0,  value=50.0,  step=1.0)
    chol  = st.number_input("Total Cholesterol (mg/dL)", min_value=50.0, max_value=400.0,  value=180.0, step=1.0)
    trig  = st.number_input("Triglycerides (mg/dL)",     min_value=10.0, max_value=1000.0, value=100.0, step=1.0)
    creat = st.number_input("Creatinine (mg/dL)",        min_value=0.1,  max_value=20.0,   value=0.9,   step=0.01)

    st.markdown('<div class="sidebar-section">💓 Blood Pressure</div>', unsafe_allow_html=True)
    sbp = st.number_input("Systolic BP (mmHg)",  min_value=60, max_value=250, value=138, step=1)
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=88,  step=1)
    hyp = st.selectbox("Hypertension diagnosis", ["No", "Yes"])

    st.markdown('<div class="sidebar-section">🌿 Lifestyle & Socioeconomic</div>', unsafe_allow_html=True)
    sleep  = st.slider("Sleep hours / night",  3.0, 12.0, 6.0, step=0.5)
    income = st.slider("Income/Poverty ratio", 0.0, 5.0,  2.5, step=0.1,
                       help="0 = below poverty line · >5 = high income")
    eth = st.selectbox("Ethnicity", list(ETHNICITY_MAP.keys()), index=3)

    st.divider()
    predict_btn = st.button("🔍 Predict Risk", type="primary", width='stretch')

# ── Raw input — same columns as training data, before any pipeline step ───────
df_raw = pd.DataFrame([{
    "age_years":            age,
    "height_cm":            height,
    "weight_kg":            weight,
    "waist_cm":             waist,
    "bmi":                  bmi,
    "hdl_cholesterol":      hdl,
    "total_cholesterol":    chol,
    "triglycerides":        trig,
    "creatinine":           creat,
    "systolic_bp":          sbp,
    "diastolic_bp":         dbp,
    "hypertension":         1 if hyp == "Yes" else 0,
    "sleep_hours":          sleep,
    "income_poverty_ratio": income,
    "ethnicity":            ETHNICITY_MAP[eth],
}])

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if model is None:
        st.error("Model not loaded. Run the training pipeline first.")
    else:
        with st.spinner("Running prediction…"):
            preds, probs, threshold = predict(df_raw)   # ← app.py
            prob = float(probs[0])
            pred = int(preds[0])
            risk = ("Low" if prob < 0.25 else "Moderate" if prob < 0.50
                    else "High" if prob < 0.70 else "Very High")
            cfg = RISK_CONFIG[risk]

            # SHAP needs the preprocessed X → preprocess_input from app.py
            shap_df = None
            if explainer is not None:
                X = preprocess_input(df_raw)            # ← app.py
                sv = explainer(X)
                cols = list(X.columns)
                shap_df = pd.DataFrame({
                    "feature_key":   cols,
                    "feature_label": [FEATURE_LABELS.get(f, f) for f in cols],
                    "shap_value":    sv.values[0],
                    "feature_value": X.iloc[0].values,
                }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)

        # ── Risk banner ──
        st.markdown(f"""
        <div class="risk-banner {cfg['css']}">
            <span style="font-size:2.2rem">{cfg['emoji']}</span>
            <div>
                <div style="font-size:1.3rem; font-weight:800; color:{cfg['color']}">{risk} Risk</div>
                <div style="font-size:0.85rem; color:#cdd8e8; margin-top:4px">{cfg['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Gauge + metrics ──
        col1, col2 = st.columns([1.1, 1])
        with col1:
            st.subheader("Risk Probability")
            st.plotly_chart(make_gauge(prob, risk), width='stretch', config={"displayModeBar": False})
        with col2:
            st.subheader("Key Metrics")
            st.metric("Probability",          f"{prob*100:.1f}%")
            st.metric("Risk Level",           risk)
            st.metric("Prediction",           "⚠️ High Risk" if pred == 1 else "✅ Low Risk")
            st.metric("Decision Threshold",   f"{threshold*100:.0f}%")

        st.divider()

        # ── SHAP chart ──
        if shap_df is not None:
            st.subheader("🔎 Feature Contributions (SHAP)")
            st.caption("Factors pushing risk **up** (red) or **down** (green) for this specific patient.")
            st.plotly_chart(make_shap_chart(shap_df), width='stretch', config={"displayModeBar": False})
            with st.expander("View full SHAP table"):
                display_df = shap_df[["feature_label", "shap_value", "feature_value"]].copy()
                display_df.columns = ["Feature", "SHAP Value", "Patient Value"]
                display_df["SHAP Value"]    = display_df["SHAP Value"].map(lambda x: f"{x:+.4f}")
                display_df["Patient Value"] = display_df["Patient Value"].map(lambda x: f"{x:.3f}")
                st.dataframe(display_df, width='stretch', hide_index=True)
        else:
            st.info("SHAP explainer not found. Run `python generate_shap_explainer.py` to generate `models/shap_explainer.pkl`.")

        # ── Disclaimer ──
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <strong>Research use only.</strong> This tool is not a validated clinical diagnostic device and does not constitute medical advice.
        Predictions are probabilistic estimates based on population-level NHANES data. Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Empty state ──
    st.markdown("""
    <div style="text-align:center; padding: 80px 40px; color: #4a6080;">
        <div style="font-size: 4rem; margin-bottom: 16px;">🩺</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #cdd8e8; margin-bottom: 10px;">Ready for assessment</div>
        <div style="font-size: 0.9rem; max-width: 420px; margin: 0 auto; line-height: 1.7;">
            Fill in the patient data in the sidebar and click <strong style="color:#00d4a0">Predict Risk</strong>
            to get the diabetes risk probability with full SHAP explainability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "0.854", "AUC-ROC"),
        (c2, "86%",   "Recall (diabetes)"),
        (c3, "0.755", "Avg. Precision"),
        (c4, "28,452","Training samples"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="color:#00d4a0">{val}</div>
        </div>
        """, unsafe_allow_html=True)