import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# Import core logic
from main import load_models, get_model_info, run_prediction_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Cardiovascular Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    /* Hide default Streamlit elements for a clean dashboard look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 95%;
    }
    
    /* Global App Background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8);
        border-right: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 25px 20px;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Subtle glow line at the top of cards */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        opacity: 0.5;
    }

    .card-ecg::before { background: linear-gradient(90deg, transparent, #00f5ff, transparent); }
    .card-clin::before { background: linear-gradient(90deg, transparent, #ffaa00, transparent); }
    .card-final::before { background: linear-gradient(90deg, transparent, #ff4b4b, transparent); }

    .title-font {
        font-size: 16px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 10px;
    }
    
    .value-font {
        font-size: 42px;
        font-weight: 900;
        text-shadow: 0px 0px 20px currentColor;
    }
    
    /* Custom divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models_cached():
    return load_models()

ecg_model, heart_model, diag_model, models_loaded = load_models_cached()

# ---------------- SIDEBAR INPUTS ----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/833/833472.png", width=60) # Heart icon
    st.markdown("<h2 style='color: white; margin-top: 0;'>Patient Profile</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 14px;'>Enter patient vitals and upload ECG data.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload ECG (CSV format)", type=["csv"])
    
    if not models_loaded:
        st.warning("⚠️ Models not found. Running in UI Demo Mode with simulated data.")

    st.divider()
    
    age = st.slider("Age (Years)", 20, 100, 55)
    bp = st.slider("Systolic Blood Pressure", 80, 200, 140)
    chol = st.slider("Cholesterol Level", 100, 350, 240)
    hr = st.slider("Resting Heart Rate", 50, 180, 85)
    
    st.divider()
    predict_button = st.button("🚀 Analyze Risk Profile", use_container_width=True, type="primary")

# ---------------- HEADER ----------------
st.markdown("""
    <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;'>
        <div>
            <h1 style='margin: 0; color: white; font-weight: 800; letter-spacing: -1px;'>AI Cardiovascular Dashboard</h1>
            <p style='margin: 0; color: #00f5ff;'>Real-time biometric and time-series ECG analysis</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------- HELPER FUNCTIONS FOR VISUALIZATIONS ----------------
def create_empty_ecg_chart():
    """Create an empty ECG chart for placeholder state"""
    fig_empty = go.Figure()
    fig_empty.update_layout(
        template="plotly_dark", height=300, 
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Awaiting Data..."),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Amplitude")
    )
    return fig_empty

def create_gauge_chart(final_risk):
    """Create a gauge chart for risk visualization"""
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=final_risk * 100,
        number={'suffix': "%", 'font': {'size': 50, 'color': 'white'}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'size': 14}},
            'bar': {'color': "rgba(255,255,255,0.8)", 'thickness': 0.2},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': "rgba(0, 245, 255, 0.4)"},  # Cyan
                {'range': [30, 70], 'color': "rgba(255, 170, 0, 0.4)"}, # Orange
                {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.4)"} # Red
            ],
            'threshold': {
                'line': {'color': "white", 'width': 6},
                'thickness': 0.8,
                'value': final_risk * 100
            }
        }
    ))
    fig_gauge.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Inter"}
    )
    return fig_gauge

def create_bar_chart(ecg_prob, heart_prob, final_risk):
    """Create a bar chart comparing risk components"""
    fig_bar = go.Figure(data=[
        go.Bar(
            name='Risk Components',
            x=['ECG Signal', 'Clinical Vitals', 'Ensemble Final'],
            y=[ecg_prob, heart_prob, final_risk],
            marker_color=['#00f5ff', '#ffaa00', '#ff4b4b'],
            marker_line_color='white',
            marker_line_width=1.5,
            opacity=0.8,
            text=[f"{ecg_prob:.1%}", f"{heart_prob:.1%}", f"{final_risk:.1%}"],
            textposition='auto',
            textfont=dict(size=16, color='white', weight='bold'),
            hovertemplate="<b>%{x}</b><br>Risk Score: %{text}<extra></extra>"
        )
    ])
    fig_bar.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat=".0%"),
        xaxis=dict(showgrid=False),
        font={'color': "white", 'family': "Inter"}
    )
    return fig_bar

def create_ecg_chart(ecg_df):
    """Create an interactive ECG trace chart"""
    fig_ecg = go.Figure()
    
    # Add main line with glowing fill
    fig_ecg.add_trace(go.Scatter(
        y=ecg_df.values.flatten(),
        mode="lines",
        line=dict(color="#00f5ff", width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 245, 255, 0.1)',
        name="ECG Voltage",
        hovertemplate="Time Step: %{x}<br>Amplitude: %{y:.3f} mV<extra></extra>"
    ))
    
    fig_ecg.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        hovermode="x unified",
        xaxis=dict(
            title="Time / Data Points",
            showgrid=True,
            gridcolor='rgba(0, 245, 255, 0.15)',
            gridwidth=1,
            zeroline=False,
            rangeslider=dict(visible=True, thickness=0.08, bgcolor="rgba(255,255,255,0.05)")
        ),
        yaxis=dict(
            title="Amplitude (mV)",
            showgrid=True,
            gridcolor='rgba(0, 245, 255, 0.15)',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='rgba(0, 245, 255, 0.4)',
            zerolinewidth=2
        ),
        font={'family': "Inter"}
    )
    return fig_ecg

def display_metric_cards(ecg_prob, heart_prob, final_risk):
    """Display the three metric cards"""
    c1, c2, c3 = st.columns(3)
    
    c1.markdown(f"""
    <div class="metric-card card-ecg">
        <div class="title-font">Deep Learning ECG Risk</div>
        <div class="value-font" style="color:#00f5ff;">{ecg_prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="metric-card card-clin">
        <div class="title-font">XGBoost Clinical Risk</div>
        <div class="value-font" style="color:#ffaa00;">{heart_prob:.1%}</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="metric-card card-final">
        <div class="title-font">Ensemble Final Risk</div>
        <div class="value-font" style="color:#ff4b4b;">{final_risk:.1%}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

def display_risk_alert(final_risk, diagnosis, risk_level, risk_icon, risk_message):
    """Display the risk alert based on final risk"""
    alert_col1, alert_col2 = st.columns([3, 1])
    with alert_col1:
        if risk_level == "LOW":
            st.success(f"🟢 **LOW RISK PROFILE** — {risk_message} (Code: {diagnosis})")
        elif risk_level == "MEDIUM":
            st.warning(f"🟡 **MEDIUM RISK PROFILE** — {risk_message} (Code: {diagnosis})")
        else:
            st.error(f"🔴 **CRITICAL RISK PROFILE** — {risk_message} (Code: {diagnosis})")

# ---------------- MAIN DASHBOARD LOGIC ----------------
if not predict_button:
    # Empty State Dashboard
    st.info("👈 Please input the patient's data in the sidebar and click **Analyze Risk Profile** to generate the interactive report.")
    
    # Show a placeholder empty ECG
    fig_empty = create_empty_ecg_chart()
    st.plotly_chart(fig_empty, use_container_width=True)

else:
    with st.spinner('Analyzing biometric and time-series data...'):
        time.sleep(1)  # Artificial delay for UI polish
        
        # Run prediction pipeline
        results = run_prediction_pipeline(
            uploaded_file, age, bp, chol, hr, 
            models_loaded, ecg_model, heart_model, diag_model
        )

    # Display metric cards
    display_metric_cards(results['ecg_prob'], results['heart_prob'], results['final_risk'])

    # Display risk alert
    display_risk_alert(
        results['final_risk'], 
        results['diagnosis'],
        results['risk_level'],
        results['risk_icon'],
        results['risk_message']
    )

    # ---------------- INTERACTIVE CHARTS ----------------
    st.markdown("<h3 style='color: white; margin-top: 20px; font-weight: 600;'>Diagnostic Analytics</h3>", unsafe_allow_html=True)
    
    col_gauge, col_bar = st.columns(2)
    
    # 1. Gauge Chart
    with col_gauge:
        fig_gauge = create_gauge_chart(results['final_risk'])
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    # 2. Bar Chart
    with col_bar:
        fig_bar = create_bar_chart(results['ecg_prob'], results['heart_prob'], results['final_risk'])
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

    # 3. ECG Area Chart
    st.markdown("<h3 style='color: white; margin-top: 10px; font-weight: 600;'>High-Resolution ECG Trace</h3>", unsafe_allow_html=True)
    
    fig_ecg = create_ecg_chart(results['ecg_df'])
    st.plotly_chart(fig_ecg, use_container_width=True)