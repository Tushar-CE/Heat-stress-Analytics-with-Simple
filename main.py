import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import hashlib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heat Stress Predictor - Simple View",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal CSS - clean white background, simple text
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 3px solid #4a90d9;
        margin-bottom: 1.5rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .result-row {
        display: flex;
        justify-content: space-between;
        padding: 0.6rem 0.8rem;
        background-color: #f8f9fa;
        border-radius: 6px;
        margin: 0.25rem 0;
        border-left: 4px solid #4a90d9;
    }
    .result-label {
        font-weight: 500;
        color: #333;
    }
    .result-value {
        font-weight: 600;
        color: #1a1a2e;
        font-size: 1.05rem;
    }
    .result-value.highlight {
        color: #d9534f;
        font-size: 1.2rem;
    }
    .result-value.good {
        color: #5cb85c;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 0.8rem 1.2rem;
        margin: 0.8rem 0;
    }
    .warning-box.good {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .warning-box.danger {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .level-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .level-badge.very-hot { background: #8B0000; color: white; }
    .level-badge.hot { background: #DC3545; color: white; }
    .level-badge.warm { background: #FF8C00; color: white; }
    .level-badge.slightly-warm { background: #FFD700; color: #333; }
    .level-badge.comfortable { background: #28A745; color: white; }
    .input-label {
        font-weight: 500;
        color: #333;
        margin-bottom: 0.2rem;
    }
    .divider {
        border: none;
        border-top: 2px dashed #ddd;
        margin: 1.5rem 0;
    }
    .footer-text {
        text-align: center;
        color: #999;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-item {
        background: #f8f9fa;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        text-align: center;
    }
    .metric-item .label {
        font-size: 0.8rem;
        color: #666;
    }
    .metric-item .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🌡️ Heat Stress Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Simple text-based output — Enter conditions to see results</div>', unsafe_allow_html=True)

# File paths
CSV_PATH = "EXBD.csv"

# Work Type Data
work_data = {
    "Rest (R)": {"M": 115, "PET_AL": 35, "description": "Sitting, light activities", "base_factor": 0.15},
    "Light (LW)": {"M": 180, "PET_AL": 35.5, "description": "Standing, light hand work", "base_factor": 0.20},
    "Moderate (MW)": {"M": 300, "PET_AL": 32, "description": "Walking, moderate lifting", "base_factor": 0.25},
    "Heavy (HW)": {"M": 415, "PET_AL": 31, "description": "Heavy lifting, shoveling", "base_factor": 0.28},
    "Very Heavy (VHW)": {"M": 520, "PET_AL": 30, "description": "Very intense labor", "base_factor": 0.30}
}

activity_to_work = {
    2.1: "Light (LW)", 2.2: "Light (LW)", 2.6: "Moderate (MW)",
    3.2: "Heavy (HW)", 3.8: "Heavy (HW)", 4.0: "Very Heavy (VHW)"
}

# Data loading functions (same as original)
@st.cache_data(ttl=3600, show_spinner=False)
def load_training_data():
    try:
        if not os.path.exists(CSV_PATH):
            return None, None
        usecols = ['T', 'RH', 'WS', 'PET', 'PMV', 'PPD', 'SET', 'RWS', 'CE', 'Height', 'PETH', 'PMVH']
        df = pd.read_csv(CSV_PATH, encoding='utf-8', usecols=usecols)
        hs_df = pd.DataFrame({
            'T(0C)': pd.to_numeric(df['T'], errors='coerce'),
            'RH(%)': pd.to_numeric(df['RH'], errors='coerce'),
            'WS(m/s)': pd.to_numeric(df['WS'], errors='coerce'),
            'PET(0C)': pd.to_numeric(df['PET'], errors='coerce'),
            'PMV': pd.to_numeric(df['PMV'], errors='coerce'),
            'PPD(%)': pd.to_numeric(df['PPD'], errors='coerce'),
            'SET (0C)': pd.to_numeric(df['SET'], errors='coerce'),
            'RWS(m/s)': pd.to_numeric(df['RWS'], errors='coerce'),
            'CE(0C)': pd.to_numeric(df['CE'], errors='coerce')
        }).dropna()
        bh_df = pd.DataFrame({
            'Height(m)': pd.to_numeric(df['Height'], errors='coerce'),
            'PET(0C)': pd.to_numeric(df['PETH'], errors='coerce'),
            'PMV': pd.to_numeric(df['PMVH'], errors='coerce')
        }).dropna()
        if len(bh_df) > 0:
            bh_df = bh_df[bh_df['Height(m)'] > 0].sort_values('Height(m)')
        return hs_df, bh_df
    except Exception:
        return None, None

@st.cache_resource(ttl=7200, show_spinner=False)
def train_neural_network(X_scaled, y_dict):
    models = {}
    scores = {}
    for target, y in y_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        nn = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        scores[target] = r2_score(y_test, y_pred)
        models[target] = nn
    return models, scores

# Load data
hs_df, bh_df = load_training_data()

# Generate sample data if needed
if hs_df is None or len(hs_df) == 0:
    np.random.seed(42)
    n_samples = 300
    T_range = np.random.uniform(25, 40, n_samples)
    RH_range = np.random.uniform(40, 80, n_samples)
    WS_range = np.random.uniform(0.5, 5, n_samples)
    PET_calc = T_range + 5 - 0.8 * WS_range + 0.015 * (RH_range - 40) + np.random.normal(0, 1, n_samples)
    PET_calc = np.clip(PET_calc, 25, 48)
    PMV_calc = 1.5 + (T_range - 25) * 0.12 - 0.15 * WS_range + 0.02 * (RH_range - 40) + np.random.normal(0, 0.2, n_samples)
    PMV_calc = np.clip(PMV_calc, 1, 4)
    hs_df = pd.DataFrame({
        'T(0C)': T_range, 'RH(%)': RH_range, 'WS(m/s)': WS_range,
        'PET(0C)': PET_calc, 'PMV': PMV_calc,
        'PPD(%)': 5 + 85 * (1 - np.exp(-0.5 * (PMV_calc - 1))),
        'SET (0C)': PET_calc - 1 + np.random.normal(0, 0.5, n_samples),
        'RWS(m/s)': WS_range * 0.8 + np.random.normal(0, 0.2, n_samples),
        'CE(0C)': 2 + 0.5 * WS_range + np.random.normal(0, 0.3, n_samples)
    })

# Prepare models
features = ['T(0C)', 'RH(%)', 'WS(m/s)']
targets = ['PET(0C)', 'PMV', 'PPD(%)', 'SET (0C)', 'RWS(m/s)', 'CE(0C)']
X = hs_df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_dict = {target: hs_df[target].values for target in targets if target in hs_df.columns}
models, model_scores = train_neural_network(X_scaled, y_dict)

# ========== SIDEBAR - Simple Inputs ==========
with st.sidebar:
    st.markdown("### ⚙️ Enter Conditions")
    st.markdown("---")
    
    T = st.number_input("🌡️ Temperature (°C)", 20.0, 50.0, 34.0, 0.1)
    RH = st.number_input("💧 Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("💨 Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    
    st.markdown("---")
    clo = st.select_slider(
        "👕 Clothing (clo)",
        options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00],
        value=0.57
    )
    met = st.select_slider(
        "💪 Activity (met)",
        options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0],
        value=3.2
    )
    height = st.slider("🏗️ Working Height (m)", 0, 100, 0, 1)
    
    st.markdown("---")
    baseline_productivity = st.number_input("📊 Baseline Productivity (units/hr)", min_value=1.0, value=100.0, step=5.0)
    
    if st.button("🔄 Predict", use_container_width=True):
        st.rerun()

# ========== MAIN CONTENT ==========

# Make prediction
input_data = np.array([[T, RH, WS]])
input_scaled = scaler.transform(input_data)

predictions = {}
for target in targets:
    if target in models:
        predictions[target] = models[target].predict(input_scaled)[0]
    else:
        # Simple physics fallback
        if target == 'PET(0C)':
            predictions[target] = T + 5 + 0.015*(RH-40) - 0.8*WS
        elif target == 'PMV':
            predictions[target] = 1.5 + (T-25)*0.12 - 0.15*WS + 0.02*(RH-40)
        elif target == 'PPD(%)':
            predictions[target] = 50
        elif target == 'SET (0C)':
            predictions[target] = T + 3
        elif target == 'RWS(m/s)':
            predictions[target] = WS * 0.8
        elif target == 'CE(0C)':
            predictions[target] = 2 + 0.5*WS
        else:
            predictions[target] = 0

# Apply adjustments
predictions['PET(0C)'] = np.clip(predictions['PET(0C)'] + clo * 0.5 + (met - 2.0) * 0.3, 20, 50)
predictions['PMV'] = np.clip(predictions['PMV'] + clo * 0.3 + (met - 2.0) * 0.2, 0, 3.5)
predictions['PPD(%)'] = np.clip(predictions['PPD(%)'] + clo * 2 + (met - 2.0) * 1.5, 5, 90)

pet = predictions["PET(0C)"]
pmv = predictions["PMV"]

# Determine thermal level
if pet > 41.0:
    pet_class, pet_desc, pet_icon, badge_class = "VERY HOT", "Extreme heat stress — Unsafe for physical work", "🔥", "very-hot"
    work_recommendation = "🚫 Suspend all outdoor work"
    productivity_impact_range = "25-30%"
elif pet > 35.0:
    pet_class, pet_desc, pet_icon, badge_class = "HOT", "High heat stress — Significant strain", "🌡️", "hot"
    work_recommendation = "⚠️ Limit work to 45-min sessions"
    productivity_impact_range = "15-25%"
elif pet > 29.0:
    pet_class, pet_desc, pet_icon, badge_class = "WARM", "Moderate heat stress — Reduced work capacity", "☀️", "warm"
    work_recommendation = "✓ Normal work with breaks"
    productivity_impact_range = "5-15%"
elif pet > 23.0:
    pet_class, pet_desc, pet_icon, badge_class = "SLIGHTLY WARM", "Slight heat stress — Comfortable", "⛅", "slightly-warm"
    work_recommendation = "✓ Normal work"
    productivity_impact_range = "0-5%"
else:
    pet_class, pet_desc, pet_icon, badge_class = "COMFORTABLE", "No heat stress — Optimal conditions", "✅", "comfortable"
    work_recommendation = "✓ Normal work"
    productivity_impact_range = "0%"

# PMV level
if pmv >= 3.0:
    pmv_class = "EXTREME DISCOMFORT"
elif pmv >= 2.5:
    pmv_class = "VERY DISCOMFORT"
elif pmv >= 2.0:
    pmv_class = "MODERATE DISCOMFORT"
elif pmv >= 1.5:
    pmv_class = "DISCOMFORT"
elif pmv >= 1.0:
    pmv_class = "SLIGHT DISCOMFORT"
else:
    pmv_class = "COMFORTABLE"

# ========== OUTPUT SECTION ==========

# 1. Overall Status
st.markdown(f"""
<div style="background: #f0f4f8; padding: 1.2rem; border-radius: 8px; margin-bottom: 1rem; text-align: center;">
    <div style="font-size: 1.5rem; font-weight: 700; color: #1a1a2e;">
        {pet_icon} Overall Status: <span class="level-badge {badge_class}">{pet_class}</span>
    </div>
    <div style="color: #444; margin-top: 0.3rem;">{pet_desc}</div>
    <div style="margin-top: 0.5rem; font-weight: 500;">{work_recommendation}</div>
</div>
""", unsafe_allow_html=True)

# 2. Key Thermal Metrics
st.markdown('<div class="section-title">📊 Key Thermal Metrics</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="result-row">
        <span class="result-label">🌡️ Physiological Equivalent Temp (PET)</span>
        <span class="result-value highlight">{pet:.1f} °C</span>
    </div>
    <div class="result-row">
        <span class="result-label">📊 Predicted Mean Vote (PMV)</span>
        <span class="result-value">{pmv:.2f}</span>
    </div>
    <div class="result-row">
        <span class="result-label">📋 PMV Interpretation</span>
        <span class="result-value" style="font-size:0.95rem;">{pmv_class}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="result-row">
        <span class="result-label">😓 Predicted % Dissatisfied (PPD)</span>
        <span class="result-value">{predictions['PPD(%)']:.1f}%</span>
    </div>
    <div class="result-row">
        <span class="result-label">🌬️ Relative Wind Speed (RWS)</span>
        <span class="result-value">{predictions['RWS(m/s)']:.1f} m/s</span>
    </div>
    <div class="result-row">
        <span class="result-label">❄️ Cooling Effect (CE)</span>
        <span class="result-value">{predictions['CE(0C)']:.1f} °C</span>
    </div>
    """, unsafe_allow_html=True)

# 3. Productivity Impact
st.markdown('<div class="section-title">📉 Productivity Impact</div>', unsafe_allow_html=True)

current_work_key = activity_to_work.get(met, "Heavy (HW)")
work = work_data[current_work_key]

def calc_productivity_loss(pet_value, work_type_key, baseline):
    work_data_local = work_data[work_type_key]
    PET_AL = work_data_local["PET_AL"]
    base_factor = work_data_local["base_factor"]
    if pet_value <= PET_AL:
        return 0
    else:
        delta_pet = pet_value - PET_AL
        PL = 30 * (1 - np.exp(-base_factor * delta_pet))
        return min(PL, 30)

productivity_loss = calc_productivity_loss(pet, current_work_key, baseline_productivity)

# Productivity loss color
if productivity_loss == 0:
    loss_color = "good"
    loss_emoji = "✅"
elif productivity_loss < 10:
    loss_color = "good"
    loss_emoji = "ℹ️"
elif productivity_loss < 20:
    loss_color = "warning-box"
    loss_emoji = "⚠️"
else:
    loss_color = "danger"
    loss_emoji = "🔴"

st.markdown(f"""
<div class="warning-box {loss_color}">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong>{loss_emoji} Current Work:</strong> {current_work_key}
            <span style="color: #666; font-size: 0.9rem;">({work['description']})</span>
        </div>
        <div style="font-size: 1.3rem; font-weight: 700;">
            {productivity_loss:.1f}% loss
        </div>
    </div>
    <div style="margin-top: 0.4rem; font-size: 0.95rem; color: #444;">
        PET Alert Level: {work['PET_AL']}°C • Estimated loss range: {productivity_impact_range}
    </div>
</div>
""", unsafe_allow_html=True)

# 4. All Work Types Loss Comparison (simple table)
st.markdown('<div class="section-title">📊 Loss by Work Type</div>', unsafe_allow_html=True)

loss_data = []
for wt in work_data.keys():
    loss = calc_productivity_loss(pet, wt, baseline_productivity)
    loss_data.append((wt, loss))

# Find max loss for highlighting
max_loss = max([l for _, l in loss_data])

st.markdown('<div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Productivity loss at current PET = {:.1f}°C</div>'.format(pet), unsafe_allow_html=True)

for wt, loss in loss_data:
    is_current = (wt == current_work_key)
    bar_color = "#dc3545" if loss >= 20 else "#ff8c00" if loss >= 10 else "#28a745"
    bar_width = max(2, (loss / 30) * 100) if loss > 0 else 2
    marker = "▶️" if is_current else "  "
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 0.2rem 0; padding: 0.2rem 0;">
        <div style="width: 120px; font-size: 0.9rem; font-weight: {'600' if is_current else '400'};">
            {marker} {wt}
        </div>
        <div style="flex: 1; background: #eee; border-radius: 4px; height: 22px; position: relative;">
            <div style="width: {bar_width}%; background: {bar_color}; height: 100%; border-radius: 4px; transition: width 0.3s;"></div>
            <div style="position: absolute; right: 6px; top: 2px; font-size: 0.75rem; font-weight: 600; color: {'white' if loss > 15 else '#333'};">
                {loss:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 5. Additional Environmental Metrics
st.markdown('<div class="section-title">🌍 Environmental Conditions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">Temperature</div>
        <div class="value">{T:.1f} °C</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">Humidity</div>
        <div class="value">{RH:.0f} %</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">Wind Speed</div>
        <div class="value">{WS:.1f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; margin-top: 0.3rem;">
    <div class="metric-item"><div class="label">Clothing</div><div class="value">{clo:.2f} clo</div></div>
    <div class="metric-item"><div class="label">Activity</div><div class="value">{met:.1f} met</div></div>
    <div class="metric-item"><div class="label">Working Height</div><div class="value">{height} m</div></div>
</div>
""", unsafe_allow_html=True)

# 6. Simple guidance (text only)
st.markdown('<div class="section-title">📋 Quick Guidance</div>', unsafe_allow_html=True)

if pet > 41:
    guidance = """
    🔴 **EXTREME HEAT — STOP WORK**
    - All outdoor work should be suspended immediately
    - Move to air-conditioned areas
    - Drink water every 15 minutes
    - Watch for signs of heat exhaustion: headache, dizziness, confusion
    """
elif pet > 35:
    guidance = """
    🟠 **HIGH HEAT — TAKE PRECAUTIONS**
    - Limit work to 45-minute sessions with 15-minute breaks
    - Wear light, breathable clothing
    - Drink water every 30 minutes
    - Use shaded rest areas
    """
elif pet > 29:
    guidance = """
    🟡 **MODERATE HEAT — STAY ALERT**
    - Take regular breaks in shade
    - Maintain hydration
    - Normal work schedule with monitoring
    """
else:
    guidance = """
    🟢 **COMFORTABLE — NORMAL WORK**
    - Regular hydration
    - Standard work schedule
    - No heat-related restrictions
    """

st.markdown(f"""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #4a90d9; white-space: pre-line; font-size: 0.95rem;">
{guidance}
</div>
""", unsafe_allow_html=True)

# 7. Simple explanation of what PET means
with st.expander("ℹ️ What do these numbers mean?"):
    st.markdown("""
    **PET (Physiological Equivalent Temperature)** - The temperature at which the body would feel the same thermal stress in a reference environment. Higher PET = more heat stress.
    
    - **>41°C**: Very Hot — Dangerous for any physical work
    - **35-41°C**: Hot — High strain, limit work duration
    - **29-35°C**: Warm — Moderate strain, take breaks
    - **23-29°C**: Slightly Warm — Comfortable for most work
    - **<23°C**: Comfortable — No heat stress
    
    **PMV (Predicted Mean Vote)** - How warm/cold people feel on average. Range: 0 (neutral) to 3.5 (very hot).
    
    **Productivity Loss** - Estimated reduction in work output due to heat stress. Based on research from construction sites.
    """)

# Footer
st.markdown('<div class="footer-text">Heat Stress Predictor — Simple Interface for Basic Analysis</div>', unsafe_allow_html=True)