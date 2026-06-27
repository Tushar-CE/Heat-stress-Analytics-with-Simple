import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="Construction Heat Stress Estimator - Simple View",
    page_icon="🏗️",
    layout="centered"
)

# ============ CUSTOM CSS - DARK THEME ============
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(145deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        padding: 1.5rem;
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 0.95rem;
        margin-top: -1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
        padding: 0.6rem 1rem;
        margin: 1.2rem 0 0.8rem 0;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(5px);
        border-radius: 12px;
        border-left: 5px solid #4ECDC4;
    }
    .result-card {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(8px);
        padding: 0.8rem 1rem;
        border-radius: 12px;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 0.3rem 0;
        transition: 0.3s;
    }
    .result-card:hover {
        background: rgba(255,255,255,0.12);
        transform: translateY(-2px);
    }
    .result-label {
        font-weight: 500;
        color: rgba(255,255,255,0.9);
        font-size: 0.85rem;
    }
    .result-value {
        font-weight: 700;
        color: #ffffff;
        font-size: 1.15rem;
    }
    .result-value.high {
        color: #FF6B6B;
    }
    .result-value.moderate {
        color: #FFD93D;
    }
    .result-value.low {
        color: #4ECDC4;
    }
    .status-box {
        padding: 1.2rem;
        border-radius: 16px;
        text-align: center;
        color: #ffffff;
        margin: 0.8rem 0;
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .status-box.critical {
        background: linear-gradient(135deg, #8B0000, #FF0000);
    }
    .status-box.high {
        background: linear-gradient(135deg, #FF4500, #FF8C00);
    }
    .status-box.moderate {
        background: linear-gradient(135deg, #FFA500, #FFD700);
        color: #333;
    }
    .status-box.low {
        background: linear-gradient(135deg, #006400, #228B22);
    }
    .status-box.comfortable {
        background: linear-gradient(135deg, #1a472a, #2d7d46);
    }
    .warning-box {
        background: rgba(255,255,255,0.07);
        backdrop-filter: blur(8px);
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 0.8rem 0;
        color: #ffffff;
    }
    .warning-box.critical {
        border-left: 5px solid #FF0000;
        background: rgba(255,0,0,0.1);
    }
    .warning-box.high {
        border-left: 5px solid #FF8C00;
        background: rgba(255,140,0,0.1);
    }
    .warning-box.moderate {
        border-left: 5px solid #FFD700;
        background: rgba(255,215,0,0.1);
    }
    .warning-box.low {
        border-left: 5px solid #4ECDC4;
        background: rgba(78,205,196,0.1);
    }
    .height-profile {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 0.5rem 0;
    }
    .height-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #ffffff;
    }
    .height-row:last-child {
        border-bottom: none;
    }
    .height-row.highlight {
        background: rgba(78,205,196,0.15);
        border-radius: 4px;
        padding: 0.4rem 0.5rem;
    }
    .bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 4px;
        height: 22px;
        position: relative;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s;
    }
    .bar-label {
        position: absolute;
        right: 8px;
        top: 2px;
        font-size: 0.7rem;
        font-weight: 600;
        color: #ffffff;
    }
    .metric-item {
        background: rgba(255,255,255,0.06);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-item .label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-item .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.2rem;
    }
    .guidance-box {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(8px);
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #4ECDC4;
        color: #ffffff;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .guidance-box.critical {
        border-left-color: #FF0000;
    }
    .guidance-box.high {
        border-left-color: #FF8C00;
    }
    .guidance-box.moderate {
        border-left-color: #FFD700;
    }
    .divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 1.5rem 0;
    }
    .footer-text {
        text-align: center;
        color: rgba(255,255,255,0.5);
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(195deg, #1a2f3f 0%, #1e3a4a 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #ffffff;
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.05);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stSlider > div > div {
        color: #ffffff;
    }
    .stSelectSlider > div {
        color: #ffffff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1a3a47, #2a4a5a, #3a5a6a);
        transform: scale(1.02);
    }
    .stExpander {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .stExpander > div {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============ DATA LOADING FUNCTIONS ============
def create_sample_data():
    """Generate synthetic construction site thermal data"""
    np.random.seed(42)
    n_samples = 500
    
    T = np.random.uniform(20, 45, n_samples)
    RH = np.random.uniform(30, 90, n_samples)
    WS = np.random.uniform(0.1, 8, n_samples)
    
    PET = T + 5 - 0.8 * WS + 0.015 * (RH - 40) + np.random.normal(0, 1, n_samples)
    PET = np.clip(PET, 20, 50)
    
    PMV = 1.5 + (T - 25) * 0.12 - 0.15 * WS + 0.02 * (RH - 40) + np.random.normal(0, 0.2, n_samples)
    PMV = np.clip(PMV, 0, 4)
    
    df = pd.DataFrame({
        'T': T, 'RH': RH, 'WS': WS, 'PET': PET, 'PMV': PMV,
        'PPD': 5 + 85 * (1 - np.exp(-0.5 * (PMV - 1))),
        'SET': PET - 1 + np.random.normal(0, 0.5, n_samples),
        'RWS': WS * 0.8 + np.random.normal(0, 0.2, n_samples),
        'CE': 2 + 0.5 * WS + np.random.normal(0, 0.3, n_samples),
        'Height': np.random.uniform(0, 100, n_samples),
        'PETH': PET - np.random.uniform(0, 6, n_samples),
        'PMVH': PMV - np.random.uniform(0, 1, n_samples)
    })
    return df

CSV_PATH = "EXBD.csv"

if not os.path.exists(CSV_PATH):
    df = create_sample_data()
    df.to_csv(CSV_PATH, index=False)
else:
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except:
        df = create_sample_data()

# Prepare dataframes
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

# ============ TRAIN MODEL ============
features = ['T(0C)', 'RH(%)', 'WS(m/s)']
targets = ['PET(0C)', 'PMV', 'PPD(%)', 'SET (0C)', 'RWS(m/s)', 'CE(0C)']

X = hs_df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_dict = {}
for target in targets:
    if target in hs_df.columns:
        y_dict[target] = hs_df[target].values

models = {}
for target, y in y_dict.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    nn = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10
    )
    nn.fit(X_train, y_train)
    models[target] = nn

# ============ CONSTRUCTION WORK DATA ============
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

# ============ HELPER FUNCTIONS ============
def calc_productivity_loss(pet_value, work_type_key, baseline):
    """Calculate construction worker productivity decrement"""
    work = work_data[work_type_key]
    PET_AL = work["PET_AL"]
    base_factor = work["base_factor"]
    
    if pet_value <= PET_AL:
        return 0
    else:
        delta_pet = pet_value - PET_AL
        PL = 30 * (1 - np.exp(-base_factor * delta_pet))
        return min(PL, 30)

def get_thermal_risk_level(pet):
    """Determine construction heat stress risk category"""
    if pet > 41.0:
        return "CRITICAL", "Immediate work stoppage required", "🔴", "critical"
    elif pet > 35.0:
        return "HIGH", "Severe heat strain - reduced work capacity", "🟠", "high"
    elif pet > 29.0:
        return "MODERATE", "Elevated heat stress - monitoring required", "🟡", "moderate"
    elif pet > 23.0:
        return "LOW", "Mild heat stress - standard precautions", "🟢", "low"
    else:
        return "COMFORTABLE", "Optimal working conditions", "✅", "comfortable"

def get_pmv_interpretation(pmv):
    """Construction worker thermal comfort interpretation"""
    if pmv >= 3.0:
        return "SEVERE DISCOMFORT - High heat strain"
    elif pmv >= 2.5:
        return "VERY UNCOMFORTABLE - Significant heat stress"
    elif pmv >= 2.0:
        return "MODERATE DISCOMFORT - Noticeable heat"
    elif pmv >= 1.5:
        return "MILD DISCOMFORT - Warm conditions"
    elif pmv >= 1.0:
        return "SLIGHT WARMTH - Acceptable"
    else:
        return "COMFORTABLE - Neutral thermal sensation"

def get_height_profile(ground_pet, ground_pmv, height_m, bh_df):
    """Calculate vertical temperature gradient for construction sites"""
    if len(bh_df) == 0:
        return None
    
    heights_original = bh_df['Height(m)'].values
    pet_pattern = bh_df['PET(0C)'].values.copy()
    pmv_pattern = bh_df['PMV'].values.copy() if 'PMV' in bh_df.columns else pet_pattern * 0.08
    
    # Ensure monotonic decrease (lapse rate)
    for i in range(1, len(pet_pattern)):
        if pet_pattern[i] > pet_pattern[i-1]:
            pet_pattern[i] = max(pet_pattern[i-1] - 0.1, 0)
    
    for i in range(1, len(pmv_pattern)):
        if pmv_pattern[i] > pmv_pattern[i-1]:
            pmv_pattern[i] = max(pmv_pattern[i-1] - 0.01, -3)
    
    if pet_pattern[0] != 0:
        pet_relative = pet_pattern / pet_pattern[0]
    else:
        pet_relative = pet_pattern
    
    if pmv_pattern[0] != 0:
        pmv_relative = pmv_pattern / pmv_pattern[0]
    else:
        pmv_relative = pmv_pattern
    
    pet_profile = ground_pet * pet_relative
    pmv_profile = ground_pmv * pmv_relative
    
    heights_smooth = np.linspace(0, 100, 100)
    pet_smooth = np.interp(heights_smooth, heights_original, pet_profile)
    pmv_smooth = np.interp(heights_smooth, heights_original, pmv_profile)
    
    if 0 <= height_m <= 100:
        pet_at_height = np.interp(height_m, heights_smooth, pet_smooth)
        pmv_at_height = np.interp(height_m, heights_smooth, pmv_smooth)
    else:
        pet_at_height = ground_pet
        pmv_at_height = ground_pmv
    
    pet_at_100 = np.interp(100, heights_smooth, pet_smooth)
    pmv_at_100 = np.interp(100, heights_smooth, pmv_smooth)
    
    return {
        'pet_at_height': pet_at_height,
        'pmv_at_height': pmv_at_height,
        'pet_at_100': pet_at_100,
        'pmv_at_100': pmv_at_100,
        'pet_profile': pet_smooth,
        'pmv_profile': pmv_smooth,
        'heights': heights_smooth,
        'reduction': ground_pet - pet_at_height,
        'lapse_rate': (ground_pet - pet_at_100) / 100 if height_m > 0 else 0
    }

# ============ MAIN UI ============
st.markdown('<div class="main-title">🏗️ Construction Heat Stress Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Site-Specific Thermal Exposure & Productivity Assessment</div>', unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### ⚙️ Site Conditions")
    st.markdown("---")
    
    st.markdown("#### 🌤️ Environmental Parameters")
    T = st.number_input("Air Temperature (°C)", 20.0, 50.0, 34.0, 0.1)
    RH = st.number_input("Relative Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    
    st.markdown("---")
    st.markdown("#### 👷 Worker Factors")
    clo = st.select_slider(
        "Clothing Insulation (clo)",
        options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00],
        value=0.57
    )
    met = st.select_slider(
        "Metabolic Rate (met)",
        options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0],
        value=3.2
    )
    
    st.markdown("---")
    st.markdown("#### 🏗️ Work Location")
    height = st.slider("Working Height Above Ground (m)", 0, 100, 0, 1)
    st.caption("PET decreases ~0.05-0.08°C per meter elevation")
    
    st.markdown("---")
    st.markdown("#### 📊 Productivity Baseline")
    baseline_productivity = st.number_input("Baseline Output (units/hr)", min_value=1.0, value=100.0, step=5.0)
    
    st.markdown("---")
    if st.button("🔄 Assess Heat Stress", use_container_width=True):
        st.rerun()

# ============ PREDICTIONS ============
input_data = np.array([[T, RH, WS]])
input_scaled = scaler.transform(input_data)

predictions = {}
for target in targets:
    if target in models:
        predictions[target] = models[target].predict(input_scaled)[0]
    else:
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

ground_pet = predictions["PET(0C)"]
ground_pmv = predictions["PMV"]

# ============ HEIGHT PROFILE ============
height_profile = get_height_profile(ground_pet, ground_pmv, height, bh_df)

# ============ RISK ASSESSMENT ============
risk_level, risk_desc, risk_icon, risk_class = get_thermal_risk_level(ground_pet)
pmv_interpretation = get_pmv_interpretation(ground_pmv)

# ============ 1. THERMAL RISK STATUS ============
st.markdown(f"""
<div class="status-box {risk_class}">
    <div style="font-size: 2rem; font-weight: 700;">
        {risk_icon} {risk_level} RISK
    </div>
    <div style="font-size: 1.1rem; margin-top: 0.3rem;">
        {risk_desc}
    </div>
    <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
        PET = {ground_pet:.1f}°C | PMV = {ground_pmv:.2f} — {pmv_interpretation}
    </div>
</div>
""", unsafe_allow_html=True)

# ============ 2. SITE CONDITIONS ============
st.markdown('<div class="section-title">📋 Site Conditions Summary</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">🌡️ Ambient Temperature</div>
        <div class="value">{T:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">💧 Humidity</div>
        <div class="value">{RH:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">💨 Wind Speed</div>
        <div class="value">{WS:.1f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">👕 Clothing</div>
        <div class="value">{clo:.2f} clo</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">💪 Activity</div>
        <div class="value">{met:.1f} met</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">🏗️ Working Height</div>
        <div class="value">{height} m</div>
    </div>
    """, unsafe_allow_html=True)

# ============ 3. THERMAL EXPOSURE METRICS ============
st.markdown('<div class="section-title">🌡️ Thermal Exposure Indicators</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pet_color = "high" if ground_pet > 35 else "moderate" if ground_pet > 29 else "low"
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">🌡️ Physiological Equivalent Temperature (PET)</div>
        <div class="result-value {pet_color}">{ground_pet:.1f} °C</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:0.2rem;">
            Effective thermal sensation temperature
        </div>
    </div>
    <div class="result-card">
        <div class="result-label">📊 Predicted Mean Vote (PMV)</div>
        <div class="result-value">{ground_pmv:.2f}</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:0.2rem;">
            {pmv_interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">😓 Predicted Percentage Dissatisfied (PPD)</div>
        <div class="result-value">{predictions['PPD(%)']:.1f}%</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:0.2rem;">
            Expected workforce thermal dissatisfaction
        </div>
    </div>
    <div class="result-card">
        <div class="result-label">🌬️ Relative Wind Speed (RWS)</div>
        <div class="result-value">{predictions['RWS(m/s)']:.1f} m/s</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:0.2rem;">
            Effective wind exposure at worker level
        </div>
    </div>
    <div class="result-card">
        <div class="result-label">❄️ Cooling Effect (CE)</div>
        <div class="result-value">{predictions['CE(0C)']:.1f} °C</div>
        <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:0.2rem;">
            Evaporative cooling potential
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ 4. VERTICAL THERMAL GRADIENT ============
if height_profile:
    st.markdown('<div class="section-title">🏗️ Vertical Thermal Gradient Analysis</div>', unsafe_allow_html=True)
    
    pet_at_height = height_profile['pet_at_height']
    pmv_at_height = height_profile['pmv_at_height']
    pet_at_100 = height_profile['pet_at_100']
    reduction = height_profile['reduction']
    lapse_rate = height_profile['lapse_rate']
    
    # Thermal profile table
    st.markdown(f"""
    <div class="height-profile">
        <div class="height-row" style="font-weight:600; border-bottom:2px solid rgba(255,255,255,0.2);">
            <span>📍 Elevation</span>
            <span>PET</span>
            <span>PMV</span>
            <span>Risk</span>
        </div>
        <div class="height-row">
            <span>Ground Level (0m)</span>
            <span><strong>{ground_pet:.1f}°C</strong></span>
            <span>{ground_pmv:.2f}</span>
            <span>{get_thermal_risk_level(ground_pet)[1].split('-')[0].strip()}</span>
        </div>
        <div class="height-row highlight">
            <span>▶ Working Height ({height}m)</span>
            <span><strong style="color:#4ECDC4;">{pet_at_height:.1f}°C</strong></span>
            <span>{pmv_at_height:.2f}</span>
            <span>{get_thermal_risk_level(pet_at_height)[1].split('-')[0].strip()}</span>
        </div>
        <div class="height-row">
            <span>Top of Structure (100m)</span>
            <span>{pet_at_100:.1f}°C</span>
            <span>{height_profile['pmv_at_100']:.2f}</span>
            <span>{get_thermal_risk_level(pet_at_100)[1].split('-')[0].strip()}</span>
        </div>
        <div class="height-row" style="background:rgba(78,205,196,0.15); border-radius:4px; margin-top:0.5rem; padding:0.4rem 0.5rem;">
            <span><strong>📉 Thermal Reduction</strong></span>
            <span><strong style="color:#4ECDC4;">{reduction:.1f}°C</strong></span>
            <span colspan="2" style="text-align:right;">
                Lapse Rate: {lapse_rate:.3f}°C/m
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Temperature reduction visualization
    st.markdown("#### 📊 Thermal Gradient Visualization")
    max_val = ground_pet
    min_val = pet_at_100
    
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <div style="display:flex; justify-content:space-between; color:rgba(255,255,255,0.7); font-size:0.8rem;">
            <span>Ground: {ground_pet:.1f}°C</span>
            <span>Working Height: {pet_at_height:.1f}°C</span>
            <span>100m: {pet_at_100:.1f}°C</span>
        </div>
        <div class="bar-container">
            <div class="bar-fill" style="width:100%; background:#FF6B6B;"></div>
            <div class="bar-label">{ground_pet:.1f}°C</div>
        </div>
        <div class="bar-container">
            <div class="bar-fill" style="width:{((pet_at_height - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 50}%; background:#FFD93D;"></div>
            <div class="bar-label">{pet_at_height:.1f}°C</div>
        </div>
        <div class="bar-container">
            <div class="bar-fill" style="width:{((pet_at_100 - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 50}%; background:#4ECDC4;"></div>
            <div class="bar-label">{pet_at_100:.1f}°C</div>
        </div>
        <div style="text-align:center; color:rgba(255,255,255,0.5); font-size:0.75rem; margin-top:0.3rem;">
            ↓ Temperature decreases with elevation — utilize vertical work positioning for heat relief
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ 5. PRODUCTIVITY IMPACT ============
st.markdown('<div class="section-title">📉 Productivity Impact Assessment</div>', unsafe_allow_html=True)

current_work_key = activity_to_work.get(met, "Heavy (HW)")
work = work_data[current_work_key]

pet_effective = height_profile['pet_at_height'] if height_profile else ground_pet
productivity_loss = calc_productivity_loss(pet_effective, current_work_key, baseline_productivity)

# Risk classification for productivity
if productivity_loss == 0:
    loss_class, loss_icon, loss_desc = "low", "✅", "No productivity decrement"
elif productivity_loss < 10:
    loss_class, loss_icon, loss_desc = "low", "ℹ️", "Minimal productivity impact"
elif productivity_loss < 20:
    loss_class, loss_icon, loss_desc = "moderate", "⚠️", "Moderate productivity reduction"
else:
    loss_class, loss_icon, loss_desc = "critical", "🔴", "Severe productivity loss"

st.markdown(f"""
<div class="warning-box {loss_class}">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
            <strong style="font-size:1.1rem;">{current_work_key}</strong>
            <span style="color:rgba(255,255,255,0.6); font-size:0.9rem; margin-left:0.5rem;">
                ({work['description']})
            </span>
        </div>
        <div style="font-size:1.2rem; font-weight:700;">
            {loss_icon} {productivity_loss:.1f}% Loss — {loss_desc}
        </div>
    </div>
    <div style="margin-top:0.4rem; font-size:0.85rem; color:rgba(255,255,255,0.6);">
        PET Alert Level: {work['PET_AL']}°C | Working Height: {height}m | Baseline: {baseline_productivity:.0f} units/hr
    </div>
</div>
""", unsafe_allow_html=True)

# Work type comparison
st.markdown("#### 📊 Productivity Decrement by Work Classification")

loss_data = []
for wt in work_data.keys():
    loss = calc_productivity_loss(pet_effective, wt, baseline_productivity)
    loss_data.append((wt, loss))

max_loss = max([l for _, l in loss_data]) if loss_data else 1

for wt, loss in loss_data:
    is_current = (wt == current_work_key)
    bar_color = "#FF6B6B" if loss >= 20 else "#FFD93D" if loss >= 10 else "#4ECDC4"
    bar_width = max(5, (loss / 30) * 100) if loss > 0 else 5
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; margin:0.25rem 0; padding:0.2rem 0.5rem; 
                {'background:rgba(78,205,196,0.1); border-radius:6px;' if is_current else ''}">
        <div style="width:140px; font-size:0.85rem; font-weight:{'600' if is_current else '400'}; color:#ffffff;">
            {'▶ ' if is_current else '  '}{wt}
        </div>
        <div style="flex:1; background:rgba(255,255,255,0.1); border-radius:4px; height:20px; position:relative;">
            <div style="width:{bar_width}%; background:{bar_color}; height:100%; border-radius:4px;"></div>
            <div style="position:absolute; right:6px; top:2px; font-size:0.7rem; font-weight:600; color:#ffffff;">
                {loss:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ 6. HEIGHT-BASED WORK RECOMMENDATION ============
if height_profile and height > 0:
    st.markdown('<div class="section-title">🏗️ Height-Based Work Optimization</div>', unsafe_allow_html=True)
    
    ground_loss = calc_productivity_loss(ground_pet, current_work_key, baseline_productivity)
    height_loss = calc_productivity_loss(pet_at_height, current_work_key, baseline_productivity)
    loss_reduction = ground_loss - height_loss
    
    st.markdown(f"""
    <div style="background:rgba(78,205,196,0.1); padding:1rem; border-radius:12px; border:1px solid rgba(78,205,196,0.2);">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; text-align:center;">
            <div>
                <div style="color:rgba(255,255,255,0.6); font-size:0.75rem; text-transform:uppercase;">Ground Level</div>
                <div style="font-size:1.2rem; font-weight:700; color:#FF6B6B;">{ground_loss:.1f}% Loss</div>
            </div>
            <div>
                <div style="color:rgba(255,255,255,0.6); font-size:0.75rem; text-transform:uppercase;">At {height}m</div>
                <div style="font-size:1.2rem; font-weight:700; color:#4ECDC4;">{height_loss:.1f}% Loss</div>
            </div>
            <div>
                <div style="color:rgba(255,255,255,0.6); font-size:0.75rem; text-transform:uppercase;">Improvement</div>
                <div style="font-size:1.2rem; font-weight:700; color:#FFD93D;">{loss_reduction:.1f}% Better</div>
            </div>
        </div>
        <div style="text-align:center; margin-top:0.5rem; color:rgba(255,255,255,0.7); font-size:0.85rem;">
            {f'Positioning work at {height}m elevation reduces productivity loss by {loss_reduction:.1f}% compared to ground level' if loss_reduction > 1 else 'Minimal benefit from elevation — consider additional cooling measures'}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============ 7. SAFETY GUIDANCE ============
st.markdown('<div class="section-title">🛡️ Site Safety Guidance</div>', unsafe_allow_html=True)

risk_level_effective, risk_desc_effective, _, risk_class_effective = get_thermal_risk_level(pet_effective)

if risk_class_effective == "critical":
    guidance = f"""
    🔴 **EMERGENCY PROTOCOL — IMMEDIATE ACTION REQUIRED**
    
    • **Work Stoppage**: All non-essential outdoor construction activities must cease immediately
    • **Worker Relocation**: Move all personnel to air-conditioned rest areas or shaded locations
    • **Hydration Protocol**: Mandatory 250ml water intake every 15 minutes with electrolyte supplementation
    • **Medical Monitoring**: Activate site heat stress response team; monitor for heat exhaustion symptoms
    • **High-Risk Workers**: Prioritize medical evaluation for workers with pre-existing conditions
    • **Resumption Criteria**: Only resume work when PET drops below 35°C with adequate cooling measures
    """
elif risk_class_effective == "high":
    guidance = f"""
    🟠 **HIGH HEAT STRESS — ENHANCED PRECAUTIONS**
    
    • **Work-Rest Cycles**: Implement 45-minute work / 15-minute rest rotation in shaded areas
    • **Cooling Measures**: Provide cooling vests, ice packs, and misting stations at work locations
    • **Hydration**: 250ml water every 30 minutes; electrolyte drinks recommended for heavy work
    • **Monitoring**: Designate safety officer to monitor worker condition and PET levels hourly
    • **Work Modifications**: Schedule heavy work (VHW, HW) during cooler morning/evening hours
    • **Height Consideration**: {f'Utilize {height}m elevation for {reduction:.1f}°C thermal relief' if height_profile and height > 0 else 'Consider elevating work platforms for thermal relief'}
    """
elif risk_class_effective == "moderate":
    guidance = f"""
    🟡 **MODERATE HEAT STRESS — STANDARD PRECAUTIONS**
    
    • **Work-Rest Cycles**: 60-minute work / 10-minute rest in shaded areas for moderate activities
    • **Hydration**: 250ml water every 45 minutes; standard electrolyte availability
    • **Monitoring**: Regular observation of workers for early signs of heat strain
    • **Work Scheduling**: Continue normal operations with increased vigilance
    • **Cooling**: Ensure adequate shade and ventilation at all workstations
    • **Height Strategy**: {f'Working at {height}m provides {reduction:.1f}°C reduction — consider vertical work positioning' if height_profile and height > 0 else 'Monitor for increasing temperatures throughout the day'}
    """
else:
    guidance = f"""
    🟢 **NORMAL OPERATIONS — ROUTINE MONITORING**
    
    • **Work Schedule**: Standard work operations with regular breaks
    • **Hydration**: Maintain normal water intake (250ml per hour minimum)
    • **Monitoring**: Continue routine safety observations
    • **Preparation**: Maintain readiness for temperature increases during peak hours
    • **Best Practice**: {f'Working at {height}m provides optimal thermal conditions' if height_profile and height > 0 else 'Continue standard safety protocols'}
    """

st.markdown(f"""
<div class="guidance-box {risk_class_effective}">
    {guidance}
</div>
""", unsafe_allow_html=True)

# ============ 8. TECHNICAL NOTES ============
with st.expander("📐 Technical Notes & Methodology"):
    st.markdown("""
    **Thermal Exposure Assessment Parameters**
    
    | Parameter | Description | Application |
    |-----------|-------------|-------------|
    | **PET** | Physiological Equivalent Temperature (°C) | Primary heat stress indicator representing thermal sensation |
    | **PMV** | Predicted Mean Vote (0-3.5 scale) | Thermal comfort assessment for construction workers |
    | **PPD** | Predicted Percentage Dissatisfied (%) | Workforce acceptance of thermal conditions |
    | **RWS** | Relative Wind Speed (m/s) | Effective wind exposure at worker level |
    | **CE** | Cooling Effect (°C) | Evaporative cooling potential from wind |
    
    **Heat Stress Risk Thresholds (Construction Context)**
    
    | PET Range | Risk Level | Work Restriction |
    |-----------|------------|------------------|
    | >41°C | Critical | Emergency — All work suspended |
    | 35-41°C | High | 45-min work / 15-min rest |
    | 29-35°C | Moderate | 60-min work / 10-min rest |
    | 23-29°C | Low | Standard operations |
    | <23°C | Comfortable | Normal operations |
    
    **Vertical Thermal Gradient**
    
    PET decreases by approximately 0.05-0.08°C per meter of elevation above ground level. This effect can be utilized for:
    - Strategic work positioning on multi-story construction sites
    - Scheduling heavy work at higher elevations during peak heat
    - Planning cooling interventions for ground-level operations
    
    **Productivity Loss Calculation**
    
    Based on empirical models from construction site studies:
