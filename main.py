import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

# ============ FALLBACK DATA CREATION ============
def create_sample_data():
    """Create sample data if CSV doesn't exist"""
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
        'T': T,
        'RH': RH,
        'WS': WS,
        'PET': PET,
        'PMV': PMV,
        'PPD': 5 + 85 * (1 - np.exp(-0.5 * (PMV - 1))),
        'SET': PET - 1 + np.random.normal(0, 0.5, n_samples),
        'RWS': WS * 0.8 + np.random.normal(0, 0.2, n_samples),
        'CE': 2 + 0.5 * WS + np.random.normal(0, 0.3, n_samples),
        'Height': np.random.uniform(0, 100, n_samples),
        'PETH': PET - np.random.uniform(0, 6, n_samples),
        'PMVH': PMV - np.random.uniform(0, 1, n_samples)
    })
    return df

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="Heat Stress Predictor - Simple View",
    page_icon="📊",
    layout="centered"
)

# ============ LOAD DATA WITH FALLBACK ============
CSV_PATH = "EXBD.csv"

# Try to load, create if not exists
if not os.path.exists(CSV_PATH):
    st.info("📊 Creating sample data (EXBD.csv not found)")
    df = create_sample_data()
    df.to_csv(CSV_PATH, index=False)
else:
    try:
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except:
        st.warning("⚠️ Could not read EXBD.csv, using sample data")
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

# Train models
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

# ============ WORK TYPE DATA ============
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
    """Calculate productivity loss based on PET and work type"""
    work = work_data[work_type_key]
    PET_AL = work["PET_AL"]
    base_factor = work["base_factor"]
    
    if pet_value <= PET_AL:
        return 0
    else:
        delta_pet = pet_value - PET_AL
        PL = 30 * (1 - np.exp(-base_factor * delta_pet))
        return min(PL, 30)

def get_thermal_level(pet):
    """Get thermal level based on PET value"""
    if pet > 41.0:
        return "VERY HOT", "Extreme heat stress - Unsafe for physical work", "🔥", "red"
    elif pet > 35.0:
        return "HOT", "High heat stress - Significant strain", "🌡️", "orange"
    elif pet > 29.0:
        return "WARM", "Moderate heat stress - Reduced work capacity", "☀️", "gold"
    elif pet > 23.0:
        return "SLIGHTLY WARM", "Slight heat stress - Comfortable", "⛅", "lightblue"
    else:
        return "COMFORTABLE", "No heat stress - Optimal conditions", "✅", "green"

def get_pmv_level(pmv):
    """Get PMV level description"""
    if pmv >= 3.0:
        return "EXTREME DISCOMFORT"
    elif pmv >= 2.5:
        return "VERY DISCOMFORT"
    elif pmv >= 2.0:
        return "MODERATE DISCOMFORT"
    elif pmv >= 1.5:
        return "DISCOMFORT"
    elif pmv >= 1.0:
        return "SLIGHT DISCOMFORT"
    else:
        return "COMFORTABLE"

def get_height_profile(ground_pet, ground_pmv, height_m, bh_df):
    """Calculate PET and PMV at different heights"""
    if len(bh_df) == 0:
        return None, None
    
    heights_original = bh_df['Height(m)'].values
    pet_pattern = bh_df['PET(0C)'].values.copy()
    pmv_pattern = bh_df['PMV'].values.copy() if 'PMV' in bh_df.columns else pet_pattern * 0.08
    
    # Ensure monotonic decrease
    for i in range(1, len(pet_pattern)):
        if pet_pattern[i] > pet_pattern[i-1]:
            pet_pattern[i] = max(pet_pattern[i-1] - 0.1, 0)
    
    for i in range(1, len(pmv_pattern)):
        if pmv_pattern[i] > pmv_pattern[i-1]:
            pmv_pattern[i] = max(pmv_pattern[i-1] - 0.01, -3)
    
    # Calculate relative values
    if pet_pattern[0] != 0:
        pet_relative = pet_pattern / pet_pattern[0]
    else:
        pet_relative = pet_pattern
    
    if pmv_pattern[0] != 0:
        pmv_relative = pmv_pattern / pmv_pattern[0]
    else:
        pmv_relative = pmv_pattern
    
    # Apply scaling to ground values
    pet_profile = ground_pet * pet_relative
    pmv_profile = ground_pmv * pmv_relative
    
    # Interpolate at specific heights
    heights_smooth = np.linspace(0, 100, 100)
    pet_smooth = np.interp(heights_smooth, heights_original, pet_profile)
    pmv_smooth = np.interp(heights_smooth, heights_original, pmv_profile)
    
    # Get values at requested height
    if 0 <= height_m <= 100:
        pet_at_height = np.interp(height_m, heights_smooth, pet_smooth)
        pmv_at_height = np.interp(height_m, heights_smooth, pmv_smooth)
    else:
        pet_at_height = ground_pet
        pmv_at_height = ground_pmv
    
    # Get 100m values
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
        'reduction': ground_pet - pet_at_height
    }

# ============ UI STYLING ============
st.markdown("""
<style>
    .main-title {font-size: 2.2rem; font-weight: 700; color: #1a1a2e; text-align: center; padding: 1rem 0; border-bottom: 3px solid #4a90d9; margin-bottom: 1.5rem;}
    .section-title {font-size: 1.3rem; font-weight: 600; color: #1a1a2e; margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.3rem; border-bottom: 2px solid #e0e0e0;}
    .result-row {display: flex; justify-content: space-between; padding: 0.6rem 0.8rem; background-color: #f8f9fa; border-radius: 6px; margin: 0.25rem 0; border-left: 4px solid #4a90d9;}
    .result-label {font-weight: 500; color: #333;}
    .result-value {font-weight: 600; color: #1a1a2e; font-size: 1.05rem;}
    .warning-box {background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 0.8rem 1.2rem; margin: 0.8rem 0;}
    .warning-box.good {background-color: #d4edda; border-color: #28a745;}
    .warning-box.danger {background-color: #f8d7da; border-color: #dc3545;}
    .level-badge {display: inline-block; padding: 0.3rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem;}
    .level-badge.red {background: #8B0000; color: white;}
    .level-badge.orange {background: #DC3545; color: white;}
    .level-badge.gold {background: #FF8C00; color: white;}
    .level-badge.lightblue {background: #FFD700; color: #333;}
    .level-badge.green {background: #28A745; color: white;}
    .divider {border: none; border-top: 2px dashed #ddd; margin: 1.5rem 0;}
    .footer-text {text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;}
    .input-label {font-weight: 500; color: #333; margin-bottom: 0.2rem;}
    .metric-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin: 0.5rem 0;}
    .metric-item {background: #f8f9fa; padding: 0.5rem 0.8rem; border-radius: 6px; text-align: center;}
    .metric-item .label {font-size: 0.8rem; color: #666;}
    .metric-item .value {font-size: 1.3rem; font-weight: 700; color: #1a1a2e;}
    .status-box {padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.3rem; font-weight: bold; margin: 0.5rem 0;}
    .status-box.red {background: #8B0000; color: white;}
    .status-box.orange {background: #DC3545; color: white;}
    .status-box.gold {background: #FF8C00; color: white;}
    .status-box.lightblue {background: #FFD700; color: #333;}
    .status-box.green {background: #28A745; color: white;}
    .height-profile {background: #f0f4f8; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;}
    .height-row {display: flex; justify-content: space-between; padding: 0.3rem 0.5rem; border-bottom: 1px solid #e0e0e0;}
    .height-row:last-child {border-bottom: none;}
    .bar-container {background: #eee; border-radius: 4px; height: 20px; position: relative; margin: 0.2rem 0;}
    .bar-fill {height: 100%; border-radius: 4px; transition: width 0.3s;}
</style>
""", unsafe_allow_html=True)

# ============ TITLE ============
st.markdown('<div class="main-title">🌡️ Heat Stress Predictor — Simple View</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #555; font-style: italic;">Text-based output with building height calculations</p>', unsafe_allow_html=True)

# ============ SIDEBAR INPUTS ============
with st.sidebar:
    st.markdown("### ⚙️ Input Parameters")
    st.markdown("---")
    
    st.markdown("#### 🌤️ Weather Conditions")
    T = st.number_input("Temperature (°C)", 20.0, 50.0, 34.0, 0.1)
    RH = st.number_input("Humidity (%)", 0.0, 100.0, 65.0, 1.0)
    WS = st.number_input("Wind Speed (m/s)", 0.0, 10.0, 1.5, 0.1)
    
    st.markdown("---")
    st.markdown("#### 👕 Personal Factors")
    clo = st.select_slider(
        "Clothing (clo)",
        options=[0.36, 0.50, 0.57, 0.61, 0.96, 1.00],
        value=0.57
    )
    met = st.select_slider(
        "Activity (met)",
        options=[2.1, 2.2, 2.6, 3.2, 3.8, 4.0],
        value=3.2
    )
    
    st.markdown("---")
    st.markdown("#### 🏗️ Building Height")
    height = st.slider("Working Height (m)", 0, 100, 0, 1)
    st.caption("PET decreases by ~0.07°C per meter height")
    
    st.markdown("---")
    st.markdown("#### 📊 Productivity")
    baseline_productivity = st.number_input("Baseline (units/hr)", min_value=1.0, value=100.0, step=5.0)
    
    st.markdown("---")
    if st.button("🔄 Calculate", use_container_width=True):
        st.rerun()

# ============ MAKE PREDICTIONS ============
input_data = np.array([[T, RH, WS]])
input_scaled = scaler.transform(input_data)

predictions = {}
for target in targets:
    if target in models:
        predictions[target] = models[target].predict(input_scaled)[0]
    else:
        # Physics fallback
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

# Get ground values
ground_pet = predictions["PET(0C)"]
ground_pmv = predictions["PMV"]

# ============ HEIGHT PROFILE CALCULATION ============
height_profile = get_height_profile(ground_pet, ground_pmv, height, bh_df)

# ============ DISPLAY RESULTS ============

# 1. Overall Status
thermal_level, thermal_desc, thermal_icon, color_class = get_thermal_level(ground_pet)
pmv_level = get_pmv_level(ground_pmv)

st.markdown(f"""
<div class="status-box {color_class}">
    {thermal_icon} {thermal_level} — {thermal_desc}
</div>
""", unsafe_allow_html=True)

# 2. Input Summary
st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">🌡️ Temperature</div>
        <div class="value">{T:.1f} °C</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">💧 Humidity</div>
        <div class="value">{RH:.0f} %</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-item">
        <div class="label">💨 Wind Speed</div>
        <div class="value">{WS:.1f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.3rem;">
    <div class="metric-item"><div class="label">👕 Clothing</div><div class="value">{clo:.2f} clo</div></div>
    <div class="metric-item"><div class="label">💪 Activity</div><div class="value">{met:.1f} met</div></div>
</div>
""", unsafe_allow_html=True)

# 3. Thermal Metrics
st.markdown('<div class="section-title">🌡️ Thermal Metrics (Ground Level)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="result-row">
        <span class="result-label">🌡️ PET (Feels Like Temp)</span>
        <span class="result-value" style="color: {'#DC3545' if ground_pet > 35 else '#FF8C00' if ground_pet > 29 else '#28A745'};">{ground_pet:.1f} °C</span>
    </div>
    <div class="result-row">
        <span class="result-label">📊 PMV (Thermal Comfort)</span>
        <span class="result-value">{ground_pmv:.2f}</span>
    </div>
    <div class="result-row">
        <span class="result-label">📋 PMV Interpretation</span>
        <span class="result-value" style="font-size:0.95rem;">{pmv_level}</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="result-row">
        <span class="result-label">😓 PPD (% Dissatisfied)</span>
        <span class="result-value">{predictions['PPD(%)']:.1f}%</span>
    </div>
    <div class="result-row">
        <span class="result-label">🌬️ Relative Wind Speed</span>
        <span class="result-value">{predictions['RWS(m/s)']:.1f} m/s</span>
    </div>
    <div class="result-row">
        <span class="result-label">❄️ Cooling Effect</span>
        <span class="result-value">{predictions['CE(0C)']:.1f} °C</span>
    </div>
    """, unsafe_allow_html=True)

# 4. Height Profile Analysis
st.markdown('<div class="section-title">🏗️ Building Height Analysis</div>', unsafe_allow_html=True)

if height_profile:
    pet_at_height = height_profile['pet_at_height']
    pmv_at_height = height_profile['pmv_at_height']
    pet_at_100 = height_profile['pet_at_100']
    reduction = height_profile['reduction']
    
    st.markdown(f"""
    <div class="height-profile">
        <div class="height-row">
            <span><strong>📍 Ground Level (0m):</strong></span>
            <span><strong>PET = {ground_pet:.1f}°C</strong> | PMV = {ground_pmv:.2f}</span>
        </div>
        <div class="height-row" style="background: #e3f2fd;">
            <span><strong>📍 Working Height ({height}m):</strong></span>
            <span><strong>PET = {pet_at_height:.1f}°C</strong> | PMV = {pmv_at_height:.2f}</span>
        </div>
        <div class="height-row">
            <span><strong>📍 Top (100m):</strong></span>
            <span><strong>PET = {pet_at_100:.1f}°C</strong></span>
        </div>
        <div class="height-row" style="background: #fff3cd;">
            <span><strong>📉 Temperature Reduction:</strong></span>
            <span><strong style="color: {'#28A745' if reduction > 3 else '#FF8C00' if reduction > 1 else '#DC3545'};">{reduction:.1f}°C</strong> (from ground to {height}m)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Height profile text-based visualization
    st.markdown("#### 📊 Vertical Profile (Text View)")
    
    # Create a simple text-based visualization
    profile_data = []
    for h in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        if h in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            idx = int(h / 100 * 99) if h < 100 else 99
            pet_val = height_profile['pet_profile'][idx]
            pmv_val = height_profile['pmv_profile'][idx]
            profile_data.append((h, pet_val, pmv_val))
    
    # Display as a table-like format
    st.markdown("""
    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; font-family: monospace;">
    <div style="display: grid; grid-template-columns: 80px 80px 80px; border-bottom: 2px solid #333; padding: 0.3rem 0; font-weight: bold;">
        <span>Height (m)</span><span>PET (°C)</span><span>PMV</span>
    </div>
    """, unsafe_allow_html=True)
    
    for h, pet_val, pmv_val in profile_data:
        is_current = abs(h - height) < 1
        bg_color = "#e3f2fd" if is_current else "transparent"
        marker = " 👈" if is_current else ""
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 80px 80px 80px; padding: 0.2rem 0; background: {bg_color};">
            <span>{h:.0f}{marker}</span>
            <span>{pet_val:.1f}</span>
            <span>{pmv_val:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visual bar representation of temperature reduction
    st.markdown("#### 📉 Temperature Reduction Visualization")
    max_val = ground_pet
    min_val = pet_at_100
    
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
            <span>Ground: {ground_pet:.1f}°C</span>
            <span>100m: {pet_at_100:.1f}°C</span>
        </div>
        <div class="bar-container">
            <div class="bar-fill" style="width: 100%; background: #DC3545;"></div>
        </div>
        <div class="bar-container">
            <div class="bar-fill" style="width: {((pet_at_100 - min_val) / (max_val - min_val)) * 100 if max_val != min_val else 50}%; background: #28A745;"></div>
        </div>
        <div style="text-align: center; font-size: 0.8rem; color: #666;">
            ↓ Temperature decreases as height increases
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Thermal level at working height
    level_at_height, desc_at_height, icon_at_height, color_at_height = get_thermal_level(pet_at_height)
    st.markdown(f"""
    <div style="background: #f0f4f8; padding: 0.8rem; border-radius: 6px; text-align: center; margin: 0.5rem 0; border-left: 4px solid {'#DC3545' if color_at_height == 'red' else '#FF8C00' if color_at_height == 'gold' else '#28A745'};">
        <strong>At {height}m working height:</strong> {icon_at_height} {level_at_height} — {desc_at_height}
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("ℹ️ Height profile data not available. Please ensure the dataset contains height information.")

# 5. Productivity Loss Analysis
st.markdown('<div class="section-title">📉 Productivity Loss Analysis</div>', unsafe_allow_html=True)

current_work_key = activity_to_work.get(met, "Heavy (HW)")
work = work_data[current_work_key]

# Calculate productivity loss at current height
if height_profile:
    pet_for_productivity = height_profile['pet_at_height']
else:
    pet_for_productivity = ground_pet

productivity_loss = calc_productivity_loss(pet_for_productivity, current_work_key, baseline_productivity)

# Productivity status
if productivity_loss == 0:
    loss_status, loss_color = "✅ No productivity loss", "good"
elif productivity_loss < 10:
    loss_status, loss_color = f"ℹ️ {productivity_loss:.1f}% loss - Minimal impact", "warning-box"
elif productivity_loss < 20:
    loss_status, loss_color = f"⚠️ {productivity_loss:.1f}% loss - Moderate impact", "warning-box"
else:
    loss_status, loss_color = f"🔴 {productivity_loss:.1f}% loss - Severe impact", "danger"

st.markdown(f"""
<div class="warning-box {loss_color}">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <strong>{current_work_key}</strong>
            <span style="color: #666; font-size: 0.9rem;">({work['description']})</span>
        </div>
        <div style="font-size: 1.2rem; font-weight: 700;">
            {loss_status}
        </div>
    </div>
    <div style="margin-top: 0.4rem; font-size: 0.9rem; color: #555;">
        📊 PET Alert Level: {work['PET_AL']}°C | Working Height: {height}m
    </div>
</div>
""", unsafe_allow_html=True)

# All work types comparison
st.markdown("#### 📊 Productivity Loss by Work Type")

# Calculate losses at current height
loss_data = []
for wt in work_data.keys():
    loss = calc_productivity_loss(pet_for_productivity, wt, baseline_productivity)
    loss_data.append((wt, loss))

max_loss = max([l for _, l in loss_data]) if loss_data else 1

for wt, loss in loss_data:
    is_current = (wt == current_work_key)
    bar_color = "#DC3545" if loss >= 20 else "#FF8C00" if loss >= 10 else "#28A745"
    bar_width = max(5, (loss / 30) * 100) if loss > 0 else 5
    marker = "▶" if is_current else " "
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 0.3rem 0; padding: 0.2rem 0; {'background: #e3f2fd; border-radius: 4px;' if is_current else ''}">
        <div style="width: 130px; font-size: 0.85rem; font-weight: {'600' if is_current else '400'};">
            {marker} {wt}
        </div>
        <div style="flex: 1; background: #eee; border-radius: 4px; height: 20px; position: relative;">
            <div style="width: {bar_width}%; background: {bar_color}; height: 100%; border-radius: 4px;"></div>
            <div style="position: absolute; right: 6px; top: 2px; font-size: 0.7rem; font-weight: 600; color: {'white' if loss > 15 else '#333'};">
                {loss:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 6. Impact of Height on Productivity
if height_profile:
    st.markdown('<div class="section-title">📐 Height Impact on Productivity</div>', unsafe_allow_html=True)
    
    # Calculate losses at different heights
    height_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    height_impact_data = []
    
    for h in height_levels:
        idx = int(h / 100 * 99) if h < 100 else 99
        pet_at_h = height_profile['pet_profile'][idx]
        loss_at_h = calc_productivity_loss(pet_at_h, current_work_key, baseline_productivity)
        height_impact_data.append((h, pet_at_h, loss_at_h))
    
    st.markdown("""
    <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 6px; font-family: monospace; font-size: 0.85rem;">
    <div style="display: grid; grid-template-columns: 80px 80px 80px 80px; border-bottom: 2px solid #333; padding: 0.3rem 0; font-weight: bold;">
        <span>Height</span><span>PET</span><span>Loss %</span><span>Status</span>
    </div>
    """, unsafe_allow_html=True)
    
    for h, pet_at_h, loss_at_h in height_impact_data:
        is_current = abs(h - height) < 1
        bg_color = "#e3f2fd" if is_current else "transparent"
        status_icon = "✅" if loss_at_h == 0 else "⚠️" if loss_at_h < 15 else "🔴"
        marker = " 👈" if is_current else ""
        
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: 80px 80px 80px 80px; padding: 0.2rem 0; background: {bg_color};">
            <span>{h:.0f}m{marker}</span>
            <span>{pet_at_h:.1f}°C</span>
            <span>{loss_at_h:.1f}%</span>
            <span>{status_icon}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendation based on height
    if height > 50:
        st.success(f"✅ Working at {height}m reduces PET by {height_profile['reduction']:.1f}°C, lowering productivity loss from {calc_productivity_loss(ground_pet, current_work_key, baseline_productivity):.1f}% to {productivity_loss:.1f}%")
    elif height > 20:
        st.info(f"ℹ️ Working at {height}m provides moderate heat relief ({height_profile['reduction']:.1f}°C reduction)")
    else:
        st.warning(f"⚠️ Working at {height}m provides minimal heat relief. Consider moving to higher levels if possible.")

# 7. Safety Guidance
st.markdown('<div class="section-title">📋 Safety Guidance</div>', unsafe_allow_html=True)

# Determine severity based on PET at working height
pet_effective = height_profile['pet_at_height'] if height_profile else ground_pet

if pet_effective > 41:
    guidance = """
    🔴 **EXTREME HEAT — STOP WORK IMMEDIATELY**
    • All outdoor work must be suspended
    • Move to air-conditioned areas
    • Drink water every 15 minutes
    • Watch for: headache, dizziness, confusion
    • Emergency protocol: Activate heat stress response team
    """
elif pet_effective > 35:
    guidance = """
    🟠 **HIGH HEAT — TAKE PRECAUTIONS**
    • Limit work to 45-minute sessions with 15-minute breaks
    • Wear light, breathable clothing
    • Drink water every 30 minutes
    • Use shaded rest areas
    • Rotate workers to minimize exposure
    """
elif pet_effective > 29:
    guidance = """
    🟡 **MODERATE HEAT — STAY ALERT**
    • Take regular breaks in shade (10 min every hour)
    • Maintain hydration (250ml every 30 min)
    • Normal work schedule with monitoring
    • Watch for signs of heat exhaustion
    """
else:
    guidance = """
    🟢 **COMFORTABLE — NORMAL WORK**
    • Regular hydration recommended
    • Standard work schedule
    • No heat-related restrictions
    • Continue normal safety protocols
    """

st.markdown(f"""
<div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #4a90d9; white-space: pre-line; font-size: 0.95rem;">
{guidance}
</div>
""", unsafe_allow_html=True)

# 8. Height-Specific Recommendation
if height_profile and height > 0:
    st.markdown(f"""
    <div style="background: #e8f5e9; padding: 0.8rem; border-radius: 6px; margin-top: 0.5rem; border-left: 4px solid #2E7D32;">
        <strong>🏗️ Height-Specific Recommendation:</strong><br>
        Working at <strong>{height}m</strong> reduces thermal exposure by <strong>{height_profile['reduction']:.1f}°C</strong> 
        compared to ground level. This results in <strong>{productivity_loss:.1f}%</strong> productivity loss 
        vs. {calc_productivity_loss(ground_pet, current_work_key, baseline_productivity):.1f}% at ground level 
        for {current_work_key}.
    </div>
    """, unsafe_allow_html=True)

# 9. Explanation Section
with st.expander("ℹ️ Understanding the Results"):
    st.markdown("""
    ### What do these numbers mean?
    
    **PET (Physiological Equivalent Temperature)**
    - The temperature at which your body would feel the same thermal stress
    - Higher PET = more heat stress on your body
    
    **Thermal Levels:**
    - **>41°C**: 🔴 Very Hot — Dangerous for physical work
    - **35-41°C**: 🟠 Hot — High strain, limit work duration
    - **29-35°C**: 🟡 Warm — Moderate strain, take breaks
    - **23-29°C**: 🟢 Slightly Warm — Comfortable
    - **<23°C**: ✅ Comfortable — No heat stress
    
    **PMV (Predicted Mean Vote)**
    - How warm/cold people feel on average
    - Scale: 0 (neutral) to 3.5 (very hot)
    
    **Building Height Impact**
    - PET decreases by approximately 0.05-0.08°C per meter of height
    - Higher working levels provide thermal relief
    - This is due to decreasing air temperature with altitude
    
    **Productivity Loss**
    - Estimated reduction in work output due to heat stress
    - Based on research from construction sites
    - Varies by work type (heavier work = more loss)
    
    ### How to use this information:
    1. **Check your PET level** to understand heat stress severity
    2. **Look at the height profile** to see how temperature changes
    3. **Review productivity loss** for different work types
    4. **Follow safety guidance** based on your situation
    5. **Consider working at higher levels** if possible for heat relief
    """)

# 10. Footer
st.markdown('<div class="footer-text">🌡️ Heat Stress Predictor — Simple Text Interface | Height Analysis Included</div>', unsafe_allow_html=True)
