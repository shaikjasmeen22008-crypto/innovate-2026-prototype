import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="TransitPulse - Global Flow & Surge Predictor",
    page_icon="🌍",
    layout="wide"
)

# ----------------- 1. Global Location Presets -----------------
GLOBAL_PRESETS = {
    "Tokyo - Yamanote Core (Japan)": {
        "zone_type": "Ultra-Dense High-Frequency Rail",
        "peak_hours": [8, 9, 18, 19],
        "base_traffic": 85,
        "weather_sensitivity": 1.3,
        "typical_weather": "Rain",
        "lat": 35.6895, "lon": 139.6917
    },
    "London - Central Red Route (UK)": {
        "zone_type": "Congestion-Charge Urban Grid",
        "peak_hours": [8, 9, 17, 18],
        "base_traffic": 78,
        "weather_sensitivity": 1.7,
        "typical_weather": "Fog",
        "lat": 51.5074, "lon": -0.1278
    },
    "New York - Midtown Manhattan Arterial (USA)": {
        "zone_type": "Mixed Multi-Modal Grid",
        "peak_hours": [8, 9, 17, 18, 19],
        "base_traffic": 90,
        "weather_sensitivity": 1.5,
        "typical_weather": "Heavy Storm",
        "lat": 40.7589, "lon": -73.9851
    },
    "Singapore - Marina Bay Spine (Singapore)": {
        "zone_type": "Smart Dynamic Pricing Corridor",
        "peak_hours": [8, 9, 18, 19],
        "base_traffic": 60,
        "weather_sensitivity": 1.4,
        "typical_weather": "Rain",
        "lat": 1.2868, "lon": 103.8545
    },
    "Bengaluru - Outer Ring Road IT Corridor (India)": {
        "zone_type": "High-Density Suburban Arterial",
        "peak_hours": [9, 10, 18, 19, 20],
        "base_traffic": 88,
        "weather_sensitivity": 1.9,
        "typical_weather": "Rain",
        "lat": 12.9352, "lon": 77.6946
    }
}

# ----------------- 2. Synthetic Data Generation -----------------
@st.cache_data
def generate_training_data():
    np.random.seed(42)
    records_per_zone = 200
    weather_types = ["Clear", "Rain", "Fog", "Heavy Storm"]
    zone_profiles = ["High-Density Grid", "Suburban Arterial", "Expressway Corridor", "Freight/Port Spine"]
    
    records = []
    weather_impacts = {"Clear": 0, "Rain": 7, "Fog": 11, "Heavy Storm": 19}

    for zone in zone_profiles:
        hours = np.random.randint(0, 24, records_per_zone)
        weathers = np.random.choice(weather_types, records_per_zone, p=[0.5, 0.25, 0.15, 0.1])
        traffics = np.random.uniform(20, 100, records_per_zone)
        headways = np.random.choice([5, 10, 15, 20], records_per_zone)
        
        for i in range(records_per_zone):
            h, w, t, hw = hours[i], weathers[i], traffics[i], headways[i]
            is_peak = 1 if (8 <= h <= 10 or 17 <= h <= 20) else 0
            
            # Non-linear synthetic delay function
            delay = (t * 0.3) + weather_impacts[w] + (is_peak * 10) + (hw * 0.25) + np.random.normal(0, 2)
            delay = max(2.0, round(delay, 1))
            
            records.append({
                "zone_type": zone,
                "hour": h,
                "weather": w,
                "traffic_index": round(t, 1),
                "scheduled_frequency_min": hw,
                "delay_minutes": delay
            })
            
    return pd.DataFrame(records)

df = generate_training_data()

# ----------------- 3. Machine Learning Model Training -----------------
@st.cache_resource
def train_model(data):
    feature_df = pd.get_dummies(
        data[["zone_type", "hour", "weather", "traffic_index", "scheduled_frequency_min"]],
        drop_first=False
    )
    X = feature_df
    y = data["delay_minutes"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(X_train, y_train)
    r2 = round(model.score(X_test, y_test), 2)
    return model, X.columns.tolist(), r2

model, feature_cols, r2_score = train_model(df)

# ----------------- 4. Header & KPI Cards -----------------
st.title("🌍 TransitPulse: Global Transit Surge Predictor")
st.caption("Universal, location-adaptive delay forecasting and fleet intervention engine | INNOVATE Round 1")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Coverage Scope", "Global (Any Lat/Lon)")
k2.metric("Telemetry Training Points", f"{len(df):,}")
k3.metric("Model Baseline R²", f"{r2_score}")
k4.metric("Avg Benchmark Delay", f"{df['delay_minutes'].mean():.1f} min")

st.markdown("---")

# ----------------- 5. Location Source Selector (Accessible Anywhere) -----------------
st.sidebar.header("📍 Location Setup")

location_mode = st.sidebar.radio(
    "Choose Location Input Method:",
    ["Global City Presets", "Custom Global Coordinates (Anywhere)"]
)

if location_mode == "Global City Presets":
    selected_name = st.sidebar.selectbox("Select Benchmark City", list(GLOBAL_PRESETS.keys()))
    loc_info = GLOBAL_PRESETS[selected_name]
    lat = loc_info["lat"]
    lon = loc_info["lon"]
    zone_category = "High-Density Grid"
    default_weather = loc_info["typical_weather"]
    default_traffic = loc_info["base_traffic"]
else:
    selected_name = st.sidebar.text_input("Location Name / Corridor Label", value="Custom Operational Sector")
    col_lat, col_lon = st.sidebar.columns(2)
    lat = col_lat.number_input("Latitude", value=25.2048, format="%.4f")   # Default to Dubai
    lon = col_lon.number_input("Longitude", value=55.2708, format="%.4f")
    
    zone_category = st.sidebar.selectbox(
        "Corridor Classification",
        ["High-Density Grid", "Suburban Arterial", "Expressway Corridor", "Freight/Port Spine"]
    )
    default_weather = "Clear"
    default_traffic = 70

st.sidebar.markdown(f"**Current Coordinates:** `{lat:.4f}, {lon:.4f}`")

# ----------------- 6. Operational Parameters -----------------
st.sidebar.subheader("⚙️ Local Telemetry & Weather")

selected_hour = st.sidebar.slider("Local Corridor Time (0-23 hr)", 0, 23, value=18)
is_peak = (8 <= selected_hour <= 10) or (17 <= selected_hour <= 20)

weather_opts = ["Clear", "Rain", "Fog", "Heavy Storm"]
selected_weather = st.sidebar.selectbox("Simulated Local Weather", weather_opts, index=weather_opts.index(default_weather))

selected_traffic = st.sidebar.slider(
    "Local Traffic Density Index (0-100)", 10, 100, 
    value=min(100, default_traffic + (15 if is_peak else -10))
)

selected_freq = st.sidebar.selectbox("Current Vehicle Headway (mins)", [5, 10, 15, 20], index=1)

# ----------------- 7. Real-Time Inference -----------------
input_dict = {col: 0 for col in feature_cols}
if f"zone_type_{zone_category}" in input_dict:
    input_dict[f"zone_type_{zone_category}"] = 1
if f"weather_{selected_weather}" in input_dict:
    input_dict[f"weather_{selected_weather}"] = 1
input_dict["hour"] = selected_hour
input_dict["traffic_index"] = selected_traffic
input_dict["scheduled_frequency_min"] = selected_freq

input_df = pd.DataFrame([input_dict])
predicted_delay = float(model.predict(input_df)[0])

# ----------------- 8. Operational Forecast & Interactive Map -----------------
col_pred, col_map = st.columns([1, 1])

with col_pred:
    st.subheader(f"Forecast for: {selected_name}")
    
    if is_peak:
        st.caption("🔴 **Peak Congestion Window Active**")
    else:
        st.caption("🟢 **Off-Peak Standard Operations**")

    if predicted_delay > 35:
        st.error(f"🚨 **High Delay Warning:** Estimated delay is **{predicted_delay:.1f} mins**")
        st.markdown("**Prescriptive AI Directives:**")
        st.info("⚡ Immediately dispatch 2 standby auxiliary fleet units and trigger transit signal priority at arterial intersections.")
    elif predicted_delay > 20:
        st.warning(f"⚠️ **Moderate Congestion Warning:** Estimated delay is **{predicted_delay:.1f} mins**")
        st.markdown("**Prescriptive AI Directives:**")
        st.info("⏱️ Compress scheduled dispatch headway by 3 minutes to prevent platform queue buildup.")
    else:
        st.success(f"✅ **Normal Corridor Flow:** Estimated delay is **{predicted_delay:.1f} mins**")
        st.markdown("**Prescriptive AI Directives:**")
        st.info("✔️ Operations running nominally. Maintain scheduled timetable.")

with col_map:
    st.subheader("Global Corridor Geo-Location")
    current_loc_df = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "Corridor": selected_name,
        "Delay (min)": round(predicted_delay, 1)
    }])
    st.map(current_loc_df, latitude="lat", longitude="lon", zoom=11)

st.markdown("---")

# ----------------- 9. Global Comparative Analytics -----------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Delay Distribution Across Weather Types")
    fig_box = px.box(
        df,
        x="weather",
        y="delay_minutes",
        color="weather",
        title="Weather Severity vs. Delay Vulnerability",
        labels={"delay_minutes": "Delay (mins)", "weather": "Condition"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

with c2:
    st.subheader("24-Hour Congestion by Zone Classification")
    hourly_zone = df.groupby(["hour", "zone_type"])["delay_minutes"].mean().reset_index()
    fig_line = px.line(
        hourly_zone,
        x="hour",
        y="delay_minutes",
        color="zone_type",
        title="Diurnal Delay Patterns by Corridor Classification",
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)