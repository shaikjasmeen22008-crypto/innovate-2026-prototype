import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="TransitPulse - Location-Based Flow & Surge Predictor",
    page_icon="📍",
    layout="wide"
)

# ----------------- 1. Location Profile Metadata -----------------
# Defines realistic environmental and traffic dynamics per location
LOCATION_PROFILES = {
    "Downtown Central Corridor (Metro Core)": {
        "zone_type": "Dense Commercial",
        "peak_hours": [8, 9, 17, 18, 19],
        "base_traffic": 75,
        "weather_sensitivity": 1.4,
        "typical_weather": "Clear",
        "lat": 12.9716, "lon": 77.5946
    },
    "Tech Park Ring Road (Suburban Express)": {
        "zone_type": "High-Density IT Hub",
        "peak_hours": [9, 10, 18, 19, 20],
        "base_traffic": 85,
        "weather_sensitivity": 1.8,
        "typical_weather": "Rain",
        "lat": 12.9352, "lon": 77.6946
    },
    "Harbor Logistics Spine (Industrial Port)": {
        "zone_type": "Heavy Freight Corridor",
        "peak_hours": [5, 6, 14, 15, 22],
        "base_traffic": 60,
        "weather_sensitivity": 2.0,
        "typical_weather": "Fog",
        "lat": 13.0827, "lon": 80.2707
    },
    "Airport Boulevard Link (Expressway)": {
        "zone_type": "High-Speed Transit Arterial",
        "peak_hours": [6, 7, 11, 16, 21],
        "base_traffic": 45,
        "weather_sensitivity": 1.2,
        "typical_weather": "Clear",
        "lat": 13.1986, "lon": 77.7066
    },
    "Old City Market Arterial (Narrow Grid)": {
        "zone_type": "Mixed Pedestrian & Transit Grid",
        "peak_hours": [11, 12, 17, 18, 19, 20],
        "base_traffic": 90,
        "weather_sensitivity": 1.6,
        "typical_weather": "Clear",
        "lat": 12.9634, "lon": 77.5750
    }
}

# ----------------- 2. Location-Grounded Synthetic Data -----------------
@st.cache_data
def generate_location_data():
    np.random.seed(42)
    records_per_loc = 150
    all_data = []

    weather_types = ["Clear", "Rain", "Fog", "Heavy Storm"]
    
    for loc_name, profile in LOCATION_PROFILES.items():
        hours = np.random.randint(0, 24, records_per_loc)
        
        # Weather distribution weighted by location profile
        if profile["typical_weather"] == "Rain":
            w_probs = [0.3, 0.45, 0.1, 0.15]
        elif profile["typical_weather"] == "Fog":
            w_probs = [0.3, 0.1, 0.5, 0.1]
        else:
            w_probs = [0.65, 0.2, 0.1, 0.05]
            
        weathers = np.random.choice(weather_types, records_per_loc, p=w_probs)
        
        # Traffic index dynamically calculated around base level and peak hours
        traffic_indices = []
        for h in hours:
            peak_boost = 25 if h in profile["peak_hours"] else -15
            traffic_val = np.clip(profile["base_traffic"] + peak_boost + np.random.normal(0, 8), 10, 100)
            traffic_indices.append(traffic_val)
        
        freqs = np.random.choice([5, 10, 15, 20], records_per_loc)
        
        # Calculate resulting delay based on location parameters
        weather_weight = {"Clear": 0, "Rain": 6, "Fog": 10, "Heavy Storm": 18}
        
        for i in range(records_per_loc):
            w = weathers[i]
            t = traffic_indices[i]
            h = hours[i]
            f = freqs[i]
            
            w_impact = weather_weight[w] * profile["weather_sensitivity"]
            peak_penalty = 12 if h in profile["peak_hours"] else 2
            noise = np.random.normal(0, 2)
            
            delay = (t * 0.28) + w_impact + peak_penalty + (f * 0.3) + noise
            delay = max(1.5, round(delay, 1))
            
            all_data.append({
                "location": loc_name,
                "zone_type": profile["zone_type"],
                "hour": h,
                "weather": w,
                "traffic_index": round(t, 1),
                "scheduled_frequency_min": f,
                "delay_minutes": delay
            })
            
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.date_range("2026-11-01", periods=len(df), freq="h")
    return df

df = generate_location_data()

# ----------------- 3. Machine Learning Training -----------------
@st.cache_resource
def train_model(data):
    feature_df = pd.get_dummies(
        data[["location", "hour", "weather", "traffic_index", "scheduled_frequency_min"]],
        drop_first=False
    )
    X = feature_df
    y = data["delay_minutes"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(X_train, y_train)
    
    score = round(model.score(X_test, y_test), 2)
    return model, X.columns.tolist(), score

model, feature_cols, r2_score = train_model(df)

# ----------------- 4. Header & Overview Metrics -----------------
st.title("📍 TransitPulse: Geo-Spatial Flow & Surge Predictor")
st.caption("Location-aware AI corridor delay forecasting and adaptive fleet rebalancing | INNOVATE Round 1")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Monitored Locations", len(LOCATION_PROFILES))
kpi2.metric("Telemetry Data Points", f"{len(df):,}")
kpi3.metric("Model Baseline R²", f"{r2_score}")
kpi4.metric("Network Avg Delay", f"{df['delay_minutes'].mean():.1f} min")

st.markdown("---")

# ----------------- 5. Location-Driven Sidebar Controls -----------------
st.sidebar.header("🎯 Location & Environment")

selected_loc = st.sidebar.selectbox("Select Target Location", list(LOCATION_PROFILES.keys()))
loc_meta = LOCATION_PROFILES[selected_loc]

st.sidebar.markdown(f"**Zone Type:** `{loc_meta['zone_type']}`")
st.sidebar.markdown(f"**Vulnerability Multiplier:** `{loc_meta['weather_sensitivity']}x`")

# Defaults automatically adapt to the chosen location
selected_hour = st.sidebar.slider(
    "Local Time (Hour of Day)", 0, 23, 
    value=loc_meta["peak_hours"][0]
)

weather_opts = ["Clear", "Rain", "Fog", "Heavy Storm"]
default_weather_idx = weather_opts.index(loc_meta["typical_weather"])
selected_weather = st.sidebar.selectbox("Local Weather Condition", weather_opts, index=default_weather_idx)

# Baseline traffic initialized according to peak status of the chosen location
is_peak = selected_hour in loc_meta["peak_hours"]
suggested_traffic = min(100, loc_meta["base_traffic"] + (20 if is_peak else -10))

selected_traffic = st.sidebar.slider(
    "Observed Traffic Density Index (0-100)", 10, 100, 
    value=suggested_traffic
)

selected_freq = st.sidebar.selectbox("Scheduled Dispatch Headway (mins)", [5, 10, 15, 20], index=1)

# Build inference sample
input_dict = {col: 0 for col in feature_cols}
if f"location_{selected_loc}" in input_dict:
    input_dict[f"location_{selected_loc}"] = 1
if f"weather_{selected_weather}" in input_dict:
    input_dict[f"weather_{selected_weather}"] = 1
input_dict["hour"] = selected_hour
input_dict["traffic_index"] = selected_traffic
input_dict["scheduled_frequency_min"] = selected_freq

input_df = pd.DataFrame([input_dict])
predicted_delay = float(model.predict(input_df)[0])

# ----------------- 6. Dashboard Main Panel -----------------
col_pred, col_map = st.columns([1, 1])

with col_pred:
    st.subheader(f"Forecast for {selected_loc}")
    
    if is_peak:
        st.caption("🔴 **Peak Rush Window Active** for this zone")
    else:
        st.caption("🟢 **Off-Peak Window Active** for this zone")
        
    if predicted_delay > 35:
        st.error(f"🚨 **High Delay Warning:** Estimated delay is **{predicted_delay:.1f} mins**")
        st.markdown("**Prescriptive AI Action:**")
        st.info("⚡ Automatically deploy 2 standby auxiliary buses and trigger signal preemption on key arterials.")
    elif predicted_delay > 20:
        st.warning(f"⚠️ **Moderate Congestion Warning:** Estimated delay is **{predicted_delay:.1f} mins**")
        st.markdown("**Prescriptive AI Action:**")
        st.info("⏱️ Adjust scheduled headway from current interval down by 3 minutes to clear platform buildup.")
    else:
        st.success(f"✅ **Normal Corridor Flow:** Estimated delay is **{predicted_delay:.1f} mins**")
        st.markdown("**Prescriptive AI Action:**")
        st.info("✔️ Maintain scheduled timetable. Fleet operations are running within optimal limits.")

with col_map:
    st.subheader("Regional Transit Map")
    map_data = []
    for name, p in LOCATION_PROFILES.items():
        loc_delays = df[df["location"] == name]["delay_minutes"].mean()
        map_data.append({
            "Location": name,
            "lat": p["lat"],
            "lon": p["lon"],
            "Avg Delay (min)": round(loc_delays, 1),
            "Zone Type": p["zone_type"]
        })
    map_df = pd.DataFrame(map_data)
    st.map(map_df, latitude="lat", longitude="lon", size=15)

st.markdown("---")

# ----------------- 7. Analytical Comparisons -----------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Average Delay by Location & Weather")
    loc_weather_agg = df.groupby(["location", "weather"])["delay_minutes"].mean().reset_index()
    fig1 = px.bar(
        loc_weather_agg,
        x="location",
        y="delay_minutes",
        color="weather",
        barmode="group",
        title="Weather Vulnerability per Location",
        labels={"delay_minutes": "Avg Delay (mins)", "location": "Corridor"}
    )
    fig1.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig1, use_container_width=True)

with col_chart2:
    st.subheader("Hourly Congestion Curve Across Locations")
    loc_hourly = df.groupby(["hour", "location"])["delay_minutes"].mean().reset_index()
    fig2 = px.line(
        loc_hourly,
        x="hour",
        y="delay_minutes",
        color="location",
        title="24-Hour Congestion Pattern Comparison",
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)