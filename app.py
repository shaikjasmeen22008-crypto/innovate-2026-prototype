import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="INNOVATE 2026 Prototype",
    page_icon="⚡",
    layout="wide"
)

# ----------------- 1. Synthetic Data Generator -----------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 600
    
    corridors = ["Sector A - Express", "Metro Corridor B", "Central Spine", "Industrial Belt", "Outer Ring"]
    weather_conds = ["Clear", "Rain", "Fog", "Heavy Storm"]
    
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-01", periods=n, freq="h"),
        "corridor": np.random.choice(corridors, n),
        "hour": np.random.randint(0, 24, n),
        "weather": np.random.choice(weather_conds, n, p=[0.6, 0.25, 0.1, 0.05]),
        "historical_traffic_index": np.random.uniform(20.0, 95.0, n),
        "scheduled_frequency_min": np.random.choice([5, 10, 15, 20], n),
    })
    
    # Generate realistic target: passenger delay index (minutes)
    weather_penalty = data["weather"].map({"Clear": 0, "Rain": 8, "Fog": 12, "Heavy Storm": 20})
    rush_hour_penalty = data["hour"].apply(lambda h: 15 if (7 <= h <= 10 or 17 <= h <= 20) else 2)
    noise = np.random.normal(0, 3, n)
    
    data["delay_minutes"] = (
        (data["historical_traffic_index"] * 0.3)
        + weather_penalty
        + rush_hour_penalty
        + noise
    ).clip(lower=1.0).round(1)
    
    return data

df = load_data()

# ----------------- 2. Quick ML Model Training -----------------
@st.cache_resource
def train_model(data):
    X = pd.get_dummies(data[["hour", "weather", "historical_traffic_index", "scheduled_frequency_min"]], drop_first=True)
    y = data["delay_minutes"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=60, random_state=42)
    model.fit(X_train, y_train)
    
    feature_columns = X.columns.tolist()
    test_score = round(model.score(X_test, y_test), 2)
    return model, feature_columns, test_score

model, feature_cols, r2_score = train_model(df)

# ----------------- 3. Header & Metrics -----------------
st.title("⚡ Dynamic Flow & Surge Predictor")
st.caption("AI-powered corridor latency forecast and resource rebalancing prototype | INNOVATE Round 1")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Analyzed Data Points", f"{len(df):,}")
kpi2.metric("Model Baseline R² Score", f"{r2_score}")
kpi3.metric("Avg Corridor Delay", f"{df['delay_minutes'].mean():.1f} min")
kpi4.metric("High Congestion Events", f"{int((df['delay_minutes'] > 30).sum())}")

st.markdown("---")

# ----------------- 4. Interactive Simulation Sidebar -----------------
st.sidebar.header("🕹️ Simulation Parameters")

sel_corridor = st.sidebar.selectbox("Select Target Corridor", df["corridor"].unique())
sel_hour = st.sidebar.slider("Hour of Day (24h format)", 0, 23, 18)
sel_weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rain", "Fog", "Heavy Storm"])
sel_traffic = st.sidebar.slider("Current Traffic Index (0-100)", 10, 100, 75)
sel_freq = st.sidebar.selectbox("Scheduled Frequency (mins)", [5, 10, 15, 20], index=1)

# Build inference vector
input_df = pd.DataFrame([{
    "hour": sel_hour,
    "weather": sel_weather,
    "historical_traffic_index": sel_traffic,
    "scheduled_frequency_min": sel_freq
}])
input_encoded = pd.get_dummies(input_df).reindex(columns=feature_cols, fill_value=0)

predicted_delay = float(model.predict(input_encoded)[0])

# ----------------- 5. Main Dashboard Body -----------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Simulated Prediction")
    
    status_color = "normal"
    if predicted_delay > 35:
        st.error(f"🚨 **High Delay Warning:** Estimated delay is **{predicted_delay:.1f} mins** on {sel_corridor}.")
    elif predicted_delay > 20:
        st.warning(f"⚠️ **Moderate Congestion:** Estimated delay is **{predicted_delay:.1f} mins** on {sel_corridor}.")
    else:
        st.success(f"✅ **Smooth Flow:** Estimated delay is **{predicted_delay:.1f} mins** on {sel_corridor}.")
    
    st.markdown("**Recommended AI Action:**")
    if predicted_delay > 35:
        st.info("Trigger automated re-routing: Deploy 2 standby auxiliary units to buffer the surge.")
    elif predicted_delay > 20:
        st.info("Adjust headway timing: Shorten dispatch interval by 3 minutes.")
    else:
        st.info("Maintain standard operating frequency; no fleet intervention required.")

    # Historical distribution chart
    fig_hist = px.histogram(
        df[df["corridor"] == sel_corridor],
        x="delay_minutes",
        nbins=25,
        title=f"Historical Delay Distribution ({sel_corridor})",
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("Hourly Delay Trend by Weather")
    hourly_trend = df.groupby(["hour", "weather"])["delay_minutes"].mean().reset_index()
    fig_trend = px.line(
        hourly_trend,
        x="hour",
        y="delay_minutes",
        color="weather",
        title="Impact of Weather & Time-of-Day on Delays",
        markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ----------------- 6. Raw Data Explorer -----------------
with st.expander("🔍 Inspect Historical Ingestion Feed"):
    st.dataframe(df.head(25), use_container_width=True)