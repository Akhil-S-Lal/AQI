import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time

# ================================
# CONFIGURATIONS
# ================================
st.set_page_config(
    page_title="Real-Time AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# OpenAQ API Key
API_KEY = "e9810a7ae0097cc9d6313383143b9a00d3b104b4fd0ff8a115fe80669c9242ab"
HEADERS = {"X-API-Key": API_KEY}

# City Mapping
CITY_ID_MAPPING = {
  "Delhi - ITO": 103,
  "Delhi - DTU": 13,
  "Delhi - R K Puram": 17,
  "Delhi - Punjabi Bagh": 50,
  "Delhi - Anand Vihar": 235,
  "Gurugram - Vikas Sadan": 301,
  "Hyderabad - Zoo Park": 407,
  "Hyderabad - Sanathnagar": 408,
  "Noida - Sector 125": 5598,
  "Noida - Sector 62": 5616,
  "Pune - Karve Road": 5661,
  "Ahmedabad - Maninagar": 5631,
  "Lucknow - Lalbagh": 2456,
  "Bengaluru - Peenya": 5607,
  "Bengaluru - Silk Board": 6975,
  "Bengaluru - Hombegowda Nagar": 6983,
  "Bengaluru - Hebbal": 6980,
  "Bengaluru - BTM Layout": 412,
  "Chennai - Manali": 2461,
  "Kolkata - Jadavpur": 716,
  "Jaipur - Shastri Nagar": 5612
}

SUPPORTED_CITIES = list(CITY_ID_MAPPING.keys())

# ================================
# MODEL LOADING
# ================================
@st.cache_resource
def load_model():
    model = joblib.load("aqi_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

FEATURE_COLUMNS = [
    'PM2.5','PM10','NO','NO2','NOx',
    'NH3','CO','SO2','O3','Benzene','Toluene','Xylene'
]

def get_aqi_bucket(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def predict_aqi(data_dict):
    # Preprocess
    missing = [col for col in FEATURE_COLUMNS if col not in data_dict]
    if missing:
        # Fill missing with median/default if needed for robustness in demo
        for m in missing:
            data_dict[m] = 0 # Or some default
            
    df = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)
    df = df.replace({None: np.nan})
    # Simple fillna for demo robustness
    df = df.fillna(0) 
    
    scaled = scaler.transform(df)
    aqi_pred = model.predict(scaled)[0]
    bucket = get_aqi_bucket(aqi_pred)
    
    return {
        "AQI_Prediction": float(round(aqi_pred, 2)),
        "AQI_Bucket": bucket
    }

# ================================
# DATA FETCHING
# ================================
def fetch_realtime_data(city):
    location_id = CITY_ID_MAPPING.get(city)
    if not location_id: return None

    # V3 Strategy: List endpoint -> Find ID -> Get Sensors -> Get Measurements
    url_loc = "https://api.openaq.org/v3/locations"
    params_loc = {"iso": "IN", "limit": 1000}
    
    try:
        resp_loc = requests.get(url_loc, headers=HEADERS, params=params_loc)
        if resp_loc.status_code != 200: return None
        
        results = resp_loc.json().get("results", [])
        target_loc = next((loc for loc in results if loc['id'] == location_id), None)
        
        if not target_loc: return None
        sensors = target_loc.get("sensors", [])
        if not sensors: return None

        pollutant_map = {}
        target_pollutants = ["pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3", "benzene", "toluene", "xylene"]
        
        for sensor in sensors:
            param_name = sensor.get("parameter", {}).get("name", "").lower()
            if param_name in target_pollutants:
                sensor_id = sensor["id"]
                try:
                    url_meas = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
                    resp_meas = requests.get(url_meas, headers=HEADERS, params={"limit": 1})
                    if resp_meas.status_code == 200:
                        meas_res = resp_meas.json().get("results", [])
                        if meas_res:
                            pollutant_map[param_name] = meas_res[0].get("value")
                except:
                    pass

        if not pollutant_map: return None

        return {
            "PM2.5": pollutant_map.get("pm25"),
            "PM10": pollutant_map.get("pm10"),
            "NO": pollutant_map.get("no"),
            "NO2": pollutant_map.get("no2"),
            "NOx": pollutant_map.get("nox"),
            "NH3": pollutant_map.get("nh3"),
            "CO": pollutant_map.get("co"),
            "SO2": pollutant_map.get("so2"),
            "O3": pollutant_map.get("o3"),
            "Benzene": pollutant_map.get("benzene"),
            "Toluene": pollutant_map.get("toluene"),
            "Xylene": pollutant_map.get("xylene"),
        }
    except:
        return None

# ================================
# DASHBOARD UI
# ================================
st.title("🌫️ Real-Time AQI Monitoring Dashboard")
st.write("Live predictions from ML model using real-time pollutant readings (OpenAQ).")

# Sidebar
st.sidebar.header("Configuration")
selected_city = st.sidebar.selectbox("Select City", SUPPORTED_CITIES)
auto_fetch = st.sidebar.checkbox("Enable Active Auto-Fetch", value=True)

# Session State for Data Persistence
if "data_history" not in st.session_state:
    st.session_state["data_history"] = pd.DataFrame(columns=["timestamp", "City", "Predicted_AQI", "Predicted_Bucket"] + FEATURE_COLUMNS)

def update_data(city):
    with st.spinner(f"Fetching data for {city}..."):
        raw = fetch_realtime_data(city)
        if raw:
            pred = predict_aqi(raw)
            
            new_row = {
                "timestamp": datetime.now(),
                "City": city,
                **raw,
                "Predicted_AQI": pred["AQI_Prediction"],
                "Predicted_Bucket": pred["AQI_Bucket"]
            }
            
            st.session_state["data_history"] = pd.concat([
                st.session_state["data_history"], 
                pd.DataFrame([new_row])
            ], ignore_index=True)
            return True
    return False

# Auto-fetch on selection change
if "last_selected_city" not in st.session_state:
    st.session_state["last_selected_city"] = selected_city

if selected_city != st.session_state["last_selected_city"]:
    st.session_state["last_selected_city"] = selected_city
    update_data(selected_city)

if st.sidebar.button("🔄 Fetch Latest Data Now"):
    if update_data(selected_city):
        st.success("Data updated!")
    else:
        st.error("Could not fetch data.")

# Display Logic
df = st.session_state["data_history"]
city_df = df[df["City"] == selected_city]

if not city_df.empty:
    latest = city_df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted AQI", value=round(latest["Predicted_AQI"], 2))
    col2.metric("Air Quality Category", value=latest["Predicted_Bucket"])
    col3.metric("Last Updated", value=str(latest["timestamp"]))

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=latest["Predicted_AQI"],
        title={'text': "AQI Level"},
        gauge={
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 100], 'color': "yellow"},
                {'range': [100, 200], 'color': "orange"},
                {'range': [200, 300], 'color': "red"},
                {'range': [300, 400], 'color': "purple"},
                {'range': [400, 500], 'color': "maroon"},
            ],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Charts
    st.subheader(f"📈 AQI Trend ({selected_city})")
    st.plotly_chart(px.line(city_df, x="timestamp", y="Predicted_AQI", markers=True), use_container_width=True)

    st.subheader(f"🧪 Pollutant Trends ({selected_city})")
    # Ensure numeric types for plotting
    plot_df = city_df.copy()
    for col in FEATURE_COLUMNS:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    
    st.plotly_chart(px.line(plot_df, x="timestamp", y=FEATURE_COLUMNS), use_container_width=True)
    
    st.subheader("📋 Latest Readings")
    st.dataframe(city_df.tail(1))

else:
    st.info("No data yet. Fetching...")
    # Try initial fetch if empty
    if update_data(selected_city):
        st.rerun()

# Auto Refresh
refresh_interval = 20
st.sidebar.write(f"Auto-refresh every {refresh_interval}s.")
st_autorefresh = st.sidebar.empty()

st_autorefresh.write(f"⏳ Auto-refresh in {refresh_interval} seconds…")
progress_bar = st.sidebar.progress(0)

for i in range(refresh_interval):
    remaining = refresh_interval - i
    st_autorefresh.write(f"⏳ Refreshing in {remaining}s...")
    progress_bar.progress((i + 1) / refresh_interval)
    time.sleep(1)

if auto_fetch:
    update_data(selected_city)

st.rerun()
