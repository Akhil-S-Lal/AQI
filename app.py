import streamlit as st
import pandas as pd
import requests
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time
from fetcher import fetch_realtime_data, SUPPORTED_CITIES

# ================================
# CONFIGURATIONS
# ================================
st.set_page_config(
    page_title="Real-Time AQI Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Cities and Data Fetching logic imported from fetcher.py to ensure consistency
# and use the latest fixes (e.g. unit conversion).

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

# Default values for imputation (25th percentile of training data)
# Using conservative values to avoid over-predicting AQI when data is missing
DEFAULT_VALUES = {
    'PM2.5': 28.16, 'PM10': 64.00, 'NO': 3.05, 'NO2': 13.10,
    'NOx': 11.35, 'NH3': 11.23, 'CO': 0.41, 'SO2': 4.25,
    'O3': 11.02, 'Benzene': 0.08, 'Toluene': 0.34, 'Xylene': 0.00
}

def get_aqi_bucket(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

def predict_aqi(data_dict):
    # Data is already imputed and smoothed in update_data
    df = pd.DataFrame([data_dict], columns=FEATURE_COLUMNS)
    
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
# fetch_realtime_data is imported from fetcher.py

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

# Store latest smoothed values for each city to enable Forward Fill and EMA
if "city_tracker" not in st.session_state:
    st.session_state["city_tracker"] = {}

def update_data(city):
    with st.spinner(f"Fetching data for {city}..."):
        raw_new = fetch_realtime_data(city)
        
        # Even if API fails (returns None), we might want to show stale data if we have it?
        # But fetch_realtime_data returns None on total failure.
        # Let's assume if it returns None, we can't do much update, but we could re-predict on old data?
        # For now, let's only update if we get *some* data (even partial).
        
        if raw_new:
            # Initialize city tracker if not present
            if city not in st.session_state["city_tracker"]:
                st.session_state["city_tracker"][city] = {}

            prev_data = st.session_state["city_tracker"][city]
            smoothed_data = {}
            alpha = 0.3 # Smoothing factor (0.3 = keep 70% old, take 30% new). Lower = smoother.

            for col in FEATURE_COLUMNS:
                new_val = raw_new.get(col)
                old_val = prev_data.get(col)

                final_val = 0

                # Case 1: New value exists
                if new_val is not None:
                    if old_val is not None:
                        # EMA Smoothing
                        final_val = (alpha * new_val) + ((1 - alpha) * old_val)
                    else:
                        # First value seen
                        final_val = new_val
                
                # Case 2: New value missing, but we have history (Forward Fill)
                elif old_val is not None:
                    final_val = old_val
                
                # Case 3: Cold Start (No new, no old) -> Use Conservative Default
                else:
                    final_val = DEFAULT_VALUES.get(col, 0)

                smoothed_data[col] = final_val
            
            # Update tracker with new smoothed state
            st.session_state["city_tracker"][city] = smoothed_data

            # Predict using SMOOTHED data
            pred = predict_aqi(smoothed_data)
            
            new_row = {
                "timestamp": datetime.now(),
                "City": city,
                **smoothed_data,
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
