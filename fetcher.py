import requests
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import time
import json

# --------------------------
# CONFIGURATIONS
# --------------------------

# 1. Your FastAPI URL
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# 2. OpenAQ API Configuration
API_KEY = "e9810a7ae0097cc9d6313383143b9a00d3b104b4fd0ff8a115fe80669c9242ab"
HEADERS = {"X-API-Key": API_KEY}

# City to Location ID Mapping (Verified Active PM2.5 Locations)
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

DEFAULT_CITY = "Delhi - ITO"

# List of supported cities for the dashboard
SUPPORTED_CITIES = list(CITY_ID_MAPPING.keys())


# --------------------------
# Function: Fetch real-time pollutant values
# --------------------------
def fetch_realtime_data(city=DEFAULT_CITY):
    print(f"🔍 Fetching data for {city}...")
    try:
        location_id = CITY_ID_MAPPING.get(city)
        if not location_id:
            print(f"❌ No Location ID found for {city}")
            return None

        # V3 Strategy: 
        # The Detail endpoint is empty. The List endpoint works but filtering by ID is broken.
        # Solution: Fetch list of IN locations and find our ID in memory.
        
        url_loc = "https://api.openaq.org/v3/locations"
        # Fetch enough locations to likely include ours. 
        # We know our IDs are in the first few pages usually, but let's be safe.
        # ID 13 is early. ID 235 is early.
        params_loc = {"iso": "IN", "limit": 1000}
        
        resp_loc = requests.get(url_loc, headers=HEADERS, params=params_loc)
        if resp_loc.status_code != 200:
            print(f"❌ Failed to get locations list: {resp_loc.status_code}")
            return None
            
        results = resp_loc.json().get("results", [])
        
        # Find our location in the list
        target_loc = None
        for loc in results:
            if loc['id'] == location_id:
                target_loc = loc
                break
        
        if not target_loc:
            print(f"❌ Location ID {location_id} not found in first 1000 IN locations.")
            return None
            
        sensors = target_loc.get("sensors", [])
        print(f"ℹ️ Found {len(sensors)} sensors for location {location_id}")
        if not sensors:
            print(f"❌ No sensors found for {city} (ID: {location_id})")
            return None

        # Create mapping {pollutant: value}
        pollutant_map = {}
        
        target_pollutants = ["pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3", "benzene", "toluene", "xylene"]
        
        for sensor in sensors:
            param_name = sensor.get("parameter", {}).get("name", "").lower()
            param_unit = sensor.get("parameter", {}).get("units", "")

            if param_name in target_pollutants:
                sensor_id = sensor["id"]
                # Fetch measurement
                url_meas = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
                try:
                    resp_meas = requests.get(url_meas, headers=HEADERS, params={"limit": 1})
                    if resp_meas.status_code == 200:
                        meas_res = resp_meas.json().get("results", [])
                        if meas_res:
                            val = meas_res[0].get("value")
                            
                            # Unit Conversion for CO (µg/m³ -> mg/m³)
                            if param_name == "co" and param_unit == "µg/m³":
                                val = val / 1000.0
                                print(f"   ✅ {param_name}: {val} mg/m³ (converted from µg/m³)")
                            else:
                                print(f"   ✅ {param_name}: {val} {param_unit}")
                                
                            pollutant_map[param_name] = val
                        else:
                            print(f"   ⚠️ {param_name} (ID {sensor_id}): No measurements found")
                    else:
                        print(f"   ⚠️ {param_name} (ID {sensor_id}): API Error {resp_meas.status_code}")
                except Exception as e:
                    print(f"   ⚠️ Failed to fetch {param_name}: {e}")

        if not pollutant_map:
            print(f"❌ No measurements retrieved for {city}")
            return None

        # Model expects these pollutant columns (map V3 names to model names)
        model_input = {
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

        print("📥 Real-Time Data Fetched:", model_input)
        return model_input

    except Exception as e:
        print("❌ Error fetching OpenAQ data:", str(e))
        return None


# --------------------------
# Function: Send data → FastAPI → Get prediction
# --------------------------
def send_to_fastapi(payload):
    try:
        response = requests.post(FASTAPI_URL, json=convert_payload_for_api(payload))
        prediction = response.json()
        print("📤 Prediction Received:", prediction)
        return prediction
    except Exception as e:
        print("❌ Error calling FastAPI:", str(e))
        return None


# Convert model input to API JSON schema
def convert_payload_for_api(data):
    return {
        "PM2_5": data["PM2.5"],
        "PM10": data["PM10"],
        "NO": data["NO"],
        "NO2": data["NO2"],
        "NOx": data["NOx"],
        "NH3": data["NH3"],
        "CO": data["CO"],
        "SO2": data["SO2"],
        "O3": data["O3"],
        "Benzene": data["Benzene"],
        "Toluene": data["Toluene"],
        "Xylene": data["Xylene"],
    }


# --------------------------
# Store results in CSV
# --------------------------
def store_result(raw_data, prediction, city=DEFAULT_CITY):
    entry = {
        "timestamp": datetime.now(),
        "City": city,
        **raw_data,
        "Predicted_AQI": prediction["prediction"]["AQI_Prediction"],
        "Predicted_Bucket": prediction["prediction"]["AQI_Bucket"]
    }

    df = pd.DataFrame([entry])
    df.to_csv("realtime_predictions.csv", mode='a', header=not pd.io.common.file_exists("realtime_predictions.csv"), index=False)
    print("💾 Stored in realtime_predictions.csv")


# --------------------------
# MAIN TASK: Full Pipeline
# --------------------------
def run_realtime_task(city=DEFAULT_CITY):
    print("\n==============================")
    print(f"⏳ Fetching New Real-Time Data for {city}")
    print("==============================")

    raw_data = fetch_realtime_data(city)
    if raw_data is None:
        return

    prediction = send_to_fastapi(raw_data)
    if prediction is None:
        return

    store_result(raw_data, prediction, city)


# --------------------------
# APScheduler — Run every X minutes
# --------------------------
def start_scheduler(interval_minutes=5):
    scheduler = BlockingScheduler()
    scheduler.add_job(run_realtime_task, 'interval', minutes=interval_minutes)
    print(f"⏱ Scheduler started — Fetching every {interval_minutes} minutes...")
    
    # Run once immediately for testing
    run_realtime_task()
    
    scheduler.start()


if __name__ == "__main__":
    start_scheduler(interval_minutes=2)  # change frequency here
