import os
import pandas as pd
import requests
from datetime import datetime

def fetch_wa_solar_data(start_date="2024-01-01", end_date="2024-12-31"):
    """
    Fetches historical solar and cloud data for Perth from Open-Meteo.
    """
    print(f"🚀 Initializing data pull for Perth: {start_date} to {end_date}...")
    
    # Perth Coordinates
    lat, lon = -31.9522, 115.8614
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["direct_normal_irradiance", "cloud_cover", "temperature_2m"],
        "timezone": "Australia/Perth"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Parse into Pandas
        hourly_data = data['hourly']
        df = pd.DataFrame(hourly_data)
        df['time'] = pd.to_datetime(df['time'])
        
        # Define output path (D: Drive structure)
        output_path = r"D:\Saqib\SkillSetExpand\WA_Solar_Grid_Optimizer\data\raw/perth_solar_raw.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"✅ Success! Data saved to: {output_path}")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")

if __name__ == "__main__":
    fetch_wa_solar_data()