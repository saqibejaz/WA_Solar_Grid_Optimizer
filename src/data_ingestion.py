import os

import pandas as pd
import requests


def fetch_wa_solar_data(start_date="2024-01-01", end_date="2024-12-31"):
    """Fetch historical solar and cloud data for Perth from Open-Meteo."""
    print(f"🚀 Initializing data pull for Perth: {start_date} to {end_date}...")

    lat, lon = -31.9522, 115.8614

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["direct_normal_irradiance", "cloud_cover", "temperature_2m"],
        "timezone": "Australia/Perth",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])

        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(base_dir, "data", "raw", "perth_solar_raw.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"✅ Data saved to: {output_path}")
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(df.head())
        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


if __name__ == "__main__":
    fetch_wa_solar_data()
