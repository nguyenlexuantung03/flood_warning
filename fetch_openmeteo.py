import requests
import pandas as pd
import time
import os

STATIONS = {
    "kim_long": {"lat": 16.4546832, "lon": 107.5603364, "name": "Kim Long - Song Huong"},
    "phu_oc":   {"lat": 16.5278132, "lon": 107.4724328, "name": "Phu Oc - Song Bo"},
}

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUTPUT_DIR = "./data_raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_station_data(station_id, lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": "precipitation_sum,rain_sum,precipitation_hours",
        "timezone": "Asia/Bangkok",
    }
    print(f"Dang lay du lieu tram {station_id}...")
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["daily"])
    df["station"] = station_id
    return df


def main():
    for station_id, info in STATIONS.items():
        df = fetch_station_data(station_id, info["lat"], info["lon"])
        out_path = os.path.join(OUTPUT_DIR, f"{station_id}_raw.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Da luu {out_path} ({len(df)} dong)")
        time.sleep(1)

    print("Xong. Chay preprocess.py de tiep tuc.")


if __name__ == "__main__":
    main()
