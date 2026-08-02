import pandas as pd
import os

DATA_RAW_DIR = "./data_raw"
OUTPUT_DIR = "./data_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIONS = ["kim_long", "phu_oc"]
ROLLING_WINDOWS = [3, 5, 7]


def load_and_clean(station_id):
    path = os.path.join(DATA_RAW_DIR, f"{station_id}_raw.csv")
    df = pd.read_csv(path)
    
    # Ép Pandas nhận diện đúng định dạng ngày/tháng/năm
    df["time"] = pd.to_datetime(df["time"], format="mixed", dayfirst=True)
    df = df.sort_values("time").reset_index(drop=True)

    df["precipitation_sum"] = df["precipitation_sum"].interpolate(method="linear")

    for w in ROLLING_WINDOWS:
        df[f"rain_cum_{w}d"] = df["precipitation_sum"].rolling(window=w, min_periods=1).sum()

    df["station"] = station_id
    return df


def load_flood_events():
    path = "flood_events.xlsx"
    if not os.path.exists(path):
        raise FileNotFoundError("Khong tim thay flood_events.xlsx")
    
    # Đọc file Excel và parse ngày tháng an toàn
    df = pd.read_excel(path)
    df["start_date"] = pd.to_datetime(df["start_date"], format="mixed", dayfirst=True)
    df["end_date"] = pd.to_datetime(df["end_date"], format="mixed", dayfirst=True)
    return df


def label_flood_days(df, events):
    df["label"] = 0
    for _, row in events.iterrows():
        mask = (df["time"] >= row["start_date"]) & (df["time"] <= row["end_date"])
        df.loc[mask, "label"] = 1
    return df


def main():
    events = load_flood_events()

    all_data = []
    for station_id in STATIONS:
        df = load_and_clean(station_id)
        df = label_flood_days(df, events)
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    # Chia theo mốc thời gian
    train_df = full_df[full_df["time"] < "2024-01-01"]
    test_df = full_df[full_df["time"] >= "2024-01-01"]

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, encoding="utf-8-sig")

    print(f"Train: {len(train_df)} dòng | Test: {len(test_df)} dòng")
    print(f"Tỷ lệ nhãn 1: {full_df['label'].mean():.2%}")
    print("Đã lưu vào ./data_clean/")


if __name__ == "__main__":
    main()