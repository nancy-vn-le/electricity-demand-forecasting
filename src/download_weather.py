"""Download historical daily temperature data for Sydney from Open-Meteo.

Open-Meteo is a free weather API — no authentication or API key required.
Data is ERA5 reanalysis (ECMWF), the same source used by professional
meteorological services. Accuracy is within ~1-2°C of station observations.

Station reference: Sydney Observatory Hill (-33.87, 151.21)

Usage:
    python src/download_weather.py

Output:
    data/raw/sydney_temperature.csv
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "sydney_temperature.csv"

LATITUDE = -33.87
LONGITUDE = 151.21
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
TIMEZONE = "Australia/Sydney"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def download_temperature() -> pd.DataFrame:
    """Fetch daily temperature data from Open-Meteo historical archive."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean"],
        "timezone": TIMEZONE,
    }

    print(f"Fetching temperature data from Open-Meteo ...")
    print(f"  Location  : Sydney ({LATITUDE}, {LONGITUDE})")
    print(f"  Period    : {START_DATE} to {END_DATE}")

    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    daily = data["daily"]

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "temp_mean": daily["temperature_2m_mean"],
    })
    df = df.set_index("date")

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = download_temperature()

    df.to_csv(OUTPUT_FILE)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Saved       : {OUTPUT_FILE}")
    print(f"Rows        : {len(df):,} days")
    print(f"Date range  : {df.index.min().date()}  to  {df.index.max().date()}")
    print(f"Temp range  : {df['temp_max'].min():.1f}°C  to  {df['temp_max'].max():.1f}°C  (daily max)")
    print(f"Elapsed     : {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
