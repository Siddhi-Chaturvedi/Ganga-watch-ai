"""
=============================================================================
data_loader.py
Fetch real Ganga river water quality data from data.gov.in API
=============================================================================
"""

import requests
import pandas as pd
import numpy as np
import yaml
import os

# ── Load config ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

API_KEY     = config["api"]["key"]
RESOURCE_ID = config["api"]["resource_id"]
BASE_URL    = config["api"]["base_url"]
RAW_PATH    = config["paths"]["raw_data"]


# ── Fetch data from data.gov.in ──────────────────────────────────────────────
def fetch_data(limit=5000, state_filter=None):
    """
    Fetch Ganga river water quality data from data.gov.in API.

    Parameters
    ----------
    limit        : max number of records to fetch
    state_filter : e.g. 'Uttar Pradesh', 'Bihar', 'West Bengal'

    Returns
    -------
    df : raw pandas DataFrame
    """
    print(f"[data_loader] Fetching data from data.gov.in ...")

    params = {
        "api-key" : API_KEY,
        "format"  : "json",
        "limit"   : limit,
    }
    if state_filter:
        params["filters[State]"] = state_filter

    url = f"{BASE_URL}/{RESOURCE_ID}"

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"[data_loader] ERROR: {e}")
        print("[data_loader] Falling back to synthetic data...")
        return generate_synthetic_data()

    records = data.get("records", [])
    if not records:
        print("[data_loader] No records returned. Falling back to synthetic data...")
        return generate_synthetic_data()

    df = pd.DataFrame(records)
    print(f"[data_loader] Fetched {len(df)} records.")
    return df


# ── Column standardisation ───────────────────────────────────────────────────
COLUMN_MAP = {
    # common data.gov.in column names → our standard names
    "do"              : "DO",
    "dissolved_oxygen": "DO",
    "ph"              : "pH",
    "bod"             : "BOD",
    "biochemical_oxygen_demand": "BOD",
    "turbidity"       : "Turbidity",
    "temperature"     : "Temperature",
    "conductivity"    : "Conductivity",
    "wqi"             : "WQI",
    "water_quality_index": "WQI",
    "year"            : "Year",
    "date"            : "Date",
    "state"           : "State",
    "station"         : "Station",
    "location"        : "Location",
}

REQUIRED_COLS = ["DO", "pH", "BOD", "Turbidity", "Temperature", "Conductivity", "WQI"]


def clean_and_standardise(df):
    """
    Rename columns, parse dates, handle missing values,
    and return a clean time-indexed DataFrame.
    """
    # Lowercase all column names first
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to standard names
    df.rename(columns=COLUMN_MAP, inplace=True)

    print(f"[data_loader] Columns after rename: {list(df.columns)}")

    # ── Parse date ──────────────────────────────────────────────────────────
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Year" in df.columns:
        # If only year is available, create a date
        df["Date"] = pd.to_datetime(df["Year"].astype(str), format="%Y")
    else:
        # No date column — create a sequential index
        df["Date"] = pd.date_range(start="2015-01-01", periods=len(df), freq="D")

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # ── Keep only numeric water quality columns ──────────────────────────────
    for col in REQUIRED_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fill missing WQI if not in dataset ──────────────────────────────────
    if "WQI" not in df.columns:
        print("[data_loader] WQI column not found. Computing from parameters...")
        df["WQI"] = compute_wqi(df)

    # ── Drop rows where ALL required columns are NaN ─────────────────────────
    df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns], how="all", inplace=True)

    # ── Forward fill then backward fill remaining NaNs ───────────────────────
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # ── Keep only required columns ───────────────────────────────────────────
    available = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[available]

    print(f"[data_loader] Clean dataset shape: {df.shape}")
    return df


# ── Compute WQI if not provided ──────────────────────────────────────────────
def compute_wqi(df):
    """Simple weighted WQI formula based on available parameters."""
    wqi = pd.Series(index=df.index, dtype=float)
    score = 0
    weights = 0

    if "DO" in df.columns:
        score  += 3.0 * df["DO"].clip(0, 14) / 14 * 100
        weights += 3.0
    if "pH" in df.columns:
        score  += 2.0 * (1 - abs(df["pH"] - 7.5) / 7.5) * 100
        weights += 2.0
    if "BOD" in df.columns:
        score  += 2.0 * (1 - df["BOD"].clip(0, 30) / 30) * 100
        weights += 2.0
    if "Turbidity" in df.columns:
        score  += 1.0 * (1 - df["Turbidity"].clip(0, 200) / 200) * 100
        weights += 1.0
    if "Temperature" in df.columns:
        score  += 1.0 * (1 - abs(df["Temperature"] - 25) / 25).clip(0, 1) * 100
        weights += 1.0

    wqi = (score / weights).clip(0, 100) if weights > 0 else pd.Series(50, index=df.index)
    return wqi


# ── Fallback: synthetic data ─────────────────────────────────────────────────
def generate_synthetic_data(n_days=1825):
    """Generate realistic synthetic Ganga water quality data as fallback."""
    print("[data_loader] Generating synthetic Ganga data (5 years daily)...")
    np.random.seed(42)
    t = np.arange(n_days)

    DO           = 7.5  + 1.5*np.sin(2*np.pi*t/365) + 0.5*np.sin(2*np.pi*t/30) - 0.001*t + np.random.normal(0, 0.3, n_days)
    pH           = 7.8  + 0.3*np.sin(2*np.pi*t/365+1) + 0.1*np.sin(2*np.pi*t/14) + np.random.normal(0, 0.1, n_days)
    BOD          = 3.5  + 1.2*np.sin(2*np.pi*t/365+3) + 0.3*t/n_days + np.abs(np.random.normal(0, 0.4, n_days))
    Turbidity    = 20   + 10*np.sin(2*np.pi*t/365+2) + 5*np.random.exponential(1, n_days)
    Temperature  = 22   + 8*np.sin(2*np.pi*t/365-1.5) + np.random.normal(0, 0.5, n_days)
    Conductivity = 400  + 50*np.sin(2*np.pi*t/365) + np.random.normal(0, 15, n_days)
    WQI = (60 - 2.5*BOD + 3.0*DO + 1.5*(pH - 7) - 0.05*Turbidity + np.random.normal(0, 1.5, n_days)).clip(0, 100)

    dates = pd.date_range(start="2019-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "DO": DO, "pH": pH, "BOD": BOD, "Turbidity": Turbidity,
        "Temperature": Temperature, "Conductivity": Conductivity, "WQI": WQI
    }, index=dates)
    df.index.name = "Date"
    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────
def load_data(use_api=True, state_filter="Uttar Pradesh"):
    """
    Full data loading pipeline.
    Set use_api=False to use synthetic data directly.
    """
    if use_api:
        raw_df  = fetch_data(limit=5000, state_filter=state_filter)
        clean_df = clean_and_standardise(raw_df)
    else:
        clean_df = generate_synthetic_data()

    # Save processed data
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    clean_df.to_csv(RAW_PATH)
    print(f"[data_loader] Data saved to {RAW_PATH}")
    return clean_df


if __name__ == "__main__":
    df = load_data(use_api=True)
    print(df.head(10))
    print(df.describe().round(2))