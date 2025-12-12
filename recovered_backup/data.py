import pandas as pd
import numpy as np
from pathlib import Path
import threading
import sys
import os

# Ensure root directory is in path to import analyze_real_data
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import analyze_real_data
except ImportError:
    analyze_real_data = None

# Color mapping for each category
CATEGORY_COLORS = {
    "Shower": "red",
    "Tap": "blue",
    "Toilet": "lime", # Using lime for bright green
    "Clothes Washer": "cyan",
    "Dishwasher": "yellow",
    "Bathtub": "magenta",
    "Irrigation": "black",
    "Evap Cooler": "gray",
    "Leak": "whitesmoke", # Using whitesmoke to be visible on white background
    "Other": "lightgray",
}

ALL_CATEGORIES = [
    "Shower", "Tap", "Clothes Washer", "Dishwasher", "Toilet",
    "Bathtub", "Irrigation", "Evap Cooler", "Leak", "Other"
]

def parse_raw_flow_data(raw_data_str, pulse=10):
    """Parse raw flow rate data from CSV string and convert for display."""
    if pd.isna(raw_data_str) or raw_data_str == "":
        return []
    try:
        flows = [float(x.strip()) for x in str(raw_data_str).split(",")]
        # Values in CSV are after /2 conversion, apply /6 for display
        flows = [round(f / 6.0, 3) for f in flows]
        return flows
    except:
        return []

def parse_datetime(row):
    """Parse date and time from CSV row."""
    try:
        date_str = row["Start date"]
        time_str = row["Start time"]
        dt = pd.to_datetime(f"{date_str} {time_str}", format="%d-%b-%y %H:%M:%S")
        return dt
    except:
        return None

# Global in-memory cache for processed DataFrames
DATA_CACHE = {}

def get_cached_data(path):
    """Retrieve data from cache if available."""
    return DATA_CACHE.get(str(path))

def update_cached_data(path, df):
    """Update the cache with new data."""
    DATA_CACHE[str(path)] = df

def load_classified_data(csv_path):
    """Load and prepare classified events from CSV."""
    print(f"Loading classified events from {csv_path}...")
    
    # Check cache first
    cached_df = get_cached_data(csv_path)
    if cached_df is not None:
        print("Returning cached data")
        return cached_df

    try:
        df = pd.read_csv(csv_path)
        
        # Parse datetimes and flow rates
        df["datetime_start"] = df.apply(parse_datetime, axis=1)
        df["flow_rates"] = df["Raw Series Data"].apply(parse_raw_flow_data)
        
        # Remove events without valid datetime
        df_events = df[df["datetime_start"].notna()].copy()
        df_events = df_events.sort_values("datetime_start").reset_index(drop=True)
        
        # Update cache
        update_cached_data(csv_path, df_events)
        
        print(f"Loaded {len(df_events)} events")
        return df_events
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def calculate_summary_stats(df_events, available_enduses=None):
    """Calculate end-use summary statistics."""
    if df_events.empty:
        return generate_empty_summary()
        
    if available_enduses is None:
        available_enduses = []

    # Create a copy and reclassify unavailable categories to "Other"
    df_filtered = df_events.copy()
    
    # If available_enduses is empty (initial load), assume all are available
    # or if specific list provided, filter
    if available_enduses:
        for idx, row in df_filtered.iterrows():
            if row["Category"] not in available_enduses and row["Category"] != "Other":
                df_filtered.at[idx, "Category"] = "Other"

    # Calculate end-use summary
    summary_data = df_filtered.groupby("Category")["Volume (L)"].sum().reset_index()
    summary_data.columns = ["Category", "Volume (L)"]
    
    total_volume = df_events["Volume (L)"].sum()

    if total_volume > 0:
        summary_data["Percentage (%)"] = (
            summary_data["Volume (L)"] / total_volume * 100
        ).round(1)
    else:
        summary_data["Percentage (%)"] = 0.0

    # Ensure all categories are included
    for cat in ALL_CATEGORIES:
        if cat not in summary_data["Category"].values:
            new_row = pd.DataFrame(
                [{"Category": cat, "Volume (L)": 0.0, "Percentage (%)": 0.0}]
            )
            summary_data = pd.concat([summary_data, new_row], ignore_index=True)

    # Sort by volume descending
    summary_data = summary_data.sort_values("Volume (L)", ascending=False).reset_index(drop=True)

    # Add Total row
    total_row = pd.DataFrame(
        [{"Category": "Total", "Volume (L)": total_volume, "Percentage (%)": 100.0}]
    )
    summary_data = pd.concat([summary_data, total_row], ignore_index=True)
    
    return summary_data

def generate_empty_summary():
    """Generate an empty summary table."""
    return pd.DataFrame({
        "Category": ALL_CATEGORIES + ["Total"],
        "Volume (L)": [0.0] * 11,
        "Percentage (%)": [0.0] * 10 + [100.0],
    })

def run_analysis_thread(csv_path):
    """Run analysis in a background thread."""
    if analyze_real_data is None:
        print("analyze_real_data module not found")
        return

    def run():
        try:
            analyze_real_data.analyze_csv_file(str(csv_path))
        except Exception as e:
            print(f"Analysis error: {e}")

    thread = threading.Thread(target=run)
    thread.start()
