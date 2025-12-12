import pandas as pd
import threading
from pathlib import Path

# Category colors - original palette
CATEGORY_COLORS = {
    "Shower": "red",
    "Tap": "blue",
    "Toilet": "lime",
    "Clothes Washer": "cyan",
    "Dishwasher": "yellow",
    "Bathtub": "magenta",
    "Irrigation": "black",
    "Evap Cooler": "gray",
    "Leak": "whitesmoke",
    "Other": "lightgray",
}

ALL_CATEGORIES = list(CATEGORY_COLORS.keys())

# Data cache
DATA_CACHE = {}
CACHE_LOCK = threading.Lock()


def get_cached_data(file_path):
    """Get cached DataFrame for a file path."""
    with CACHE_LOCK:
        return DATA_CACHE.get(file_path)


def update_cached_data(file_path, df):
    """Update the cache with new DataFrame."""
    with CACHE_LOCK:
        DATA_CACHE[file_path] = df.copy()


def invalidate_cache(file_path=None):
    """Invalidate cache for a specific file or all files."""
    with CACHE_LOCK:
        if file_path:
            DATA_CACHE.pop(file_path, None)
        else:
            DATA_CACHE.clear()


def parse_datetime(dt_str):
    """Parse datetime string to pandas datetime."""
    try:
        return pd.to_datetime(dt_str)
    except Exception:
        return None


def load_classified_data(file_path):
    """Load classified CSV data and return DataFrame."""
    try:
        df = pd.read_csv(file_path)

        # Parse datetime columns
        if "datetime_start" in df.columns:
            df["datetime_start"] = pd.to_datetime(df["datetime_start"])
        if "datetime_end" in df.columns:
            df["datetime_end"] = pd.to_datetime(df["datetime_end"])

        # Create display columns if they don't exist
        if "datetime_start" in df.columns:
            df["Start date"] = df["datetime_start"].dt.strftime("%d-%b-%y")
            df["Start time"] = df["datetime_start"].dt.strftime("%H:%M:%S")

        if "duration_seconds" in df.columns:
            df["Duration (h:m:s)"] = df["duration_seconds"].apply(
                lambda x: (
                    f"{int(x//3600)}:{int((x%3600)//60):02d}:{int(x%60):02d}"
                    if pd.notna(x)
                    else "0:00:00"
                )
            )

        if "volume" in df.columns:
            df["Volume (L)"] = df["volume"].round(2)

        if "max_flow_rate" in df.columns:
            df["Max flow (litre/min)"] = df["max_flow_rate"].round(2)

        # Update cache
        update_cached_data(str(file_path), df)

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def calculate_summary_stats(df, categories=None):
    """Calculate summary statistics for categories. Returns list of dicts for DataTable."""
    if df.empty:
        # Return empty summary with all categories
        summary_data = []
        for cat in ALL_CATEGORIES:
            summary_data.append({"Category": cat, "Volume (L)": 0, "Percentage (%)": 0})
        summary_data.append(
            {"Category": "Total", "Volume (L)": 0, "Percentage (%)": 100}
        )
        return summary_data  # Return list, not DataFrame

    # Filter by categories if specified
    if categories:
        df = df[df["Category"].isin(categories)]

    # Calculate volumes by category
    if "volume" in df.columns:
        volume_col = "volume"
    elif "Volume (L)" in df.columns:
        volume_col = "Volume (L)"
    else:
        volume_col = None

    summary_data = []
    total_volume = 0

    if volume_col:
        for cat in ALL_CATEGORIES:
            cat_volume = df[df["Category"] == cat][volume_col].sum()
            summary_data.append({"Category": cat, "Volume (L)": round(cat_volume, 2)})
            total_volume += cat_volume

        # Calculate percentages
        for item in summary_data:
            if total_volume > 0:
                item["Percentage (%)"] = round(
                    (item["Volume (L)"] / total_volume) * 100, 1
                )
            else:
                item["Percentage (%)"] = 0

        # Add total row
        summary_data.append(
            {
                "Category": "Total",
                "Volume (L)": round(total_volume, 2),
                "Percentage (%)": 100,
            }
        )
    else:
        for cat in ALL_CATEGORIES:
            summary_data.append({"Category": cat, "Volume (L)": 0, "Percentage (%)": 0})
        summary_data.append(
            {"Category": "Total", "Volume (L)": 0, "Percentage (%)": 100}
        )

    return summary_data  # Return list, not DataFrame


def run_analysis_thread(csv_path):
    """Run the analysis in a separate thread."""
    from src.main import run_analysis

    thread = threading.Thread(target=run_analysis, args=(csv_path,))
    thread.start()
