"""
Data preparation script for ML training.
Processes labeled end-use data and computes features for the CNN-LSTM model.

Usage:
    python prepare_ml_data.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import mode
from glob import glob
from typing import List, Tuple, Dict
from tqdm import tqdm


# Feature columns
FEATURE_COLS = [
    "Duration_seconds",
    "Volume",
    "Max flow",
    "Mode flow",
    "raw_series_length",
    "flow_std",
    "num_peaks",
    "hour_of_day",
    "rise_time",
    "fall_time",
    "plateau_ratio",
    "flow_slope",
]


def compute_additional_features(raw_series: np.ndarray, start_hour: int = 12) -> dict:
    """
    Compute additional shape-based features from raw flow series.
    """
    if len(raw_series) == 0:
        return {
            "raw_series_length": 0,
            "flow_std": 0.0,
            "num_peaks": 0,
            "hour_of_day": start_hour,
            "rise_time": 0,
            "fall_time": 0,
            "plateau_ratio": 0.0,
            "flow_slope": 0.0,
        }

    raw_series_length = len(raw_series)
    flow_std = float(np.std(raw_series))

    try:
        peaks, _ = find_peaks(raw_series, height=np.mean(raw_series) * 0.5)
        num_peaks = len(peaks)
    except:
        num_peaks = 0

    hour_of_day = start_hour
    max_idx = np.argmax(raw_series)
    rise_time = max_idx
    fall_time = len(raw_series) - max_idx - 1

    max_flow = np.max(raw_series)
    if max_flow > 0:
        threshold = max_flow * 0.8
        plateau_count = np.sum(raw_series >= threshold)
        plateau_ratio = plateau_count / len(raw_series)
    else:
        plateau_ratio = 0.0

    if len(raw_series) > 1:
        x = np.arange(len(raw_series))
        try:
            slope = np.polyfit(x, raw_series, 1)[0]
            flow_slope = float(slope)
        except:
            flow_slope = 0.0
    else:
        flow_slope = 0.0

    return {
        "raw_series_length": raw_series_length,
        "flow_std": flow_std,
        "num_peaks": num_peaks,
        "hour_of_day": hour_of_day,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "plateau_ratio": plateau_ratio,
        "flow_slope": flow_slope,
    }


def process_labeled_file(
    file_path: str,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, str]:
    """
    Process a single labeled end-use file.

    Returns:
        raw_series_list: List of flow rate time series
        features: Feature array (n_events, n_features)
        labels: Label array (n_events,)
        property_id: Property identifier
    """
    # Extract property ID from filename
    property_id = os.path.basename(file_path).replace(".csv", "")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return [], np.array([]), np.array([]), property_id

    # Required columns
    required_cols = [
        "Combined_Label",
        "Start",
        "Duration_seconds",
        "Volume",
        "Max flow",
        "Mode flow",
    ]

    for col in required_cols:
        if col not in df.columns:
            print(f"  Missing column {col} in {file_path}")
            return [], np.array([]), np.array([]), property_id

    # Parse start time for hour_of_day
    try:
        df["Start"] = pd.to_datetime(df["Start"])
        df["hour_of_day"] = df["Start"].dt.hour
    except:
        df["hour_of_day"] = 12

    raw_series_list = []
    features_list = []
    labels_list = []

    for idx, row in df.iterrows():
        label = row["Combined_Label"]

        # Skip unlabeled or invalid
        if pd.isna(label) or label in ["Unknown", ""]:
            continue

        # Get raw series columns (if available)
        raw_cols = [c for c in df.columns if c.startswith("flow_") or c.isdigit()]

        if len(raw_cols) > 0:
            raw_series = row[raw_cols].values.astype(float)
            raw_series = raw_series[~np.isnan(raw_series)]
        else:
            # Create synthetic series from duration and volume
            duration = row["Duration_seconds"]
            volume = row["Volume"]
            n_points = max(1, int(duration / 10))
            avg_flow = volume / n_points if n_points > 0 else 0
            raw_series = np.full(n_points, avg_flow)

        if len(raw_series) == 0:
            continue

        # Compute additional features
        hour = row.get("hour_of_day", 12)
        additional = compute_additional_features(raw_series, int(hour))

        # Build feature vector
        features = np.array(
            [
                row["Duration_seconds"],
                row["Volume"],
                row["Max flow"],
                row["Mode flow"],
                additional["raw_series_length"],
                additional["flow_std"],
                additional["num_peaks"],
                additional["hour_of_day"],
                additional["rise_time"],
                additional["fall_time"],
                additional["plateau_ratio"],
                additional["flow_slope"],
            ],
            dtype=np.float32,
        )

        raw_series_list.append(raw_series.astype(np.float32))
        features_list.append(features)
        labels_list.append(label)

    if len(features_list) == 0:
        return [], np.array([]), np.array([]), property_id

    return raw_series_list, np.array(features_list), np.array(labels_list), property_id


def main():
    """Main data preparation function."""

    # Configuration
    DATA_DIRS = [
        r"D:\ALL END USE DATA\SYDNEY WATER DATA\TRANCHE 1\Tranche 1 - Classified",
        r"D:\ALL END USE DATA\SYDNEY WATER DATA\TRANCHE 2\Tranche 2 - Classified",
        r"D:\ALL END USE DATA\MELBOURNE WATER DATA\Melbourne - Classified",
    ]
    OUTPUT_PATH = "prepared_ml_data_12features.pkl"

    print("=" * 60)
    print("ML DATA PREPARATION")
    print("=" * 60)
    print(f"Output: {OUTPUT_PATH}")
    print(f"Features: {FEATURE_COLS}")
    print()

    # Collect all labeled files
    all_files = []
    for data_dir in DATA_DIRS:
        if os.path.exists(data_dir):
            files = glob(os.path.join(data_dir, "*.csv"))
            all_files.extend(files)
            print(f"  Found {len(files)} files in {data_dir}")
        else:
            print(f"  Directory not found: {data_dir}")

    print(f"\nTotal files: {len(all_files)}")

    if len(all_files) == 0:
        print("No files found!")
        return

    # Process all files
    all_raw_series = []
    all_features = []
    all_labels = []
    all_property_ids = []

    print("\nProcessing files...")

    for file_path in tqdm(all_files):
        raw_series, features, labels, property_id = process_labeled_file(file_path)

        if len(labels) > 0:
            all_raw_series.extend(raw_series)
            all_features.append(features)
            all_labels.extend(labels)
            all_property_ids.extend([property_id] * len(labels))

    # Combine features
    all_features = np.vstack(all_features) if all_features else np.array([])
    all_labels = np.array(all_labels)
    all_property_ids = np.array(all_property_ids)

    print(f"\nTotal events: {len(all_labels)}")
    print(f"Total properties: {len(np.unique(all_property_ids))}")

    # Print class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for label, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({100*count/len(all_labels):.1f}%)")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")

    data = {
        "raw_series": all_raw_series,
        "features": all_features,
        "labels": all_labels,
        "property_ids": all_property_ids,
        "feature_cols": FEATURE_COLS,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)

    print("Done!")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
