"""
Test script for complete pipeline: data loading -> event segmentation -> classification.

Usage:
    python test_complete_pipeline.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ml.inference import CNNLSTMInference


def read_raw_data(file_path: str) -> pd.DataFrame:
    """Read raw meter data from CSV."""
    print(f"Reading: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"  Total volume: {df['Usage'].sum():.1f} L")

    return df


def segment_events(
    df: pd.DataFrame, gap_threshold: int = 2
) -> Tuple[List, pd.DataFrame]:
    """Segment raw data into discrete events."""
    from scipy.stats import mode

    usage = df["Usage"].values
    datetime_col = df["Datetime"]

    events = []
    features_list = []
    start_times_list = []

    n = len(usage)
    i = 0

    while i < n:
        if usage[i] == 0:
            i += 1
            continue

        start_idx = i
        j = i
        zero_count = 0

        while j < n:
            if usage[j] == 0:
                zero_count += 1
                if zero_count >= gap_threshold:
                    break
            else:
                zero_count = 0
            j += 1

        end_idx = j - zero_count if zero_count > 0 else j

        if end_idx > start_idx:
            event_usage = usage[start_idx:end_idx]

            volume = np.sum(event_usage)
            duration = (end_idx - start_idx) * 10
            max_flow = np.max(event_usage) * 6
            mode_result = mode(event_usage[event_usage > 0], keepdims=True)
            mode_flow = (mode_result.mode[0] if len(mode_result.mode) > 0 else 0) * 6

            events.append(event_usage)

            start_dt = datetime_col.iloc[start_idx]
            features_list.append(
                {
                    "Volume": volume,
                    "Duration": duration,
                    "Max_flow": max_flow,
                    "Mode_flow": mode_flow,
                    "start_time": start_dt,
                }
            )

        i = j

    features_df = pd.DataFrame(features_list)

    print(f"  Events: {len(events)}")

    return events, features_df


def classify_events(
    events: List[np.ndarray],
    features: pd.DataFrame,
    model: CNNLSTMInference,
) -> pd.DataFrame:
    """Classify all events using CNN-LSTM model."""

    results = []

    for i, (event, row) in enumerate(zip(events, features.itertuples())):
        start_hour = row.start_time.hour if hasattr(row, "start_time") else 12

        probs = model.predict(
            raw_series=event,
            duration=row.Duration,
            volume=row.Volume,
            max_flow=row.Max_flow,
            mode_flow=row.Mode_flow,
            start_hour=start_hour,
        )

        # Get predicted class
        pred_class = max(probs, key=probs.get)
        pred_prob = probs[pred_class]

        results.append(
            {
                "event_idx": i,
                "predicted_class": pred_class,
                "confidence": pred_prob,
                "Volume": row.Volume,
                "Duration": row.Duration,
                "Max_flow": row.Max_flow,
                **probs,
            }
        )

    return pd.DataFrame(results)


def main():
    """Main test function."""

    print("=" * 70)
    print("COMPLETE PIPELINE TEST")
    print("=" * 70)

    # Configuration
    RAW_DATA_PATH = r"D:\ALL END USE DATA\SYDNEY WATER DATA\TRANCHE 1\Tranche 1 - Raw data\BTJB0601.csv"
    CHECKPOINT_PATH = "checkpoints/best_model.pth"

    # Check files exist
    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Data file not found: {RAW_DATA_PATH}")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Model checkpoint not found: {CHECKPOINT_PATH}")
        return

    # Step 1: Load data
    print("\n" + "-" * 40)
    print("STEP 1: LOAD DATA")
    print("-" * 40)

    df = read_raw_data(RAW_DATA_PATH)

    # Step 2: Segment events
    print("\n" + "-" * 40)
    print("STEP 2: SEGMENT EVENTS")
    print("-" * 40)

    events, features = segment_events(df)

    # Step 3: Load model
    print("\n" + "-" * 40)
    print("STEP 3: LOAD MODEL")
    print("-" * 40)

    model = CNNLSTMInference(checkpoint_path=CHECKPOINT_PATH)

    # Step 4: Classify events
    print("\n" + "-" * 40)
    print("STEP 4: CLASSIFY EVENTS")
    print("-" * 40)

    print(f"Classifying {len(events)} events...")
    start_time = time.time()

    results = classify_events(events, features, model)

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(events)/elapsed:.1f} events/sec)")

    # Step 5: Summary
    print("\n" + "-" * 40)
    print("STEP 5: RESULTS SUMMARY")
    print("-" * 40)

    # Class distribution
    print("\nPredicted class distribution:")
    class_counts = results["predicted_class"].value_counts()
    for cls, count in class_counts.items():
        pct = 100 * count / len(results)
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Volume by class
    print("\nVolume by predicted class:")
    for cls in class_counts.index:
        cls_volume = results[results["predicted_class"] == cls]["Volume"].sum()
        pct = 100 * cls_volume / results["Volume"].sum()
        print(f"  {cls}: {cls_volume:.1f} L ({pct:.1f}%)")

    # Confidence distribution
    print("\nConfidence distribution:")
    print(f"  Mean: {results['confidence'].mean():.3f}")
    print(f"  Min:  {results['confidence'].min():.3f}")
    print(f"  Max:  {results['confidence'].max():.3f}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
