"""
Test script for clotheswasher classifier.

Usage:
    python test_clotheswasher_classifier.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from typing import Tuple, List

from src.ml.inference import CNNLSTMInference
from src.classifiers.clotheswasher_classifier import ClotheswasherClassifier


def read_raw_data(file_path: str) -> pd.DataFrame:
    """Read raw meter data from CSV."""
    print(f"Reading raw data from: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"  Total volume: {df['Usage'].sum():.2f} L")

    return df


def segment_events(
    df: pd.DataFrame,
    gap_threshold: int = 2,  # Number of consecutive zeros to define gap
) -> Tuple[List[np.ndarray], np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Segment continuous meter data into discrete water use events.

    Returns:
        events: List of raw flow series (L/10s) for each event
        event_indices: Original indices in df for each event's start
        event_features: DataFrame with Volume, Duration, Max_flow, Mode_flow
        start_times: DataFrame with year, month, day, hour, minute, second
        end_times: DataFrame with year, month, day, hour, minute, second
    """
    print("\nSegmenting events...")

    usage = df["Usage"].values
    datetime_col = df["Datetime"]

    events = []
    event_indices = []
    event_features_list = []
    start_times_list = []
    end_times_list = []

    n = len(usage)
    i = 0

    while i < n:
        # Skip zeros
        if usage[i] == 0:
            i += 1
            continue

        # Found start of event
        start_idx = i

        # Find end of event (gap_threshold consecutive zeros)
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

        # Event is from start_idx to j - gap_threshold (or j if no zeros at end)
        end_idx = j - zero_count if zero_count > 0 else j

        if end_idx > start_idx:
            # Extract event
            event_usage = usage[start_idx:end_idx]

            # Calculate features
            volume = np.sum(event_usage)  # L (already in liters per 10s)
            duration = (end_idx - start_idx) * 10  # seconds
            max_flow = np.max(event_usage) * 6  # L/min
            mode_result = mode(event_usage[event_usage > 0], keepdims=True)
            mode_flow = (
                mode_result.mode[0] if len(mode_result.mode) > 0 else 0
            ) * 6  # L/min

            events.append(event_usage)
            event_indices.append(start_idx)

            event_features_list.append(
                {
                    "Volume": volume,
                    "Duration": duration,
                    "Max_flow": max_flow,
                    "Mode_flow": mode_flow,
                }
            )

            # Start time
            start_dt = datetime_col.iloc[start_idx]
            start_times_list.append(
                {
                    "year": start_dt.year,
                    "month": start_dt.month,
                    "day": start_dt.day,
                    "hour": start_dt.hour,
                    "minute": start_dt.minute,
                    "second": start_dt.second,
                }
            )

            # End time
            end_dt = datetime_col.iloc[end_idx - 1]
            end_times_list.append(
                {
                    "year": end_dt.year,
                    "month": end_dt.month,
                    "day": end_dt.day,
                    "hour": end_dt.hour,
                    "minute": end_dt.minute,
                    "second": end_dt.second,
                }
            )

        i = j

    event_features = pd.DataFrame(event_features_list)
    start_times = pd.DataFrame(start_times_list)
    end_times = pd.DataFrame(end_times_list)

    print(f"  Found {len(events)} events")
    print(
        f"  Volume range: {event_features['Volume'].min():.2f} - {event_features['Volume'].max():.2f} L"
    )
    print(
        f"  Duration range: {event_features['Duration'].min():.0f} - {event_features['Duration'].max():.0f} s"
    )
    print(
        f"  Max flow range: {event_features['Max_flow'].min():.2f} - {event_features['Max_flow'].max():.2f} L/min"
    )

    # Debug: Check data scale
    print(f"\n  DEBUG - Data scale check:")
    print(f"    Raw series for CNN (L/10s): max={max(np.max(e) for e in events):.4f}")
    print(f"    Features Max_flow (L/min): max={event_features['Max_flow'].max():.4f}")
    print(
        f"    Ratio (should be ~6x): {event_features['Max_flow'].max() / max(np.max(e) for e in events):.1f}x"
    )

    print(f"\n  DEBUG - Data scale check:")
    print(f"    First event max value: {np.max(events[0]):.4f}")
    print(f"    First event mean value: {np.mean(events[0]):.4f}")
    print(f"    All events max value: {max(np.max(e) for e in events):.4f}")
    print(f"    Sample of first event (first 10 values): {events[0][:10]}")

    return events, np.array(event_indices), event_features, start_times, end_times


def plot_clotheswasher_events(
    events: list,
    classified_indices: list,
    num_loads: int,
):
    """
    Plot all classified clotheswasher events concatenated into a single vector
    """

    if len(classified_indices) == 0:
        print("\nNo clotheswasher events found!")
        return

    print(
        f"\nPlotting {len(classified_indices)} clotheswasher events ({num_loads} loads)..."
    )

    # Concatenate all clotheswasher events into a single vector
    all_flow_values = []
    for idx in classified_indices:
        all_flow_values.extend(events[idx])

    all_flow_values = np.array(all_flow_values)

    # Convert from L/10s to L/min (multiply by 6)
    all_flow_values_lpm = all_flow_values * 6

    # Create time axis (10-second intervals)
    time_seconds = np.arange(len(all_flow_values)) * 10
    time_minutes = time_seconds / 60

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(time_minutes, all_flow_values_lpm, "b-", linewidth=0.5)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Flow Rate (L/min)")
    ax.set_title(
        f"All Clotheswasher Events Concatenated ({len(classified_indices)} events, {num_loads} loads)"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = "clotheswasher_all_events.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to: {output_path}")

    plt.show()


def main():
    """
    Main test script
    """

    print("=" * 80)
    print("CLOTHESWASHER CLASSIFIER TEST")
    print("=" * 80)

    # Configuration
    RAW_DATA_PATH = r"D:\ALL END USE DATA\SYDNEY WATER DATA\TRANCHE 1\Tranche 1 - Raw data\BTJB0601.csv"
    CHECKPOINT_PATH = "checkpoints/best_model.pth"

    # Step 1: Read raw data
    print("\n" + "=" * 80)
    print("STEP 1: READ RAW DATA")
    print("=" * 80)

    if not os.path.exists(RAW_DATA_PATH):
        print(f"ERROR: Raw data file not found: {RAW_DATA_PATH}")
        return

    df = read_raw_data(RAW_DATA_PATH)

    # Step 2: Segment events
    print("\n" + "=" * 80)
    print("STEP 2: SEGMENT EVENTS")
    print("=" * 80)

    events, event_indices, event_features, start_times, end_times = segment_events(df)

    # Step 3: Load CNN-LSTM model
    print("\n" + "=" * 80)
    print("STEP 3: LOAD CNN-LSTM MODEL")
    print("=" * 80)

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Model checkpoint not found: {CHECKPOINT_PATH}")
        print("Please train the model first using: python -m src.ml.train")
        return

    import torch

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Check if scaler is in checkpoint
    if "feature_scaler" in checkpoint:
        print("[OK] Loaded StandardScaler from checkpoint")
        print(f"  Num features: {checkpoint.get('num_features', 'unknown')}")
        if "feature_cols" in checkpoint:
            print(f"  Feature names: {checkpoint['feature_cols']}")
    else:
        print(
            "[WARNING] No StandardScaler in checkpoint - features won't be normalized"
        )

    cnn_model = CNNLSTMInference(checkpoint_path=CHECKPOINT_PATH)

    # Step 4: Initialize clotheswasher classifier
    print("\n" + "=" * 80)
    print("STEP 4: INITIALIZE CLOTHESWASHER CLASSIFIER")
    print("=" * 80)

    classifier = ClotheswasherClassifier(cnn_model=cnn_model)
    print("[OK] Clotheswasher classifier initialized")

    # Step 5: Classify events
    print("\n" + "=" * 80)
    print("STEP 5: CLASSIFY CLOTHESWASHER EVENTS")
    print("=" * 80)

    import time

    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Starting classification...")

    results = classifier.classify(
        events=events,
        event_features=event_features,
        start_times=start_times,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(
        f"\n[{time.strftime('%H:%M:%S')}] Classification completed in {elapsed:.2f} seconds"
    )

    # Print results
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    print(f"  Total events analyzed: {len(events)}")
    print(f"  Clotheswasher events found: {len(results['indices'])}")
    print(f"  Number of loads: {results['num_loads']}")
    print(f"  Threshold flow rate: {results['threshold_flow']:.2f} L/min")
    print(f"  Average load duration: {results['avg_duration']:.1f} seconds")

    if len(results["volumes_per_load"]) > 0:
        print(f"\n  Volumes per load:")
        for i, vol in enumerate(results["volumes_per_load"], 1):
            print(f"    Load {i}: {vol:.1f} L")

    # Step 6: Plot results
    print("\n" + "=" * 80)
    print("STEP 6: VISUALIZE RESULTS")
    print("=" * 80)

    if len(results["indices"]) > 0:
        # Plot all events concatenated
        plot_clotheswasher_events(
            events=events,
            classified_indices=results["indices"],
            num_loads=results["num_loads"],
        )
    else:
        print("  No clotheswasher events found to plot.")

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
