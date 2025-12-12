"""
Event extraction module for water usage data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def extract_events(df):
    """
    Extract water usage events from raw flow data.

    Args:
        df: DataFrame with timestamp and flow_rate columns

    Returns:
        DataFrame with extracted events
    """
    # Ensure we have the required columns
    if "timestamp" not in df.columns and "datetime" not in df.columns:
        # Try to find a datetime column
        datetime_cols = [
            c for c in df.columns if "date" in c.lower() or "time" in c.lower()
        ]
        if datetime_cols:
            df["timestamp"] = pd.to_datetime(df[datetime_cols[0]])
        else:
            raise ValueError("No timestamp column found in data")

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])

    # Find flow rate column
    flow_col = None
    for col in df.columns:
        if "flow" in col.lower() or "rate" in col.lower():
            flow_col = col
            break

    if flow_col is None:
        # Use the first numeric column that's not timestamp
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            flow_col = numeric_cols[0]
        else:
            raise ValueError("No flow rate column found in data")

    df["flow_rate"] = pd.to_numeric(df[flow_col], errors="coerce").fillna(0)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Define threshold for event detection
    threshold = 0.1  # L/min

    # Find event boundaries
    df["is_flowing"] = df["flow_rate"] > threshold
    df["event_change"] = df["is_flowing"].diff().fillna(False)

    # Group consecutive flowing periods into events
    events = []
    current_event = None

    for idx, row in df.iterrows():
        if row["is_flowing"]:
            if current_event is None:
                current_event = {
                    "start_idx": idx,
                    "datetime_start": row["timestamp"],
                    "flow_rates": [row["flow_rate"]],
                }
            else:
                current_event["flow_rates"].append(row["flow_rate"])
        else:
            if current_event is not None:
                # Close the current event
                current_event["end_idx"] = idx - 1
                current_event["datetime_end"] = df.loc[idx - 1, "timestamp"]
                current_event["duration_seconds"] = (
                    current_event["datetime_end"] - current_event["datetime_start"]
                ).total_seconds()
                current_event["volume"] = (
                    sum(current_event["flow_rates"]) / 60
                )  # Convert to liters
                current_event["max_flow_rate"] = max(current_event["flow_rates"])
                current_event["mean_flow_rate"] = np.mean(current_event["flow_rates"])
                events.append(current_event)
                current_event = None

    # Handle case where data ends during an event
    if current_event is not None:
        current_event["end_idx"] = len(df) - 1
        current_event["datetime_end"] = df.iloc[-1]["timestamp"]
        current_event["duration_seconds"] = (
            current_event["datetime_end"] - current_event["datetime_start"]
        ).total_seconds()
        current_event["volume"] = sum(current_event["flow_rates"]) / 60
        current_event["max_flow_rate"] = max(current_event["flow_rates"])
        current_event["mean_flow_rate"] = np.mean(current_event["flow_rates"])
        events.append(current_event)

    # Convert to DataFrame
    events_df = pd.DataFrame(events)

    return events_df
