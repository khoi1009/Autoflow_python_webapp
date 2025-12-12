"""
Simple rule-based classifier for water usage events.
"""

import pandas as pd
import numpy as np


def classify_events(events_df):
    """
    Classify water usage events based on their characteristics.

    Args:
        events_df: DataFrame with extracted events

    Returns:
        DataFrame with Category column added
    """
    if events_df.empty:
        events_df["Category"] = []
        return events_df

    categories = []

    for idx, event in events_df.iterrows():
        duration = event.get("duration_seconds", 0)
        volume = event.get("volume", 0)
        max_flow = event.get("max_flow_rate", 0)
        mean_flow = event.get("mean_flow_rate", 0)

        # Simple rule-based classification
        category = classify_single_event(duration, volume, max_flow, mean_flow)
        categories.append(category)

    events_df["Category"] = categories

    # Add display columns
    events_df["Start date"] = pd.to_datetime(events_df["datetime_start"]).dt.strftime(
        "%d-%b-%y"
    )
    events_df["Start time"] = pd.to_datetime(events_df["datetime_start"]).dt.strftime(
        "%H:%M:%S"
    )
    events_df["Duration (h:m:s)"] = events_df["duration_seconds"].apply(format_duration)
    events_df["Volume (L)"] = events_df["volume"].round(2)
    events_df["Max flow (litre/min)"] = events_df["max_flow_rate"].round(2)

    return events_df


def classify_single_event(duration, volume, max_flow, mean_flow):
    """
    Classify a single event based on its characteristics.

    Returns one of:
    - Shower, Tap, Toilet, Clothes Washer, Dishwasher,
      Bathtub, Irrigation, Evap Cooler, Leak, Other
    """
    # Toilet: Short duration, moderate volume, high initial flow
    if 3 <= duration <= 120 and 4 <= volume <= 12 and max_flow >= 8:
        return "Toilet"

    # Shower: Medium-long duration, higher volume
    if 120 <= duration <= 1200 and volume >= 20 and 5 <= mean_flow <= 15:
        return "Shower"

    # Bathtub: Long duration, high volume
    if duration >= 300 and volume >= 80:
        return "Bathtub"

    # Clothes Washer: Multiple cycles pattern, specific flow characteristics
    if 60 <= duration <= 3600 and 30 <= volume <= 150:
        return "Clothes Washer"

    # Dishwasher: Moderate duration and volume
    if 60 <= duration <= 3600 and 10 <= volume <= 40:
        return "Dishwasher"

    # Tap: Short duration, low volume
    if duration <= 120 and volume <= 10:
        return "Tap"

    # Irrigation: Very long duration, high volume
    if duration >= 600 and volume >= 50 and mean_flow <= 20:
        return "Irrigation"

    # Leak: Very low flow over long period
    if mean_flow < 0.5 and duration >= 300:
        return "Leak"

    # Evap Cooler: Specific pattern
    if 2 <= mean_flow <= 5 and duration >= 300:
        return "Evap Cooler"

    # Default to Other
    return "Other"


def format_duration(seconds):
    """Format duration in seconds to h:m:s string."""
    if pd.isna(seconds):
        return "0:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
