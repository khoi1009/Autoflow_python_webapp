"""
Data loader for water end-use classification.
Handles data preparation, feature computation, and property-aware train/val/test splits.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import find_peaks
from scipy.stats import mode
from collections import defaultdict
from typing import Tuple, List, Dict, Optional


# Feature columns used for classification (must match inference.py)
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
    Compute 7 additional shape-based features from raw flow series.

    Args:
        raw_series: Flow rate time series (L/10s)
        start_hour: Hour of day when event started (0-23)

    Returns:
        Dictionary with computed features
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

    # 1. raw_series_length - number of 10-second intervals
    raw_series_length = len(raw_series)

    # 2. flow_std - standard deviation of flow
    flow_std = float(np.std(raw_series))

    # 3. num_peaks - number of peaks in the flow signal
    try:
        peaks, _ = find_peaks(raw_series, height=np.mean(raw_series) * 0.5)
        num_peaks = len(peaks)
    except:
        num_peaks = 0

    # 4. hour_of_day - already provided
    hour_of_day = start_hour

    # 5. rise_time - time to reach max flow from start
    max_idx = np.argmax(raw_series)
    rise_time = max_idx  # in 10-second units

    # 6. fall_time - time from max flow to end
    fall_time = len(raw_series) - max_idx - 1

    # 7. plateau_ratio - fraction of time near max flow (within 80%)
    max_flow = np.max(raw_series)
    if max_flow > 0:
        threshold = max_flow * 0.8
        plateau_count = np.sum(raw_series >= threshold)
        plateau_ratio = plateau_count / len(raw_series)
    else:
        plateau_ratio = 0.0

    # 8. flow_slope - overall trend (linear regression slope)
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


class WaterEndUseDataset(Dataset):
    """
    PyTorch Dataset for water end-use events.
    """

    def __init__(
        self,
        raw_series_list: List[np.ndarray],
        features: np.ndarray,
        labels: np.ndarray,
        max_length: int = 200,
    ):
        self.raw_series_list = raw_series_list
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get raw series and pad/truncate to max_length
        raw_series = self.raw_series_list[idx]

        if len(raw_series) > self.max_length:
            raw_series = raw_series[: self.max_length]
        else:
            raw_series = np.pad(
                raw_series, (0, self.max_length - len(raw_series)), mode="constant"
            )

        series_tensor = torch.tensor(raw_series, dtype=torch.float32)

        return series_tensor, self.features[idx], self.labels[idx]


def get_data_loaders(
    data_dir: str,
    batch_size: int = 64,
    max_length: int = 200,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    use_precomputed: bool = True,
    precomputed_path: str = "prepared_ml_data_12features.pkl",
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    torch.Tensor,
    LabelEncoder,
    StandardScaler,
    List[str],
]:
    """
    Create train/val/test data loaders with property-aware splits.

    Returns:
        train_loader, val_loader, test_loader, class_weights, label_encoder, feature_scaler, feature_cols
    """
    np.random.seed(random_seed)

    # Try to load precomputed data
    if use_precomputed and os.path.exists(precomputed_path):
        print(f"Loading precomputed data from {precomputed_path}...")
        with open(precomputed_path, "rb") as f:
            data = pickle.load(f)

        all_raw_series = data["raw_series"]
        all_features = data["features"]
        all_labels = data["labels"]
        property_ids = data["property_ids"]

        print(
            f"  Loaded {len(all_labels)} events from {len(set(property_ids))} properties"
        )
    else:
        raise FileNotFoundError(
            f"Precomputed data not found at {precomputed_path}. "
            "Run prepare_ml_data.py first to generate the data."
        )

    # Remove unwanted classes (Evap cooler, Other)
    classes_to_remove = ["Evap cooler", "Other"]
    mask = ~np.isin(all_labels, classes_to_remove)

    all_raw_series = [s for s, m in zip(all_raw_series, mask) if m]
    all_features = all_features[mask]
    all_labels = all_labels[mask]
    property_ids = property_ids[mask]

    print(f"  After removing {classes_to_remove}: {len(all_labels)} events")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    print(f"  Classes: {list(label_encoder.classes_)}")

    # Property-aware split
    unique_properties = np.unique(property_ids)
    np.random.shuffle(unique_properties)

    n_properties = len(unique_properties)
    n_val = int(n_properties * val_ratio)
    n_test = int(n_properties * test_ratio)
    n_train = n_properties - n_val - n_test

    train_properties = set(unique_properties[:n_train])
    val_properties = set(unique_properties[n_train : n_train + n_val])
    test_properties = set(unique_properties[n_train + n_val :])

    print(f"  Property split: {n_train} train / {n_val} val / {n_test} test")

    # Create masks
    train_mask = np.array([p in train_properties for p in property_ids])
    val_mask = np.array([p in val_properties for p in property_ids])
    test_mask = np.array([p in test_properties for p in property_ids])

    # Split data
    train_series = [s for s, m in zip(all_raw_series, train_mask) if m]
    val_series = [s for s, m in zip(all_raw_series, val_mask) if m]
    test_series = [s for s, m in zip(all_raw_series, test_mask) if m]

    train_features = all_features[train_mask]
    val_features = all_features[val_mask]
    test_features = all_features[test_mask]

    train_labels = encoded_labels[train_mask]
    val_labels = encoded_labels[val_mask]
    test_labels = encoded_labels[test_mask]

    print(
        f"  Event split: {len(train_labels)} train / {len(val_labels)} val / {len(test_labels)} test"
    )

    # Normalize features
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train_features)
    val_features = feature_scaler.transform(val_features)
    test_features = feature_scaler.transform(test_features)

    # Compute class weights for imbalanced data
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts)

    # Boost clotheswasher weight
    cw_idx = list(label_encoder.classes_).index("Clotheswasher")
    class_weights[cw_idx] *= 2.0

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(
        f"  Class weights: {dict(zip(label_encoder.classes_, class_weights.numpy()))}"
    )

    # Create datasets
    train_dataset = WaterEndUseDataset(
        train_series, train_features, train_labels, max_length
    )
    val_dataset = WaterEndUseDataset(val_series, val_features, val_labels, max_length)
    test_dataset = WaterEndUseDataset(
        test_series, test_features, test_labels, max_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        label_encoder,
        feature_scaler,
        FEATURE_COLS,
    )
