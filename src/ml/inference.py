"""
Inference module for CNN-LSTM water end-use classifier.
"""

import os
import numpy as np
import torch
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple
from .models import CNNLSTM


def compute_additional_features_single(
    raw_series: np.ndarray, start_hour: int = 12
) -> dict:
    """
    Compute additional shape-based features from a single raw flow series.

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

    # 1. raw_series_length
    raw_series_length = len(raw_series)

    # 2. flow_std
    flow_std = float(np.std(raw_series))

    # 3. num_peaks
    try:
        peaks, _ = find_peaks(raw_series, height=np.mean(raw_series) * 0.5)
        num_peaks = len(peaks)
    except:
        num_peaks = 0

    # 4. hour_of_day
    hour_of_day = start_hour

    # 5. rise_time
    max_idx = np.argmax(raw_series)
    rise_time = max_idx

    # 6. fall_time
    fall_time = len(raw_series) - max_idx - 1

    # 7. plateau_ratio
    max_flow = np.max(raw_series)
    if max_flow > 0:
        threshold = max_flow * 0.8
        plateau_count = np.sum(raw_series >= threshold)
        plateau_ratio = plateau_count / len(raw_series)
    else:
        plateau_ratio = 0.0

    # 8. flow_slope
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


class CNNLSTMInference:
    """
    Inference wrapper for CNN-LSTM model.
    """

    def __init__(
        self, checkpoint_path: str = "checkpoints/best_model.pth", device: str = None
    ):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load label encoder
        self.label_encoder = checkpoint["label_encoder"]
        self.categories = list(self.label_encoder.classes_)

        # Load feature scaler (if available)
        self.feature_scaler = checkpoint.get("feature_scaler", None)
        self.feature_cols = checkpoint.get("feature_cols", None)

        # Auto-detect num_features from model weights
        fc1_weight = checkpoint["model_state_dict"]["fc1.weight"]
        fc1_input_size = fc1_weight.shape[1]
        # fc1_input_size = hidden_size * 2 (bidirectional) + num_features
        # hidden_size = 64, so: num_features = fc1_input_size - 128
        num_features = fc1_input_size - 128

        # Initialize model
        self.model = CNNLSTM(
            num_classes=len(self.categories),
            num_features=num_features,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.max_length = 200
        self.best_f1 = checkpoint.get("best_f1", 0.0)

        print(f"[OK] Loaded model from: {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Categories: {self.categories}")
        print(f"  Num features: {num_features}")
        print(f"  Best F1 Score: {self.best_f1:.4f}")

    def prepare_input(
        self,
        raw_series: np.ndarray,
        duration: float,
        volume: float,
        max_flow: float,
        mode_flow: float,
        start_hour: int = 12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for the model.

        Args:
            raw_series: Flow rate time series (L/10s)
            duration: Event duration in seconds
            volume: Total volume in liters
            max_flow: Maximum flow rate (L/min)
            mode_flow: Mode flow rate (L/min)
            start_hour: Hour of day when event started

        Returns:
            Tuple of (series_tensor, features_tensor)
        """
        # Pad or truncate series
        if len(raw_series) > self.max_length:
            series = raw_series[: self.max_length]
        else:
            series = np.pad(
                raw_series, (0, self.max_length - len(raw_series)), mode="constant"
            )

        series_tensor = torch.tensor(series, dtype=torch.float32).unsqueeze(0)

        # Compute additional features
        additional = compute_additional_features_single(raw_series, start_hour)

        # Build feature vector (must match FEATURE_COLS order)
        features = np.array(
            [
                duration,
                volume,
                max_flow,
                mode_flow,
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
        ).reshape(1, -1)

        # Apply scaler if available
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)

        features_tensor = torch.tensor(features, dtype=torch.float32)

        return series_tensor.to(self.device), features_tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        raw_series: np.ndarray,
        duration: float,
        volume: float,
        max_flow: float,
        mode_flow: float,
        start_hour: int = 12,
    ) -> Dict[str, float]:
        """
        Predict class probabilities for a single event.

        Returns:
            Dictionary mapping class names to probabilities
        """
        series_tensor, features_tensor = self.prepare_input(
            raw_series, duration, volume, max_flow, mode_flow, start_hour
        )

        logits = self.model(series_tensor, features_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        return {cat: float(prob) for cat, prob in zip(self.categories, probs)}

    @torch.no_grad()
    def predict_batch(
        self,
        raw_series_list: List[np.ndarray],
        durations: np.ndarray,
        volumes: np.ndarray,
        max_flows: np.ndarray,
        mode_flows: np.ndarray,
        start_hours: np.ndarray = None,
    ) -> List[Dict[str, float]]:
        """
        Predict class probabilities for a batch of events.

        Returns:
            List of dictionaries mapping class names to probabilities
        """
        batch_size = len(raw_series_list)

        if start_hours is None:
            start_hours = np.full(batch_size, 12)

        # Prepare all inputs
        all_series = []
        all_features = []

        for i in range(batch_size):
            series_tensor, features_tensor = self.prepare_input(
                raw_series_list[i],
                durations[i],
                volumes[i],
                max_flows[i],
                mode_flows[i],
                start_hours[i],
            )
            all_series.append(series_tensor)
            all_features.append(features_tensor)

        # Stack tensors
        series_batch = torch.cat(all_series, dim=0)
        features_batch = torch.cat(all_features, dim=0)

        # Forward pass
        logits = self.model(series_batch, features_batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        # Convert to list of dicts
        results = []
        for i in range(batch_size):
            results.append(
                {cat: float(probs[i, j]) for j, cat in enumerate(self.categories)}
            )

        return results
