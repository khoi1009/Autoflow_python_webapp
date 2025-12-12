"""
Script to add feature scaler to an existing checkpoint.

Usage:
    python add_scaler_to_checkpoint.py
"""

import os
import pickle
import torch
from sklearn.preprocessing import StandardScaler


def main():
    """Add scaler from precomputed data to existing checkpoint."""

    CHECKPOINT_PATH = "checkpoints/best_model.pth"
    PRECOMPUTED_PATH = "prepared_ml_data_12features.pkl"

    print("Adding feature scaler to checkpoint...")

    # Check files exist
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return

    if not os.path.exists(PRECOMPUTED_PATH):
        print(f"ERROR: Precomputed data not found: {PRECOMPUTED_PATH}")
        return

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Check if scaler already exists
    if "feature_scaler" in checkpoint:
        print("Checkpoint already has feature_scaler!")
        return

    # Load precomputed data
    print(f"Loading precomputed data: {PRECOMPUTED_PATH}")
    with open(PRECOMPUTED_PATH, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    feature_cols = data.get("feature_cols", None)

    print(f"  Features shape: {features.shape}")

    # Fit scaler on all data (or just training data ideally)
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(features)

    # Add to checkpoint
    checkpoint["feature_scaler"] = scaler
    checkpoint["feature_cols"] = feature_cols
    checkpoint["num_features"] = features.shape[1]

    # Save updated checkpoint
    print(f"Saving updated checkpoint: {CHECKPOINT_PATH}")
    torch.save(checkpoint, CHECKPOINT_PATH)

    print("Done!")
    print(f"  Scaler mean: {scaler.mean_}")
    print(f"  Scaler std: {scaler.scale_}")


if __name__ == "__main__":
    main()
