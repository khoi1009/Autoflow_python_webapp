"""
Training script for CNN-LSTM water end-use classifier.

Usage:
    python -m src.ml.train
"""

import torch
from .models import CNNLSTM
from .dataloader import get_data_loaders, FEATURE_COLS
from .trainer import Trainer


def main():
    """Main training function."""

    # Configuration
    DATA_DIR = "D:/ALL END USE DATA"  # Not used with precomputed data
    PRECOMPUTED_PATH = "prepared_ml_data_12features.pkl"
    CHECKPOINT_DIR = "checkpoints"

    BATCH_SIZE = 64
    MAX_LENGTH = 200
    NUM_EPOCHS = 100
    PATIENCE = 7
    LEARNING_RATE = 0.001
    GAMMA = 2.0  # Focal loss gamma

    NUM_CLASSES = 7  # After removing Evap cooler and Other
    NUM_FEATURES = 12
    HIDDEN_SIZE = 64
    NUM_LSTM_LAYERS = 2
    DROPOUT = 0.3

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        label_encoder,
        feature_scaler,
        feature_cols,
    ) = get_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        precomputed_path=PRECOMPUTED_PATH,
    )

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

    # Initialize model
    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)

    model = CNNLSTM(
        num_classes=NUM_CLASSES,
        num_features=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        label_encoder=label_encoder,
        feature_scaler=feature_scaler,
        feature_cols=feature_cols,
        device=device,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    history = trainer.train(num_epochs=NUM_EPOCHS, patience=PATIENCE)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation F1: {trainer.best_f1:.4f}")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
