"""
Trainer class for CNN-LSTM model training.
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from typing import Dict, Optional, Tuple
from .models import CNNLSTM, FocalLoss


class Trainer:
    """
    Trainer for CNN-LSTM water end-use classifier.
    """

    def __init__(
        self,
        model: CNNLSTM,
        train_loader,
        val_loader,
        class_weights: torch.Tensor,
        label_encoder,
        feature_scaler,
        feature_cols: list,
        device: str = "cuda",
        lr: float = 0.001,
        gamma: float = 2.0,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights.to(device)
        self.label_encoder = label_encoder
        self.feature_scaler = feature_scaler
        self.feature_cols = feature_cols
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss function with focal loss
        self.criterion = FocalLoss(alpha=class_weights, gamma=gamma)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # Tracking
        self.best_f1 = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (series, features, labels) in enumerate(self.train_loader):
            series = series.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(series, features)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for series, features, labels in self.val_loader:
            series = series.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(series, features)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")

        return val_loss, val_f1

    def save_checkpoint(self, epoch: int, val_f1: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
            "label_encoder": self.label_encoder,
            "feature_scaler": self.feature_scaler,
            "feature_cols": self.feature_cols,
            "num_features": self.model.num_features,
        }

        path = os.path.join(self.checkpoint_dir, "last_model.pth")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (F1: {val_f1:.4f})")

    def train(self, num_epochs: int = 100, patience: int = 7) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience

        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        print(f"  Features: {self.feature_cols}")
        print()

        no_improve = 0

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_f1 = self.validate()

            # Update scheduler
            self.scheduler.step(val_f1)

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_f1)

            # Check for improvement
            is_best = val_f1 > self.best_f1
            if is_best:
                self.best_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_f1, is_best)

            # Print progress
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Best F1: {self.best_f1:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping
            if no_improve >= patience:
                print(
                    f"\nEarly stopping after {epoch+1} epochs (no improvement for {patience} epochs)"
                )
                break

        print(f"\nTraining completed. Best F1: {self.best_f1:.4f}")

        # Print final classification report
        self._print_final_report()

        return self.history

    @torch.no_grad()
    def _print_final_report(self):
        """Print final classification report on validation set."""
        self.model.eval()
        all_preds = []
        all_labels = []

        for series, features, labels in self.val_loader:
            series = series.to(self.device)
            features = features.to(self.device)

            outputs = self.model(series, features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print("\n" + "=" * 60)
        print("VALIDATION SET CLASSIFICATION REPORT")
        print("=" * 60)
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=self.label_encoder.classes_,
                digits=4,
            )
        )

        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print(cm)
