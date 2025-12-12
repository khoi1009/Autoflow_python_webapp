"""
CNN-LSTM Model for water end-use classification.
Combines convolutional layers for pattern extraction with LSTM for temporal dependencies.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM model for water end-use event classification.

    Architecture:
    - 1D CNN layers to extract local patterns from flow rate time series
    - LSTM layer to capture temporal dependencies
    - Fully connected layers with feature concatenation
    - Output: class probabilities for each end-use category
    """

    def __init__(
        self,
        num_classes: int = 7,
        num_features: int = 12,
        hidden_size: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(CNNLSTM, self).__init__()

        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_size = hidden_size

        # 1D CNN layers for pattern extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(dropout)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Fully connected layers
        # LSTM output (bidirectional) + features
        fc_input_size = hidden_size * 2 + num_features

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        self.dropout_fc = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_series: torch.Tensor, x_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x_series: Flow rate time series, shape (batch, seq_len)
            x_features: Additional features, shape (batch, num_features)

        Returns:
            Class logits, shape (batch, num_classes)
        """
        batch_size = x_series.size(0)

        # Add channel dimension for CNN: (batch, seq_len) -> (batch, 1, seq_len)
        x = x_series.unsqueeze(1)

        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # Reshape for LSTM: (batch, channels, seq) -> (batch, seq, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden_size)
        h_forward = h_n[-2, :, :]  # Last layer forward
        h_backward = h_n[-1, :, :]  # Last layer backward
        lstm_features = torch.cat([h_forward, h_backward], dim=1)

        # Concatenate LSTM output with additional features
        combined = torch.cat([lstm_features, x_features], dim=1)

        # Fully connected layers
        out = self.relu(self.fc1(combined))
        out = self.dropout_fc(out)
        out = self.relu(self.fc2(out))
        out = self.dropout_fc(out)
        out = self.fc3(out)

        return out


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights (tensor or None)
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits, shape (batch, num_classes)
            targets: Ground truth labels, shape (batch,)
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
