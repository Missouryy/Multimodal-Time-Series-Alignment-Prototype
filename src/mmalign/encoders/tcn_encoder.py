"""
Temporal Convolutional Network (TCN) encoder for time-series data.
Alternative to LSTM with dilated causal convolutions for capturing long-range dependencies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import weight_norm
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Chomp1d(nn.Module if TORCH_AVAILABLE else object):
    """Remove padding from the end to ensure causality."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module if TORCH_AVAILABLE else object):
    """Single temporal block with dilated causal convolution."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Temporal Convolutional Network encoder."""
    def __init__(self, input_dim, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            input_dim: number of input features
            num_channels: list of channel sizes for each layer (e.g., [64, 64, 128])
            kernel_size: convolution kernel size
            dropout: dropout rate
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TCN encoder. Install with: pip install torch")
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                       dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim, seq_len)
        Returns:
            encoded: (batch, output_dim, seq_len)
        """
        return self.network(x)


def encode_timeseries_with_tcn(
    values: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    num_channels: list[int] = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
    pretrained_model: Optional[TCNEncoder] = None,
) -> np.ndarray:
    """
    Encode a time series using TCN.
    
    Args:
        values: (N, D) array
        timestamps: optional (N,) array (not used in this simple version)
        num_channels: list of channel sizes (default [64, 64, 128])
        kernel_size: convolution kernel size
        dropout: dropout rate
        pretrained_model: if provided, use this model
    
    Returns:
        encoded: (N, output_dim) array
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    input_dim = values.shape[1]
    
    if num_channels is None:
        num_channels = [64, 64, 128]
    
    if pretrained_model is None:
        model = TCNEncoder(input_dim, num_channels, kernel_size, dropout)
        model.eval()
    else:
        model = pretrained_model
        model.eval()
    
    # Convert to tensor (batch, channels, length)
    x = torch.from_numpy(values).float().permute(1, 0).unsqueeze(0)  # (1, D, N)
    
    with torch.no_grad():
        encoded = model(x)  # (1, output_dim, N)
    
    return encoded.squeeze(0).permute(1, 0).numpy()  # (N, output_dim)

