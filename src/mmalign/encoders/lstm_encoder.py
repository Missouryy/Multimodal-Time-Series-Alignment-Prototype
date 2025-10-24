"""
LSTM encoder for time-series data.
Captures temporal dependencies in sequential sensor/audio features.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Optional torch dependency
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LSTMEncoder(nn.Module if TORCH_AVAILABLE else object):
    """Simple LSTM encoder for multivariate time series."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, bidirectional: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM encoder. Install with: pip install torch")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            encoded: (batch, seq_len, output_dim)
        """
        output, (h_n, c_n) = self.lstm(x)
        return output


def encode_timeseries_with_lstm(
    values: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    hidden_dim: int = 64,
    num_layers: int = 2,
    bidirectional: bool = True,
    pretrained_model: Optional[LSTMEncoder] = None,
) -> np.ndarray:
    """
    Encode a time series using LSTM.
    
    Args:
        values: (N, D) array
        timestamps: optional (N,) array (not used in this simple version)
        hidden_dim: LSTM hidden dimension
        num_layers: number of LSTM layers
        bidirectional: use bidirectional LSTM
        pretrained_model: if provided, use this model instead of creating a new one
    
    Returns:
        encoded: (N, output_dim) array
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    
    input_dim = values.shape[1]
    
    if pretrained_model is None:
        model = LSTMEncoder(input_dim, hidden_dim, num_layers, bidirectional)
        model.eval()
    else:
        model = pretrained_model
        model.eval()
    
    # Convert to tensor
    x = torch.from_numpy(values).float().unsqueeze(0)  # (1, N, D)
    
    with torch.no_grad():
        encoded = model(x)  # (1, N, output_dim)
    
    return encoded.squeeze(0).numpy()  # (N, output_dim)

