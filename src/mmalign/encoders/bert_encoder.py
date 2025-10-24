"""
BERT-based text encoder for extracting contextual embeddings from text sequences.
Useful for text with timestamps (e.g., subtitles, transcripts).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# Optional transformers dependency
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BERTTextEncoder:
    """BERT encoder for text snippets."""
    
    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required. Install with: pip install transformers torch"
            )
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    
    def encode(self, texts: List[str], max_length: int = 128) -> np.ndarray:
        """
        Encode a list of text snippets.
        
        Args:
            texts: list of strings
            max_length: maximum token length
        
        Returns:
            embeddings: (len(texts), hidden_dim) array
        """
        if not texts:
            return np.array([])
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)
        
        return embeddings.cpu().numpy()


def encode_text_segments_with_bert(
    text_segments: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    pretrained_encoder: Optional[BERTTextEncoder] = None,
) -> np.ndarray:
    """
    Encode text segments using BERT.
    
    Args:
        text_segments: list of text strings
        model_name: Hugging Face model name
        max_length: max token length
        pretrained_encoder: if provided, use this encoder
    
    Returns:
        embeddings: (len(text_segments), hidden_dim) array
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and torch are required. Install with: pip install transformers torch"
        )
    
    if pretrained_encoder is None:
        encoder = BERTTextEncoder(model_name=model_name)
    else:
        encoder = pretrained_encoder
    
    return encoder.encode(text_segments, max_length=max_length)


def encode_timestamped_text(
    timestamps: np.ndarray,
    texts: List[str],
    bin_size: float = 1.0,
    model_name: str = "bert-base-uncased",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode timestamped text into a uniform time grid.
    
    Args:
        timestamps: (N,) array of start times
        texts: list of N text snippets
        bin_size: time bin size in seconds
        model_name: BERT model name
    
    Returns:
        bin_centers: (M,) array of time bin centers
        bin_embeddings: (M, hidden_dim) array of averaged embeddings per bin
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers and torch are required. Install with: pip install transformers torch"
        )
    
    if len(timestamps) == 0:
        return np.array([]), np.array([])
    
    encoder = BERTTextEncoder(model_name=model_name)
    embeddings = encoder.encode(texts)  # (N, hidden_dim)
    
    # Bin the embeddings
    t_min, t_max = float(timestamps.min()), float(timestamps.max())
    edges = np.arange(t_min, t_max + bin_size, bin_size)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Aggregate embeddings per bin
    bin_sums = np.zeros((len(bin_centers), embeddings.shape[1]))
    bin_counts = np.zeros(len(bin_centers))
    
    for i, t in enumerate(timestamps):
        bin_idx = int((t - t_min) / bin_size)
        if 0 <= bin_idx < len(bin_centers):
            bin_sums[bin_idx] += embeddings[i]
            bin_counts[bin_idx] += 1
    
    # Normalize
    mask = bin_counts > 0
    bin_embeddings = np.zeros_like(bin_sums)
    bin_embeddings[mask] = bin_sums[mask] / bin_counts[mask, None]
    
    return bin_centers, bin_embeddings

