"""
CNN encoder for visual data (video frames, images).
Extracts spatial features that can be aligned with other modalities.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CNNFrameEncoder:
    """Pre-trained ResNet-based frame encoder."""
    
    def __init__(self, model_name: str = "resnet18", device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch and torchvision are required. Install with: pip install torch torchvision"
            )
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.output_dim = 512 if model_name == "resnet18" else 2048
    
    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Encode a single frame (RGB image).
        
        Args:
            frame: (H, W, 3) RGB image as uint8 numpy array
        
        Returns:
            feature: (output_dim,) array
        """
        # Convert to PIL
        img = Image.fromarray(frame.astype(np.uint8))
        img_t = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        
        with torch.no_grad():
            features = self.model(img_t)  # (1, output_dim, 1, 1)
            features = features.squeeze()  # (output_dim,)
        
        return features.cpu().numpy()
    
    def encode_frames(self, frames: list[np.ndarray]) -> np.ndarray:
        """
        Encode a list of frames.
        
        Args:
            frames: list of (H, W, 3) RGB images
        
        Returns:
            features: (len(frames), output_dim) array
        """
        features = []
        for frame in frames:
            features.append(self.encode_frame(frame))
        return np.stack(features, axis=0)


def encode_video_frames_with_cnn(
    frames: list[np.ndarray],
    model_name: str = "resnet18",
    pretrained_encoder: Optional[CNNFrameEncoder] = None,
) -> np.ndarray:
    """
    Encode video frames using a pre-trained CNN.
    
    Args:
        frames: list of (H, W, 3) RGB frames
        model_name: "resnet18" or "resnet50"
        pretrained_encoder: if provided, use this encoder
    
    Returns:
        features: (len(frames), output_dim) array
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch and torchvision are required. Install with: pip install torch torchvision"
        )
    
    if pretrained_encoder is None:
        encoder = CNNFrameEncoder(model_name=model_name)
    else:
        encoder = pretrained_encoder
    
    return encoder.encode_frames(frames)


def simple_spatial_features(frame: np.ndarray) -> dict:
    """
    Extract simple handcrafted spatial features from a frame (fallback without torch).
    
    Args:
        frame: (H, W, 3) RGB image
    
    Returns:
        features: dict with mean RGB, std RGB, edge energy
    """
    mean_rgb = frame.mean(axis=(0, 1)).tolist()
    std_rgb = frame.std(axis=(0, 1)).tolist()
    
    # Simple edge energy (gradient magnitude)
    gray = frame.mean(axis=2)
    gy, gx = np.gradient(gray)
    edge_energy = float(np.sqrt(gx**2 + gy**2).mean())
    
    return {
        "mean_r": mean_rgb[0],
        "mean_g": mean_rgb[1],
        "mean_b": mean_rgb[2],
        "std_r": std_rgb[0],
        "std_g": std_rgb[1],
        "std_b": std_rgb[2],
        "edge_energy": edge_energy,
    }

