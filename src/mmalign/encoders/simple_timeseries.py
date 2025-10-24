from __future__ import annotations

import numpy as np


def extract_basic_statistics(values: np.ndarray) -> dict:
    values = values if values.ndim == 2 else values.reshape(-1, 1)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }
