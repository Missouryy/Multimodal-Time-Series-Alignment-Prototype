from __future__ import annotations

import numpy as np


def frame_mean_rgb(frame: np.ndarray) -> list:
    # frame expected in RGB
    mean_vals = frame.mean(axis=(0, 1))
    return [float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2])]
