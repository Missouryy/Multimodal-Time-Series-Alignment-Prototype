from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class ModalityType(Enum):
    SENSOR = 1
    AUDIO = 2
    VIDEO = 3
    TEXT = 4


@dataclass
class TimeSeries:
    name: str
    modality: ModalityType
    timestamps: np.ndarray  # shape (N,)
    values: np.ndarray  # shape (N, D)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy DataFrame with columns: time, v0..v{D-1}."""
        n = self.values.shape[1] if self.values.ndim == 2 else 1
        vals = self.values if self.values.ndim == 2 else self.values.reshape(-1, 1)
        
        # 确保时间戳和值的长度匹配
        if len(self.timestamps) != len(vals):
            raise ValueError(
                f"Timestamps length ({len(self.timestamps)}) does not match values length ({len(vals)}). "
                f"This may occur if using an encoder that changes the number of data points."
            )
        
        cols = [f"v{i}" for i in range(n)]
        return pd.DataFrame({"time": self.timestamps, **{c: vals[:, i] for i, c in enumerate(cols)}})

    def copy_with(
        self,
        *,
        timestamps: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "TimeSeries":
        return TimeSeries(
            name=name or self.name,
            modality=self.modality,
            timestamps=timestamps if timestamps is not None else self.timestamps.copy(),
            values=values if values is not None else self.values.copy(),
            metadata=metadata if metadata is not None else dict(self.metadata),
        )

    @property
    def duration(self) -> float:
        if len(self.timestamps) == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def rate_hz(self) -> Optional[float]:
        if len(self.timestamps) < 2:
            return None
        dt = np.diff(self.timestamps)
        med = np.median(dt)
        if med <= 0:
            return None
        return float(1.0 / med)
