"""Lightweight scaler utilities for univariate time-series."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from darts import TimeSeries


@dataclass
class ScalerState:
    """Serializable scaler state."""

    mean: float = 0.0
    std: float = 1.0


class ScalerWrapper:
    """Simple z-score scaler for Darts TimeSeries objects."""

    def __init__(self) -> None:
        self._state = ScalerState()
        self._fitted: bool = False

    @property
    def mean_(self) -> float:
        return self._state.mean

    @property
    def std_(self) -> float:
        return self._state.std

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_from_series(self, series: TimeSeries) -> "ScalerWrapper":
        """Fit scaler parameters using a Darts series."""
        values = series.univariate_values()
        return self.fit(values)

    def fit(self, values: np.ndarray) -> "ScalerWrapper":
        """Fit scaler parameters from a numpy array."""
        flattened = np.asarray(values, dtype=np.float64).ravel()
        if flattened.size == 0:
            raise ValueError("Cannot fit scaler on an empty array.")
        mean = float(np.mean(flattened))
        std = float(np.std(flattened))
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        self._state = ScalerState(mean=mean, std=std)
        self._fitted = True
        return self

    def transform(self, series: TimeSeries) -> TimeSeries:
        """Apply scaling to a time-series."""
        self._assert_fitted()
        values = series.univariate_values(copy=True).astype(np.float32)
        scaled = (values - self.mean_) / self.std_
        return TimeSeries.from_times_and_values(
            series.time_index,
            scaled.reshape(-1, 1),
            columns=series.components,
        )

    def inverse_transform(self, series: TimeSeries) -> TimeSeries:
        """Undo scaling."""
        self._assert_fitted()
        values = series.univariate_values(copy=True).astype(np.float32)
        restored = values * self.std_ + self.mean_
        return TimeSeries.from_times_and_values(
            series.time_index,
            restored.reshape(-1, 1),
            columns=series.components,
        )

    def to_dict(self) -> Dict[str, float]:
        """Return JSON-serialisable representation."""
        return asdict(self._state)

    @classmethod
    def from_dict(cls, payload: Dict[str, float]) -> "ScalerWrapper":
        """Reconstruct scaler from a dictionary."""
        scaler = cls()
        scaler._state = ScalerState(mean=float(payload["mean"]), std=float(payload["std"]))
        scaler._fitted = True
        return scaler

    def save(self, path: Path) -> None:
        """Persist scaler parameters to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle)

    @classmethod
    def load(cls, path: Path) -> "ScalerWrapper":
        """Load scaler parameters from disk."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted before use.")


__all__ = ["ScalerWrapper"]
