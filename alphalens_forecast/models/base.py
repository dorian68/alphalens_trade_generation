"""Abstract base forecaster interface."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


class BaseForecaster(ABC):
    """Unified interface implemented by forecasting backends."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(
        self,
        target: pd.Series,
        regressors: Optional[pd.DataFrame] = None,
    ) -> None:
        """Fit the underlying model to the target series."""

    @abstractmethod
    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate a forecast dataframe containing mean path and quantiles."""

    def save(self, path: Path) -> None:
        """Persist the fitted model to ``path`` using a backend-specific format."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save().")

    @classmethod
    def load_native(cls, path: Path) -> "BaseForecaster":
        """Reconstruct a model previously saved via ``save``."""
        raise NotImplementedError(f"{cls.__name__} does not implement load_native().")

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisable representation of the model state."""
        return {"pickled_model": pickle.dumps(self)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore the model from ``state`` produced by ``state_dict``."""
        restored = pickle.loads(state["pickled_model"])
        self.__dict__.update(restored.__dict__)
