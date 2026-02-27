"""Abstract base forecaster interface."""
from __future__ import annotations

import io
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _is_cuda_deserialize_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "attempting to deserialize object on a cuda device" in message
        or "torch.cuda.is_available() is false" in message
    )


class BaseForecaster(ABC):
    """Unified interface implemented by forecasting backends."""

    name: str

    def __init__(self, name: str, device: str = "cpu") -> None:
        self.name = name
        # Device is an execution concern; non-Torch models safely ignore it.
        self.device = device
        self._dataloader_config = None

    def set_device(self, device: str) -> None:
        """Update the runtime device without altering model behavior."""
        self.device = device

    def set_dataloader_config(self, config) -> None:
        """Store optional DataLoader config for Torch-backed models."""
        self._dataloader_config = config

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
        restored = self._safe_unpickle(state["pickled_model"])
        self.__dict__.update(restored.__dict__)

    @staticmethod
    def _safe_unpickle(payload: bytes) -> Any:
        """Unpickle models across platforms by normalizing Path classes."""

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "pathlib" and name in {"WindowsPath", "PosixPath"}:
                    return Path
                return super().find_class(module, name)

        def _load() -> Any:
            return _CompatUnpickler(io.BytesIO(payload)).load()

        try:
            return _load()
        except Exception as exc:  # noqa: BLE001
            if not _is_cuda_deserialize_error(exc):
                raise
            logger.warning(
                "CUDA deserialization error during unpickle; retrying on CPU."
            )
            try:
                import torch
            except Exception:
                raise

            load_from_bytes = getattr(torch.storage, "_load_from_bytes", None)
            if load_from_bytes is None:
                raise

            def _cpu_load_from_bytes(buffer: bytes, *args: Any, **kwargs: Any) -> Any:
                return torch.load(io.BytesIO(buffer), map_location="cpu")

            try:
                torch.storage._load_from_bytes = _cpu_load_from_bytes
                return _load()
            finally:
                torch.storage._load_from_bytes = load_from_bytes
