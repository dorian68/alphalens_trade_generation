"""Model selection heuristics tied to timeframe."""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Type

from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.neuralprophet_model import NeuralProphetForecaster
from alphalens_forecast.models.nhits_model import NHiTSForecaster
from alphalens_forecast.models.prophet_model import ProphetForecaster
from alphalens_forecast.models.tft_model import TFTForecaster

logger = logging.getLogger(__name__)


MODEL_TYPES = {
    "nhits": NHiTSForecaster,
    "neuralprophet": NeuralProphetForecaster,
    "prophet": ProphetForecaster,
    "tft": TFTForecaster,
}
_TORCH_BACKED = {"nhits", "neuralprophet", "tft"}


def timeframe_to_minutes(timeframe: str) -> int:
    """Translate timeframe strings used by Twelve Data into minutes."""
    units = {
        "min": 1,
        "h": 60,
        "day": 60 * 24,
        "d": 60 * 24,
    }
    timeframe = timeframe.strip().lower()
    for unit, multiplier in units.items():
        if timeframe.endswith(unit):
            value = timeframe[: -len(unit)]
            return int(float(value) * multiplier)
    raise ValueError(f"Unsupported timeframe '{timeframe}'")


def select_model_type(timeframe: str) -> str:
    """Return the canonical model type for the requested timeframe."""
    minutes = timeframe_to_minutes(timeframe)
    if minutes <= 30:
        return "nhits"
    if minutes < 240:
        return "neuralprophet"
    return "nhits"


def resolve_device(device: Optional[str], model_type: str) -> str:
    """Resolve the requested device for Torch-backed models, falling back safely."""
    resolved = (device or "cpu").strip()
    if model_type not in _TORCH_BACKED:
        return resolved
    if resolved.lower().startswith("cuda"):
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA requested but torch import failed; falling back to CPU. (%s)", exc)
            return "cpu"
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
    return resolved


def instantiate_model(model_type: str, device: Optional[str] = None) -> BaseForecaster:
    """Instantiate the forecaster class for the provided type label."""
    try:
        cls: Type[BaseForecaster] = MODEL_TYPES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model type '{model_type}'") from exc
    resolved_device = resolve_device(device, model_type)
    return cls(device=resolved_device)


def select_model(timeframe: str, device: Optional[str] = None) -> Tuple[str, BaseForecaster]:
    """Return both the model type label and an instantiated model."""
    model_type = select_model_type(timeframe)
    return model_type, instantiate_model(model_type, device=device)


__all__ = [
    "instantiate_model",
    "resolve_device",
    "select_model",
    "select_model_type",
    "timeframe_to_minutes",
    "MODEL_TYPES",
]
