"""Model selection heuristics tied to timeframe."""
from __future__ import annotations

from typing import Tuple, Type

from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.neuralprophet_model import NeuralProphetForecaster
from alphalens_forecast.models.nhits_model import NHiTSForecaster
from alphalens_forecast.models.prophet_model import ProphetForecaster


MODEL_TYPES = {
    "nhits": NHiTSForecaster,
    "neuralprophet": NeuralProphetForecaster,
    "prophet": ProphetForecaster,
}


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
    return "prophet"


def instantiate_model(model_type: str) -> BaseForecaster:
    """Instantiate the forecaster class for the provided type label."""
    try:
        cls: Type[BaseForecaster] = MODEL_TYPES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model type '{model_type}'") from exc
    return cls()


def select_model(timeframe: str) -> Tuple[str, BaseForecaster]:
    """Return both the model type label and an instantiated model."""
    model_type = select_model_type(timeframe)
    return model_type, instantiate_model(model_type)


__all__ = ["instantiate_model", "select_model", "select_model_type", "timeframe_to_minutes", "MODEL_TYPES"]
