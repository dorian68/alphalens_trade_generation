"""Offline training entrypoints for AlphaLens models."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import pandas as pd
from tqdm import tqdm

from alphalens_forecast.core import get_log_returns, prepare_features, prepare_residuals
from alphalens_forecast.data import DataProvider
from alphalens_forecast.models import EGARCHVolModel
from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.config import TrainingConfig
from alphalens_forecast.models.selection import instantiate_model
from alphalens_forecast.training_schedule import TRAINING_FREQUENCIES

logger = logging.getLogger(__name__)


def _resolve_training_device(device: Optional[str]) -> str:
    """Default to TORCH_DEVICE when no explicit device was provided."""
    if device is None or not str(device).strip():
        return os.getenv("TORCH_DEVICE", "cpu")
    return device


class _TrainingProgress:
    """Lightweight tqdm-based spinner to surface model training progress."""

    def __init__(self, description: str, enabled: bool) -> None:
        self._description = description
        self._enabled = enabled
        self._bar: Optional[tqdm] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None

    def __enter__(self) -> "_TrainingProgress":
        if self._enabled:
            self._stop_event = threading.Event()
            self._bar = tqdm(
                total=None,
                desc=self._description,
                leave=False,
                mininterval=0.5,
                dynamic_ncols=True,
                disable=False,
                unit="step",
            )
            self._thread = threading.Thread(target=self._pulse, daemon=True)
            self._thread.start()
        return self

    def _pulse(self) -> None:
        assert self._bar is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            self._bar.update(1)
            time.sleep(0.5)

    def update(self, message: str) -> None:
        if self._enabled and self._bar is not None:
            self._bar.set_postfix_str(message, refresh=True)

    def close(self) -> None:
        if not self._enabled:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._bar is not None:
            self._bar.close()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return False


def _default_provider(provider: Optional[DataProvider]) -> DataProvider:
    if isinstance(provider, DataProvider):
        return provider
    return DataProvider()


def _default_router(router: Optional[ModelRouter]) -> ModelRouter:
    return router or ModelRouter()


def train_mean_model(
    model_type: str,
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    device: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> BaseForecaster:
    """Shared training loop for Prophet/NeuralProphet/NHiTS backends."""
    frame_override = price_frame
    provider_input = data_provider
    resolved_device = _resolve_training_device(device)
    if isinstance(provider_input, pd.DataFrame):
        frame_override = provider_input
        provider_input = None
    provider = _default_provider(provider_input)
    router = _default_router(model_router)
    desc = f"Training {model_type.upper()} [{symbol} @ {timeframe}]"
    with _TrainingProgress(desc, enabled=True) as progress:
        progress.update("Loading price history")
        frame = frame_override if frame_override is not None else provider.load_data(symbol, timeframe)
        progress.update("Preparing features")
        features = prepare_features(frame)
        model = instantiate_model(model_type, device=resolved_device)
        if training_config is not None:
            model.set_dataloader_config(training_config)
        progress.update("Fitting model")
        print(f"Training model...")
        model.fit(features.target, features.regressors)
        print(f"Model trained")
        metadata = {
            "n_observations": len(frame),
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "training_frequency": TRAINING_FREQUENCIES[model_type]["frequency"],
        }
        progress.update("Saving model")
        print(f"saving model...")
        router.save_model(model_type, symbol, timeframe, model, metadata=metadata)
        progress.update("Done")
    print(f"Trained and saved {model_type} model for {symbol} @ {timeframe}")
    return model


def train_nhits(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    device: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> BaseForecaster:
    """Train/persist an N-HiTS mean model."""
    return train_mean_model(
        "nhits",
        symbol,
        timeframe,
        price_frame=price_frame,
        data_provider=data_provider,
        model_router=model_router,
        device=device,
        training_config=training_config,
    )


def train_neuralprophet(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    device: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> BaseForecaster:
    """Train/persist a NeuralProphet mean model."""
    return train_mean_model(
        "neuralprophet",
        symbol,
        timeframe,
        price_frame=price_frame,
        data_provider=data_provider,
        model_router=model_router,
        device=device,
        training_config=training_config,
    )


def train_prophet(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    device: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> BaseForecaster:
    """Train/persist a Prophet mean model."""
    return train_mean_model(
        "prophet",
        symbol,
        timeframe,
        price_frame=price_frame,
        data_provider=data_provider,
        model_router=model_router,
        device=device,
        training_config=training_config,
    )


def train_tft(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    device: Optional[str] = None,
    training_config: Optional[TrainingConfig] = None,
) -> BaseForecaster:
    """Train/persist a TFT sequence model."""
    provider = _default_provider(data_provider)
    router = _default_router(model_router)
    frame = price_frame if price_frame is not None else provider.load_data(symbol, timeframe)
    features = prepare_features(frame)
    resolved_device = _resolve_training_device(device)
    model = instantiate_model("tft", device=resolved_device)
    if training_config is not None:
        model.set_dataloader_config(training_config)
    model.fit(features.target, features.regressors)
    metadata = {
        "n_observations": len(frame),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "training_frequency": TRAINING_FREQUENCIES.get("tft", {}).get("frequency", "unspecified"),
    }
    router.save_model("tft", symbol, timeframe, model, metadata=metadata)
    return model


def train_egarch(
    symbol: str,
    timeframe: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    residuals: Optional[pd.Series] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    show_progress: bool = True,
) -> EGARCHVolModel:
    """
    Train an EGARCH model for the given book.

    Parameters
    ----------
    show_progress:
        When True, surface a tqdm spinner so long-running fits expose their status.
    """
    provider = _default_provider(data_provider)
    router = _default_router(model_router)
    desc = f"Training EGARCH [{symbol} @ {timeframe}]"
    with _TrainingProgress(desc, show_progress) as progress:
        progress.update("Loading price history")
        frame = price_frame if price_frame is not None else provider.load_data(symbol, timeframe)

        progress.update("Preparing residuals")
        if residuals is not None:
            resids = residuals
        else:
            log_returns = get_log_returns(frame)
            resids = prepare_residuals(log_returns)

        progress.update("Fitting EGARCH model")
        model = EGARCHVolModel()
        model.fit(resids)

        progress.update("Persisting checkpoint")
        metadata = {
            "n_observations": len(resids.dropna()),
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "training_frequency": TRAINING_FREQUENCIES["egarch"]["frequency"],
        }
        router.save_egarch(symbol, timeframe, model, metadata=metadata)
        progress.update("Done")
    return model


MEAN_TRAINERS: Dict[str, Callable[..., BaseForecaster]] = {
    "nhits": train_nhits,
    "neuralprophet": train_neuralprophet,
    "prophet": train_prophet,
    "tft": train_tft,
}


__all__ = [
    "MEAN_TRAINERS",
    "train_tft",
    "train_egarch",
    "train_mean_model",
    "train_neuralprophet",
    "train_nhits",
    "train_prophet",
]
