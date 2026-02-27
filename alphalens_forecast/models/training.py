"""Training helpers for regime-specific baselines."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from alphalens_forecast.config import AppConfig
from alphalens_forecast.core import prepare_features
from alphalens_forecast.data import DataProvider
from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.regime_baselines import (
    ARIMAForecaster,
    ARIMAParams,
    ETSForecaster,
    ETSParams,
    FlatForecaster,
    KalmanForecaster,
    KalmanParams,
    MeanReversionForecaster,
    MeanReversionParams,
    MomentumForecaster,
    MomentumParams,
    OUForecaster,
    OUParams,
)
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.regime_detection.deterministic import (
    REGIME_BREAKOUT,
    REGIME_RANGE,
    REGIME_STRESS_CHOP,
)

logger = logging.getLogger(__name__)

_REGIME_MODEL_PREFIX = {
    REGIME_RANGE: "range",
    REGIME_BREAKOUT: "breakout",
    REGIME_STRESS_CHOP: "stress",
}


def _regime_model_type(regime_label: str, model_choice: str) -> str:
    prefix = _REGIME_MODEL_PREFIX.get(regime_label, "regime")
    safe_choice = "".join(ch for ch in model_choice.lower() if ch.isalnum() or ch in {"_", "-"}).strip("-_")
    return f"regime_{prefix}_{safe_choice}"


def _build_regime_baseline(regime_label: str, choice: str) -> BaseForecaster:
    choice_norm = str(choice or "").strip().lower()
    if regime_label == REGIME_RANGE:
        if choice_norm in {"mean_reversion", "meanreversion", "mr"}:
            return MeanReversionForecaster(params=MeanReversionParams(window=60, half_life=12.0))
        if choice_norm == "arima":
            return ARIMAForecaster(params=ARIMAParams())
        if choice_norm == "ets":
            return ETSForecaster(params=ETSParams())
        if choice_norm == "ou":
            return OUForecaster(params=OUParams())
        return MeanReversionForecaster(params=MeanReversionParams(window=60, half_life=12.0))
    if regime_label == REGIME_BREAKOUT:
        if choice_norm == "momentum":
            return MomentumForecaster(params=MomentumParams(window=25, max_abs_drift=0.03))
        if choice_norm == "arima":
            return ARIMAForecaster(params=ARIMAParams())
        if choice_norm == "ets":
            return ETSForecaster(params=ETSParams())
        if choice_norm == "ou":
            return OUForecaster(params=OUParams())
        return MomentumForecaster(params=MomentumParams(window=25, max_abs_drift=0.03))
    if regime_label == REGIME_STRESS_CHOP:
        if choice_norm == "flat":
            return FlatForecaster()
        if choice_norm == "kalman":
            return KalmanForecaster(params=KalmanParams())
        if choice_norm == "arima":
            return ARIMAForecaster(params=ARIMAParams())
        return FlatForecaster()
    raise ValueError(f"Unknown regime label '{regime_label}'.")


def _regime_choice_from_config(config: AppConfig, regime_label: str) -> str:
    if regime_label == REGIME_RANGE:
        return getattr(config, "regime_range_model", "mean_reversion")
    if regime_label == REGIME_STRESS_CHOP:
        return getattr(config, "regime_stress_model", "flat")
    if regime_label == REGIME_BREAKOUT:
        return getattr(config, "regime_breakout_model", "momentum")
    return "baseline"


def train_regime_models(
    symbol: str,
    timeframe: str,
    regime_label: str,
    *,
    price_frame: Optional[pd.DataFrame] = None,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    config: Optional[AppConfig] = None,
    force_retrain: bool = False,
    model_choice: Optional[str] = None,
) -> BaseForecaster:
    """Train a regime baseline and persist it via ModelRouter."""
    cfg = config or AppConfig()
    provider = data_provider or DataProvider(auto_refresh=False)
    router = model_router or ModelRouter()
    choice = model_choice or _regime_choice_from_config(cfg, regime_label)
    model_type = _regime_model_type(regime_label, choice)

    if not force_retrain:
        try:
            if router.has_model(model_type, symbol, timeframe):
                existing = router.load_model(model_type, symbol, timeframe)
            else:
                existing = None
        except Exception:  # noqa: BLE001
            existing = None
        if isinstance(existing, BaseForecaster):
            logger.info(
                "Regime baseline already exists for %s @ %s (%s); skipping retrain.",
                symbol,
                timeframe,
                model_type,
            )
            return existing

    frame = price_frame if price_frame is not None else provider.load_data(symbol, timeframe)
    features = prepare_features(frame)
    target = features.target

    model = _build_regime_baseline(regime_label, choice)
    model.fit(target)

    metadata = {
        "n_observations": int(len(target)),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "regime": regime_label,
        "choice": choice,
    }
    router.save_model(model_type, symbol, timeframe, model, metadata=metadata)
    logger.info(
        "Trained regime baseline %s for %s @ %s (regime=%s).",
        model_type,
        symbol,
        timeframe,
        regime_label,
    )
    return model


__all__ = ["train_regime_models"]
