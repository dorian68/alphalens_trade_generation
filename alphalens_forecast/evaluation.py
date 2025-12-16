"""Evaluation helpers for the AlphaLens forecasting stack."""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import NormalDist

try:
    from tqdm import TqdmWarning  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover
    TqdmWarning = Warning  # type: ignore[assignment]

from alphalens_forecast.core.feature_engineering import to_neural_prophet_frame
from alphalens_forecast.data import DataProvider
from alphalens_forecast.forecasting import FREQ_MAP
from alphalens_forecast.models import BaseForecaster, EGARCHVolModel, ModelRouter
from alphalens_forecast.models.neuralprophet_model import NeuralProphetForecaster
from alphalens_forecast.models.nhits_model import NHiTSForecaster
from alphalens_forecast.models.prophet_model import ProphetForecaster
from alphalens_forecast.training import (
    train_egarch,
    train_neuralprophet,
    train_nhits,
    train_prophet,
    train_tft,
)
from alphalens_forecast.utils.timeseries import (
    align_series_to_timeframe,
    build_timeseries,
    series_to_dataframe,
    series_to_price_frame,
)

logger = logging.getLogger(__name__)

_TRAINING_REGISTRY: Dict[str, Callable[..., BaseForecaster]] = {
    "nhits": train_nhits,
    "neuralprophet": train_neuralprophet,
    "prophet": train_prophet,
    "tft": train_tft,
}


def _configure_noise_filters() -> None:
    """Silence noisy third-party warnings/logging that overwhelm notebooks."""
    os.environ.setdefault("TQDM_DISABLE", "1")
    warnings.filterwarnings("ignore", category=TqdmWarning)
    warnings.filterwarnings("ignore", module="property_cached")
    warnings.filterwarnings("ignore", message="Importing plotly failed", category=UserWarning)
    warnings.filterwarnings("ignore", message="The provided DatetimeIndex was associated with a timezone")
    warnings.filterwarnings(
        "ignore",
        message=".*timezone.*not supported.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", message="Number of forecast steps is defined by n_forecasts")
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import tqdm.auto as tqdm_auto  # type: ignore

        original_tqdm = getattr(tqdm_auto, "_alphalens_original_tqdm", None) or tqdm_auto.tqdm

        @wraps(original_tqdm)
        def _silent_tqdm(*args, **kwargs):
            kwargs["disable"] = True
            return original_tqdm(*args, **kwargs)

        tqdm_auto._alphalens_original_tqdm = original_tqdm
        tqdm_auto.tqdm = _silent_tqdm
    except Exception:  # pragma: no cover
        pass
    for name in (
        "NP",
        "neuralprophet",
        "pytorch_lightning",
        "lightning.pytorch",
        "lightning.fabric",
        "darts",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False


_configure_noise_filters()


@contextmanager
def _suppress_forecasting_warnings() -> None:
    """Temporarily silence noisy third-party warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def rolling_forecast(
    model: BaseForecaster,
    df_train: pd.Series,
    df_test: pd.Series,
    timeframe: str,
    horizon: int = 1,
    max_steps: Optional[int] = 96,
) -> np.ndarray:
    """Generate a rolling one-step-ahead forecast aligned with ``df_test`` indices."""
    if df_train is None or df_train.empty:
        raise ValueError("Rolling forecasts require non-empty training history.")
    if df_test.empty:
        return np.array([], dtype=float)

    history = df_train.sort_index().copy()
    freq = _resolve_freq(timeframe)
    preds: list[float] = []
    ordered_test = df_test.sort_index()
    limit = len(ordered_test)
    if max_steps is not None:
        limit = min(limit, max_steps)
    with _suppress_forecasting_warnings():
        for idx, (timestamp, actual) in enumerate(ordered_test.items()):
            if idx >= limit:
                break
            preds.append(_forecast_one_step(model, history, timeframe, freq, horizon))
            history.loc[timestamp] = float(actual)
            history = history.sort_index()
    return np.asarray(preds, dtype=float)


def _forecast_one_step(
    model: BaseForecaster,
    history: pd.Series,
    timeframe: str,
    freq: str,
    horizon: int,
) -> float:
    ordered_history = history.sort_index()

    def _regularized(series: pd.Series) -> pd.Series:
        try:
            return align_series_to_timeframe(series, timeframe)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to align history to %s cadence: %s", timeframe, exc)
            return series

    if isinstance(model, NeuralProphetForecaster):
        if model._model is None:
            raise RuntimeError("NeuralProphetForecaster must be fitted before forecasting.")
        aligned = _regularized(ordered_history)
        model._train_frame = to_neural_prophet_frame(aligned)
        model._ensure_trainer_ready(freq)
        future = model._model.make_future_dataframe(
            model._train_frame,
            periods=horizon,
            n_historic_predictions=False,
        )
        forecast = model._model.predict(future)
        column = "yhat1" if "yhat1" in forecast.columns else "yhat"
        predictions = forecast[column].dropna()
        if predictions.empty:
            raise ValueError("NeuralProphet rolling forecast yielded NaNs.")
        return float(predictions.iloc[-1])

    if isinstance(model, NHiTSForecaster):
        if model._model is None:
            raise RuntimeError("NHiTSForecaster must be fitted before forecasting.")
        aligned = _regularized(ordered_history)
        frame = series_to_dataframe(aligned)
        target_series = build_timeseries(frame)
        scaled_series = model._scaler.transform(target_series).astype(np.float32)
        prediction = model._model.predict(n=horizon, series=scaled_series)
        reverted = model._scaler.inverse_transform(prediction)
        forecast_df = reverted.pd_dataframe()
        return float(forecast_df.iloc[-1, 0])

    if isinstance(model, ProphetForecaster):
        if model._model is None:
            raise RuntimeError("ProphetForecaster must be fitted before forecasting.")
        offset = pd.tseries.frequencies.to_offset(freq)
        next_timestamp = ordered_history.index[-1] + offset
        future = pd.DataFrame({"ds": [next_timestamp]})
        forecast = model._model.predict(future)
        return float(forecast["yhat"].iloc[-1])

    forecast_frame = model.forecast(steps=horizon, freq=freq)
    for column in ("yhat", "yhat1"):
        if column in forecast_frame.columns:
            return float(forecast_frame[column].astype(float).iloc[-1])
    raise KeyError("Forecast dataframe missing 'yhat' column.")


def time_split(
    data: pd.Series | pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """
    Split a chronologically ordered series/dataframe into train/validation/test segments.

    Parameters
    ----------
    data:
        Time indexed pandas object.
    train_ratio:
        Fraction of observations assigned to the training window.
    val_ratio:
        Fraction assigned to validation (the rest goes to the test window).
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("time_split requires a DatetimeIndex.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")
    ordered = data.sort_index()
    n_obs = len(ordered)
    if n_obs < 3:
        raise ValueError("Need at least 3 observations to create train/val/test splits.")

    train_size = max(int(np.floor(n_obs * train_ratio)), 1)
    val_size = max(int(np.floor(n_obs * val_ratio)), 1)
    if train_size + val_size >= n_obs:
        val_size = max(1, n_obs - train_size - 1)
    test_size = n_obs - train_size - val_size
    if test_size <= 0 or val_size <= 0:
        raise ValueError("Not enough observations to materialise validation/test splits.")

    train_end = train_size
    val_end = train_end + val_size
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return train, val, test


def _resolve_freq(timeframe: str) -> str:
    key = timeframe.lower()
    try:
        return FREQ_MAP[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe '{timeframe}'.") from exc


def train_model(
    model_name: str,
    symbol: str,
    timeframe: str,
    train_series: pd.Series,
    *,
    model_router: Optional[ModelRouter] = None,
) -> BaseForecaster | EGARCHVolModel:
    """Train and persist the requested model using the supplied training series."""
    if train_series.empty:
        raise ValueError("Training series is empty.")
    router = model_router or ModelRouter()
    normalized_name = model_name.lower()
    frame = series_to_price_frame(train_series)
    if normalized_name == "egarch":
        return train_egarch(symbol, timeframe, price_frame=frame, model_router=router)
    trainer = _TRAINING_REGISTRY.get(normalized_name)
    if trainer is None:
        raise ValueError(f"Unknown model '{model_name}'.")
    return trainer(symbol, timeframe, price_frame=frame, model_router=router)


def load_model(
    model_name: str,
    symbol: str,
    timeframe: str,
    *,
    model_router: Optional[ModelRouter] = None,
) -> BaseForecaster | EGARCHVolModel:
    """Load a previously saved model checkpoint."""
    router = model_router or ModelRouter()
    normalized_name = model_name.lower()
    if normalized_name == "egarch":
        model = router.load_egarch(symbol, timeframe)
    else:
        model = router.load_model(normalized_name, symbol, timeframe)
    if model is None:
        raise FileNotFoundError(
            f"No saved {normalized_name} model found for {symbol} @ {timeframe}."
        )
    return model


def test_model(
    model_name: str,
    model: BaseForecaster | EGARCHVolModel,
    test_series: pd.Series,
    timeframe: str,
    *,
    train_series: Optional[pd.Series] = None,
    rolling_steps: Optional[int] = 96,
) -> pd.Series:
    """Run inference on the test slice and align predictions to its index."""
    if test_series.empty:
        raise ValueError("Test series is empty.")
    steps = len(test_series)
    normalized_name = model_name.lower()
    if normalized_name == "egarch":
        if not isinstance(model, EGARCHVolModel):
            raise TypeError("EGARCH testing requires an EGARCHVolModel instance.")
        with _suppress_forecasting_warnings():
            egarch_forecast = model.forecast(steps)
        predictions = pd.Series(
            egarch_forecast.sigma.to_numpy(dtype=float),
            index=test_series.index,
            name="sigma",
        )
        return predictions

    if not isinstance(model, BaseForecaster):
        raise TypeError(f"{model_name} must inherit from BaseForecaster.")
    if train_series is not None and not train_series.empty:
        try:
            with _suppress_forecasting_warnings():
                rolling_values = rolling_forecast(
                    model,
                    train_series,
                    test_series,
                    timeframe,
                    max_steps=rolling_steps,
                )
            target_index = test_series.sort_index().index[: len(rolling_values)]
            if len(rolling_values) < len(test_series):
                logger.info(
                    "Rolling forecast limited to %s steps (requested %s).",
                    len(rolling_values),
                    rolling_steps,
                )
            return pd.Series(rolling_values, index=target_index, name="prediction")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rolling forecast failed (%s); falling back to bulk forecast.", exc)

    freq = _resolve_freq(timeframe)
    with _suppress_forecasting_warnings():
        forecast_frame = model.forecast(steps=steps, freq=freq)
    if "yhat" in forecast_frame.columns:
        values = forecast_frame["yhat"].to_numpy(dtype=float)
    elif "yhat1" in forecast_frame.columns:
        values = forecast_frame["yhat1"].to_numpy(dtype=float)
    else:
        raise KeyError("Forecast dataframe missing 'yhat' column.")
    if len(values) < steps:
        repeats = int(np.ceil(steps / len(values)))
        values = np.tile(values, repeats)
    return pd.Series(values[:steps], index=test_series.index, name="prediction")


def plot_forecast_vs_real(
    predictions: pd.Series,
    actual: pd.Series,
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show_metrics: bool = True,
    show_confidence: bool = True,
    confidence: float = 0.95,
) -> plt.Axes:
    """Plot model predictions against realised values."""
    axis = ax
    if axis is None:
        _, axis = plt.subplots(figsize=(14, 6))
    aligned_actual = actual.reindex(predictions.index).astype(float)
    axis.plot(actual.index, actual.values, label="Real", linewidth=2)
    axis.plot(predictions.index, predictions.values, label="Forecast", linestyle="--")
    if show_confidence:
        residuals = (aligned_actual - predictions).dropna()
        if not residuals.empty:
            z_score = NormalDist().inv_cdf((1 + confidence) / 2)
            sigma = float(residuals.std(ddof=0))
            upper = predictions + z_score * sigma
            lower = predictions - z_score * sigma
            axis.fill_between(
                predictions.index,
                lower.values,
                upper.values,
                color="tab:blue",
                alpha=0.15,
                label=f"{int(confidence*100)}% CI",
            )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Value")
    if title:
        axis.set_title(title)
    if show_metrics:
        metrics = _regression_metrics(aligned_actual, predictions)
        text = "\n".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items())
        axis.text(
            0.02,
            0.98,
            text,
            transform=axis.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )
    axis.legend()
    if ax is None:
        plt.tight_layout()
    return axis


def _regression_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    diff = actual - predicted
    rmse = float(np.sqrt(np.nanmean(np.square(diff))))
    mae = float(np.nanmean(np.abs(diff)))
    denom = actual.replace(0, np.nan)
    mape = float(np.nanmean(np.abs(diff / denom)) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def evaluate_on_test(
    model_name: str,
    symbol: str,
    timeframe: str,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    data_provider: Optional[DataProvider] = None,
    model_router: Optional[ModelRouter] = None,
    retrain: bool = False,
    plot: bool = True,
    include_vol: bool = False,
    vol_model_name: str = "egarch",
) -> Dict[str, object]:
    """
    High-level helper that trains/loads a model and evaluates it on the test split.

    Returns a dictionary containing the model, predictions, metrics, and raw splits.
    When ``include_vol`` is True, EGARCH evaluation is added alongside the mean model.
    """
    provider = data_provider or DataProvider()
    router = model_router or ModelRouter()
    price_frame = provider.load_data(symbol, timeframe)
    if "close" not in price_frame.columns:
        raise KeyError("Price frame must contain a 'close' column.")
    close_series = price_frame["close"].astype(float).dropna().sort_index()
    train, val, test = time_split(close_series, train_ratio=train_ratio, val_ratio=val_ratio)

    model: BaseForecaster | EGARCHVolModel | None = None
    if not retrain:
        try:
            model = load_model(model_name, symbol, timeframe, model_router=router)
            logger.info(
                "Loaded existing %s model for %s @ %s.",
                model_name,
                symbol,
                timeframe,
            )
        except FileNotFoundError:
            logger.info(
                "No saved %s model was available for %s @ %s; training a new one.",
                model_name,
                symbol,
                timeframe,
            )
    if model is None:
        model = train_model(model_name, symbol, timeframe, train, model_router=router)

    predictions = test_model(model_name, model, test, timeframe, train_series=train)
    actual_for_eval = test
    if model_name.lower() == "egarch":
        actual_for_eval = (
            series_to_price_frame(test)["log_return"]
            .abs()
            .reindex(test.index, fill_value=0.0)
        )

    metrics = _regression_metrics(actual_for_eval, predictions)
    if plot:
        plot_forecast_vs_real(
            predictions,
            actual_for_eval,
            title=f"{symbol} {timeframe} – {model_name.upper()}",
        )

    volatility_summary: Optional[Dict[str, object]] = None
    if include_vol:
        vol_model: EGARCHVolModel | None = None
        vol_name = vol_model_name.lower()
        if vol_name != "egarch":
            raise ValueError("Only 'egarch' volatility models are supported.")
        if not retrain:
            try:
                vol_model = load_model(vol_name, symbol, timeframe, model_router=router)
                logger.info(
                    "Loaded existing %s volatility model for %s @ %s.",
                    vol_name,
                    symbol,
                    timeframe,
                )
            except FileNotFoundError:
                logger.info(
                    "No saved %s volatility model for %s @ %s; training a new one.",
                    vol_name,
                    symbol,
                    timeframe,
                )
        if vol_model is None:
            vol_model = train_model(vol_name, symbol, timeframe, train, model_router=router)  # type: ignore[arg-type]
        vol_predictions = test_model(vol_name, vol_model, test, timeframe, train_series=train)
        vol_actual = (
            series_to_price_frame(test)["log_return"]
            .abs()
            .reindex(test.index, fill_value=0.0)
        )
        vol_metrics = _regression_metrics(vol_actual, vol_predictions)
        if plot:
            plot_forecast_vs_real(
                vol_predictions,
                vol_actual,
                title=f"{symbol} {timeframe} – {vol_name.upper()}",
            )
        volatility_summary = {
            "model": vol_model,
            "predictions": vol_predictions,
            "actual": vol_actual,
            "metrics": vol_metrics,
        }

    result: Dict[str, object] = {
        "model": model,
        "predictions": predictions,
        "actual": actual_for_eval,
        "metrics": metrics,
        "splits": {"train": train, "val": val, "test": test},
    }
    if volatility_summary is not None:
        result["volatility"] = volatility_summary
    return result


def evaluate_mean_and_vol_on_test(
    mean_model_name: str,
    symbol: str,
    timeframe: str,
    **kwargs,
) -> Dict[str, object]:
    """Convenience wrapper that evaluates both mean and EGARCH models on the test split."""
    return evaluate_on_test(
        mean_model_name,
        symbol,
        timeframe,
        include_vol=True,
        **kwargs,
    )


__all__ = [
    "evaluate_on_test",
    "evaluate_mean_and_vol_on_test",
    "load_model",
    "plot_forecast_vs_real",
    "test_model",
    "rolling_forecast",
    "time_split",
    "train_model",
]
