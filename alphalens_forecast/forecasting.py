"""Forecast orchestration components."""
from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from alphalens_forecast.config import AppConfig
from alphalens_forecast.core import (
    HorizonForecast,
    MonteCarloSimulator,
    RiskEngine,
    get_log_returns,
    horizon_to_steps,
    interval_to_hours,
    prepare_features,
    prepare_residuals,
)
from alphalens_forecast.data import DataProvider
from alphalens_forecast.models import (
    BaseForecaster,
    EGARCHForecast,
    EGARCHVolModel,
    ModelRouter,
    NHiTSForecaster,
)
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
from alphalens_forecast.backtesting import TrajectoryRecorder
from alphalens_forecast.models.selection import resolve_device, select_model_type
from alphalens_forecast.training import MEAN_TRAINERS, train_egarch
from alphalens_forecast.utils.model_store import ModelStore, StoredArtifacts
from alphalens_forecast.utils.text import slugify
from alphalens_forecast.utils.timeseries import align_series_to_timeframe, series_to_price_frame
from alphalens_forecast.regime_detection.deterministic import (
    DeterministicRegimeDetector,
    RegimeConfig,
    REGIME_BREAKOUT,
    REGIME_RANGE,
    REGIME_STRESS_CHOP,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
)
from alphalens_forecast.trading.overlays.context_insights_overlay import ContextInsightsOverlay
from alphalens_forecast.trading.overlays.performance_overlay import PerformanceOverlay
from alphalens_forecast.trading.overlays.regime_risk_overlay import RegimeRiskOverlay

logger = logging.getLogger(__name__)

FREQ_MAP: Dict[str, str] = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1h": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1day": "1D",
}


def _trim_to_complete_candle(frame: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], bool]:
    """
    Ensure the last row corresponds to a fully-formed candle.

    Returns the (possibly trimmed) frame, the last complete timestamp, and a flag indicating
    whether a row was dropped. This avoids using an in-progress candle as the live forecast origin.
    """
    if frame.empty:
        return frame, None, False
    freq = FREQ_MAP.get(timeframe.lower())
    if not freq:
        return frame, frame.index[-1], False
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception:  # noqa: BLE001
        return frame, frame.index[-1], False

    last_ts = frame.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(timezone.utc)
    now_utc = datetime.now(timezone.utc)
    # If we are still within the current candle interval, drop the last row to avoid lookahead.
    if now_utc < last_ts + offset:
        trimmed = frame.iloc[:-1].copy()
        if not trimmed.empty:
            return trimmed, trimmed.index[-1], True
        return trimmed, None, True
    return frame, frame.index[-1], False


def compute_dataframe_hash(frame: pd.DataFrame) -> str:
    """Compute a deterministic hash of the price frame."""
    hashed = pd.util.hash_pandas_object(frame, index=True).values
    digest = hashlib.sha256(hashed.tobytes()).hexdigest()
    return digest


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-serialisable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_safe({k: v for k, v in value.__dict__.items() if not k.startswith("_")})
    return str(value)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _regime_blend_weight(
    regime_label: Optional[str],
    regime_confidence: Optional[float],
    config: AppConfig,
    *,
    horizon_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Optional[float]:
    if not getattr(config, "regime_blend_enabled", False):
        return None
    if regime_label is None:
        return None
    label = str(regime_label)
    if label == REGIME_RANGE:
        base = config.regime_blend_range
    elif label == REGIME_BREAKOUT:
        base = config.regime_blend_breakout
    else:
        base = config.regime_blend_trend
    base = _clamp(float(base), 0.0, 1.0)
    horizon_alpha = _clamp(float(getattr(config, "regime_blend_horizon_alpha", 0.0) or 0.0), 0.0, 1.0)
    if horizon_alpha > 0 and horizon_steps is not None and max_steps is not None and max_steps > 0:
        frac = _clamp(float(horizon_steps) / float(max_steps), 0.0, 1.0)
        if label in {REGIME_RANGE, REGIME_STRESS_CHOP}:
            base = base + frac * horizon_alpha * (1.0 - base)
        else:
            base = base * (1.0 - frac * horizon_alpha)
        base = _clamp(float(base), 0.0, 1.0)
    conf_value = 0.5
    if regime_confidence is not None and np.isfinite(regime_confidence):
        conf_value = float(_clamp(float(regime_confidence), 0.0, 1.0))
    alpha = _clamp(float(getattr(config, "regime_blend_confidence_alpha", 0.5)), 0.0, 1.0)
    weight_last = base + (1.0 - conf_value) * alpha * (1.0 - base)
    return float(_clamp(weight_last, 0.0, 1.0))


def _compute_bias_corrections(
    *,
    mean_model: BaseForecaster,
    target_series: pd.Series,
    timeframe: str,
    horizons: Sequence[Tuple[int, int]],
    window: int,
    min_history: int,
) -> Dict[str, float]:
    if window <= 0:
        return {}
    if len(target_series) < window + max(1, min_history):
        return {}
    train_series = target_series.iloc[:-window]
    test_series = target_series.iloc[-window:]
    if train_series.empty or test_series.empty:
        return {}
    try:
        from alphalens_forecast.evaluation import rolling_forecast  # type: ignore
    except Exception:
        return {}
    bias_map: Dict[str, float] = {}
    for horizon_hours, steps in horizons:
        try:
            preds = rolling_forecast(
                mean_model,
                train_series,
                test_series,
                timeframe,
                horizon=steps,
                max_steps=window,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bias correction rolling forecast failed for %sh: %s", horizon_hours, exc)
            continue
        if preds is None or len(preds) == 0:
            continue
        idx = test_series.sort_index().index[: len(preds)]
        actual = pd.to_numeric(test_series.loc[idx], errors="coerce").to_numpy(dtype=float)
        residuals = np.asarray(preds, dtype=float) - actual
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            continue
        bias_map[f"{horizon_hours}h"] = float(np.mean(residuals))
    return bias_map


def summarize_mean_model(model: BaseForecaster) -> Dict[str, Any]:
    """Extract metadata describing the fitted mean model."""
    summary: Dict[str, Any] = {
        "name": getattr(model, "name", model.__class__.__name__),
        "class": model.__class__.__name__,
    }
    backend = getattr(model, "_model", None)
    if backend is not None:
        summary["backend_class"] = backend.__class__.__name__
        params = getattr(backend, "model_params", None) or getattr(backend, "config", None)
        if params is not None:
            summary["hyperparameters"] = _json_safe(params)
        training_losses = getattr(backend, "training_loss", None)
        if training_losses is not None:
            summary["training_loss"] = _json_safe(training_losses)
    return summary


def _load_provider_frame(
    provider: DataProvider,
    symbol: str,
    timeframe: str,
    refresh_data: bool,
) -> pd.DataFrame:
    if refresh_data:
        try:
            return provider.load_data(symbol, timeframe, refresh=True)
        except TypeError:
            logger.debug("DataProvider.load_data lacks refresh support; loading without refresh.")
    return provider.load_data(symbol, timeframe)


def _is_trending_regime(label: str) -> bool:
    return label in {REGIME_TREND_UP, REGIME_TREND_DOWN}


_REGIME_MODEL_PREFIX = {
    REGIME_RANGE: "range",
    REGIME_BREAKOUT: "breakout",
    REGIME_STRESS_CHOP: "stress",
}
_REGIME_GLOBAL_SYMBOL = "global"
_REGIME_GLOBAL_TIMEFRAME = "global"


def _normalize_model_choice(raw_value: Optional[str], default: str) -> str:
    if raw_value is None:
        return default
    value = str(raw_value).strip().lower()
    return value or default


def _regime_model_type(regime_label: str, model_choice: str) -> str:
    prefix = _REGIME_MODEL_PREFIX.get(regime_label, "regime")
    safe_choice = "".join(ch for ch in model_choice.lower() if ch.isalnum() or ch in {"_", "-"}).strip("-_")
    return f"regime_{prefix}_{safe_choice}"


def _tag_regime_model(model: BaseForecaster, choice: str) -> BaseForecaster:
    setattr(model, "_regime_choice", choice)
    return model


def _maybe_load_cached_baseline(
    *,
    config: AppConfig,
    model_router: Optional[ModelRouter],
    symbol: Optional[str],
    timeframe: Optional[str],
    regime_label: str,
    model_choice: str,
) -> Optional[BaseForecaster]:
    if not getattr(config, "regime_baseline_cache", False):
        return None
    if model_router is None:
        return None
    model_type = _regime_model_type(regime_label, model_choice)
    candidates = _regime_cache_candidates(symbol, timeframe, config)
    for cand_symbol, cand_timeframe in candidates:
        try:
            if hasattr(model_router, "has_model") and not model_router.has_model(
                model_type,
                cand_symbol,
                cand_timeframe,
            ):
                continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to check cached regime baseline %s for %s @ %s: %s",
                model_type,
                cand_symbol,
                cand_timeframe,
                exc,
            )
            continue
        try:
            model = model_router.load_model(model_type, cand_symbol, cand_timeframe)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load cached regime baseline %s for %s @ %s: %s",
                model_type,
                cand_symbol,
                cand_timeframe,
                exc,
            )
            continue
        if model is None:
            continue
        if not isinstance(model, BaseForecaster):
            logger.warning(
                "Cached regime baseline %s for %s @ %s is not a BaseForecaster; ignoring.",
                model_type,
                cand_symbol,
                cand_timeframe,
            )
            continue
        setattr(model, "_prefit", True)
        setattr(model, "_regime_choice", model_choice)
        return model
    return None


def _regime_cache_candidates(
    symbol: Optional[str],
    timeframe: Optional[str],
    config: AppConfig,
) -> Sequence[Tuple[str, str]]:
    if not getattr(config, "regime_per_instrument_models", False):
        if symbol is None or timeframe is None:
            return ()
        return ((symbol, timeframe),)
    candidates: list[tuple[str, str]] = []
    if symbol is not None and timeframe is not None:
        candidates.append((symbol, timeframe))
    if timeframe is not None:
        candidates.append((_REGIME_GLOBAL_SYMBOL, timeframe))
    candidates.append((_REGIME_GLOBAL_SYMBOL, _REGIME_GLOBAL_TIMEFRAME))
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for entry in candidates:
        if entry in seen:
            continue
        seen.add(entry)
        ordered.append(entry)
    return tuple(ordered)


def _select_regime_baseline(
    regime_label: str,
    *,
    config: AppConfig,
    model_router: Optional[ModelRouter] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> Optional[BaseForecaster]:
    if regime_label == REGIME_RANGE:
        choice = _normalize_model_choice(getattr(config, "regime_range_model", None), "mean_reversion")
        cached = _maybe_load_cached_baseline(
            config=config,
            model_router=model_router,
            symbol=symbol,
            timeframe=timeframe,
            regime_label=regime_label,
            model_choice=choice,
        )
        if cached is not None:
            return cached
        if choice in {"mean_reversion", "meanreversion", "mr"}:
            return _tag_regime_model(
                MeanReversionForecaster(params=MeanReversionParams(window=60, half_life=12.0)),
                choice,
            )
        if choice == "arima":
            return _tag_regime_model(ARIMAForecaster(params=ARIMAParams()), choice)
        if choice == "ets":
            return _tag_regime_model(ETSForecaster(params=ETSParams()), choice)
        if choice == "ou":
            return _tag_regime_model(OUForecaster(params=OUParams()), choice)
        logger.warning("Unknown RANGE model choice '%s'; using default baseline.", choice)
        return _tag_regime_model(
            MeanReversionForecaster(params=MeanReversionParams(window=60, half_life=12.0)),
            "mean_reversion",
        )
    if regime_label == REGIME_BREAKOUT:
        choice = _normalize_model_choice(getattr(config, "regime_breakout_model", None), "momentum")
        cached = _maybe_load_cached_baseline(
            config=config,
            model_router=model_router,
            symbol=symbol,
            timeframe=timeframe,
            regime_label=regime_label,
            model_choice=choice,
        )
        if cached is not None:
            return cached
        if choice == "momentum":
            return _tag_regime_model(
                MomentumForecaster(params=MomentumParams(window=25, max_abs_drift=0.03)),
                choice,
            )
        if choice == "arima":
            return _tag_regime_model(ARIMAForecaster(params=ARIMAParams()), choice)
        if choice == "ets":
            return _tag_regime_model(ETSForecaster(params=ETSParams()), choice)
        if choice == "ou":
            return _tag_regime_model(OUForecaster(params=OUParams()), choice)
        logger.warning("Unknown BREAKOUT model choice '%s'; using default baseline.", choice)
        return _tag_regime_model(
            MomentumForecaster(params=MomentumParams(window=25, max_abs_drift=0.03)),
            "momentum",
        )
    if regime_label == REGIME_STRESS_CHOP:
        choice = _normalize_model_choice(getattr(config, "regime_stress_model", None), "flat")
        cached = _maybe_load_cached_baseline(
            config=config,
            model_router=model_router,
            symbol=symbol,
            timeframe=timeframe,
            regime_label=regime_label,
            model_choice=choice,
        )
        if cached is not None:
            return cached
        if choice == "flat":
            return _tag_regime_model(FlatForecaster(), choice)
        if choice == "kalman":
            return _tag_regime_model(KalmanForecaster(params=KalmanParams()), choice)
        if choice == "arima":
            return _tag_regime_model(ARIMAForecaster(params=ARIMAParams()), choice)
        logger.warning("Unknown STRESS_CHOP model choice '%s'; using default baseline.", choice)
        return _tag_regime_model(FlatForecaster(), "flat")
    return None


def summarize_garch_model(
    garch: EGARCHVolModel,
    forecast: EGARCHForecast,
) -> Dict[str, Any]:
    """Extract metadata for the EGARCH volatility model."""
    summary: Dict[str, Any] = {
        "class": garch.__class__.__name__,
        "distribution": "Student-t",
        "degrees_of_freedom": float(forecast.dof),
        "skew": float(forecast.skew),
        "forecast_method": getattr(forecast, "method", "unknown"),
    }
    result = getattr(garch, "_result", None)
    if result is not None:
        params = getattr(result, "params", None)
        if params is not None:
            summary["parameters"] = {str(k): float(v) for k, v in params.items()}
        convergence = getattr(result, "convergence", None)
        if convergence is not None:
            summary["converged"] = convergence == 0
    summary["sigma_last"] = float(forecast.sigma.iloc[-1])
    return summary


def make_run_timestamp() -> Tuple[str, str]:
    """Return (iso_timestamp, slug_timestamp) for artifact naming."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    iso = now.isoformat().replace("+00:00", "Z")
    slug = iso.replace("-", "").replace(":", "")
    return iso, slug


def format_timestamp(ts: pd.Timestamp) -> str:
    """Render timestamp in ISO 8601."""
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def build_future_index(last_index: pd.Timestamp, freq: str, steps: int) -> pd.DatetimeIndex:
    """Construct a future datetime index."""
    start = last_index + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=start, periods=steps, freq=freq)


def compute_student_t_quantiles(
    median_log: float,
    sigma: float,
    dof: float,
) -> Dict[str, float]:
    """Compute price quantiles assuming Student-t innovations on log returns."""
    from scipy.stats import t

    dist = t(df=dof)
    q20_log = median_log + dist.ppf(0.20) * sigma
    q50_log = median_log
    q80_log = median_log + dist.ppf(0.80) * sigma
    return {
        "p20": float(np.exp(q20_log)),
        "p50": float(np.exp(q50_log)),
        "p80": float(np.exp(q80_log)),
    }


@dataclass
class OrchestrationResult:
    """Collect outputs and metadata from a forecasting run."""

    payload: Dict[str, Any]
    price_frame: pd.DataFrame
    residuals: pd.Series
    mean_model: Optional[BaseForecaster]
    vol_model: Optional[EGARCHVolModel]
    garch_forecast: Optional[EGARCHForecast]
    metadata: Dict[str, Any]
    predictions: Dict[str, pd.DataFrame] = field(default_factory=dict)
    data_hash: Optional[str] = None
    as_of: Optional[str] = None
    used_cached_artifacts: bool = False
    durations: Dict[str, float] = field(default_factory=dict)
    run_timestamp_iso: Optional[str] = None
    run_timestamp_slug: Optional[str] = None
    trajectories: List[Dict[str, Any]] = field(default_factory=list)
    rolling_predictions: Dict[str, pd.Series] = field(default_factory=dict)

    @property
    def volatility(self) -> Optional[EGARCHForecast]:
        """Backwards-compatible alias for volatility forecasts."""
        return self.garch_forecast


class _SeriesDataProvider:
    """Minimal provider that serves a pre-loaded price frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        del symbol, timeframe
        return self._frame.copy()


class ForecastEngine:
    """High-level forecast orchestrator used by the CLI and services."""

    def __init__(
        self,
        config: AppConfig,
        data_provider: DataProvider,
        model_router: ModelRouter,
    ) -> None:
        self._config = config
        self._data_provider = data_provider
        self._model_router = model_router

    def forecast(
        self,
        *,
        symbol: str,
        timeframe: str,
        horizons: Iterable[int],
        paths: int,
        use_montecarlo: bool,
        trade_mode: str = "spot",
        reuse_cached: bool = False,
        model_store: Optional[ModelStore] = None,
        show_progress: bool = True,
        trajectory_recorder: Optional[TrajectoryRecorder] = None,
        price_frame: Optional[pd.DataFrame] = None,
        mean_model_override: Optional[BaseForecaster] = None,
        vol_model_override: Optional[EGARCHVolModel] = None,
        rolling_eval: bool = False,
        rolling_steps: Optional[int] = None,
        force_retrain: bool = False,
        refresh_data: bool = False,
        execution_price: Optional[float] = None,
        execution_price_source: Optional[str] = None,
        enable_regime_switching: Optional[bool] = None,
        regime_lookback: Optional[int] = None,
        enable_performance_patches: Optional[bool] = None,
    ) -> OrchestrationResult:
        """Run the full pipeline: data -> mean model -> vol -> Monte Carlo."""
        run_start = time.perf_counter()
        run_timestamp_iso, run_timestamp_slug = make_run_timestamp()
        trade_mode_normalized = str(trade_mode or "spot").strip().lower()
        if trade_mode_normalized not in {"spot", "forward"}:
            raise ValueError("trade_mode must be one of: spot, forward")

        logger.info(
            "Starting forecast for %s [%s] | horizons=%s",
            symbol,
            timeframe,
            ",".join(str(h) for h in horizons),
        )

        durations: Dict[str, float] = {}
        fetch_start = time.perf_counter()
        frame_override = price_frame.copy() if price_frame is not None else None
        if frame_override is None:
            price_frame = _load_provider_frame(self._data_provider, symbol, timeframe, refresh_data)
            source_label = "fetched"
        else:
            price_frame = frame_override
            source_label = "provided"
        price_frame, last_complete_ts, dropped_tail = _trim_to_complete_candle(price_frame, timeframe)
        if dropped_tail:
            logger.info(
                "Dropped in-progress candle for %s @ %s; using %s as last complete bar.",
                symbol,
                timeframe,
                format_timestamp(last_complete_ts) if last_complete_ts is not None else "n/a",
            )
        if price_frame.empty:
            raise ValueError(f"No complete candles available for {symbol} @ {timeframe}.")
        durations["fetch_seconds"] = time.perf_counter() - fetch_start
        logger.info(
            "%s %d rows in %.2fs | first=%s | last=%s",
            source_label.capitalize(),
            len(price_frame),
            durations["fetch_seconds"],
            format_timestamp(price_frame.index[0]),
            format_timestamp(price_frame.index[-1]),
        )

        switching_enabled = (
            self._config.regime_switching if enable_regime_switching is None else enable_regime_switching
        )
        performance_patches_enabled = (
            self._config.performance_patches if enable_performance_patches is None else enable_performance_patches
        )
        regime_lookback_value = regime_lookback or self._config.regime_lookback
        regime_label: Optional[str] = None
        regime_confidence: Optional[float] = None
        regime_mode_used: Optional[str] = None
        regime_scores: Optional[Dict[str, float]] = None
        regime_is_trend = True
        if switching_enabled:
            try:
                detector = DeterministicRegimeDetector(
                    RegimeConfig(
                        lookback=regime_lookback_value,
                        enable_walk_forward_calib=self._config.regime_walk_forward_calib,
                        calib_window=self._config.regime_calib_window,
                        calib_min_history=self._config.regime_calib_min_history,
                        calib_trend_quantile=self._config.regime_calib_trend_quantile,
                        calib_range_quantile=self._config.regime_calib_range_quantile,
                        calib_chop_quantile=self._config.regime_calib_chop_quantile,
                        calib_vol_expansion_quantile=self._config.regime_calib_vol_expansion_quantile,
                        enable_score_smoothing=self._config.regime_score_smoothing,
                        score_smoothing_alpha=self._config.regime_score_smoothing_alpha,
                        enable_vol_mom_confirm=self._config.regime_vol_mom_confirm,
                        volume_window=self._config.regime_volume_window,
                        volume_z_threshold=self._config.regime_volume_z_threshold,
                        volume_ratio_threshold=self._config.regime_volume_ratio_threshold,
                        momentum_window=self._config.regime_momentum_window,
                        momentum_threshold=self._config.regime_momentum_threshold,
                        regime_mode=self._config.regime_mode,
                    )
                )
                regime_result = detector.detect(price_frame)
                regime_label = regime_result.regime
                regime_confidence = regime_result.confidence
                regime_scores = regime_result.scores or None
                regime_mode_used = None
                try:
                    regime_mode_used = (regime_result.meta or {}).get("mode_used")
                except Exception:
                    regime_mode_used = None
                regime_is_trend = _is_trending_regime(regime_label)
                logger.info(
                    "Regime detected: %s (confidence=%.3f) | route=%s",
                    regime_label,
                    regime_confidence,
                    "trend" if regime_is_trend else "alternate",
                )
            except Exception as exc:  # noqa: BLE001
                switching_enabled = False
                regime_is_trend = True
                if logger.isEnabledFor(logging.DEBUG):
                    required = ["open", "high", "low", "close"]
                    missing = [col for col in required if col not in price_frame.columns]
                    nan_counts = {}
                    try:
                        if not missing:
                            nan_counts = price_frame[required].isna().sum().to_dict()
                    except Exception:
                        nan_counts = {}
                    logger.debug(
                        "Regime detection failed for %s [%s] | rows=%d cols=%s missing=%s nan_counts=%s "
                        "index_range=%s..%s | error=%s",
                        symbol,
                        timeframe,
                        len(price_frame) if price_frame is not None else -1,
                        list(price_frame.columns) if hasattr(price_frame, "columns") else None,
                        missing,
                        nan_counts,
                        format_timestamp(price_frame.index.min()) if hasattr(price_frame, "index") and len(price_frame) else None,
                        format_timestamp(price_frame.index.max()) if hasattr(price_frame, "index") and len(price_frame) else None,
                        exc,
                    )
                logger.warning("Regime detection failed; using default pipeline. (%s)", exc)

        data_hash = compute_dataframe_hash(price_frame)
        logger.debug("Price frame hash: %s", data_hash)
        if (
            model_store
            and reuse_cached
            and frame_override is None
            and not force_retrain
            and execution_price is None
            and (not switching_enabled or regime_is_trend)
        ):
            reused = self._reuse_from_store(
                model_store=model_store,
                symbol=symbol,
                timeframe=timeframe,
                data_hash=data_hash,
                price_frame=price_frame,
                durations=durations,
                run_timestamp_iso=run_timestamp_iso,
                run_timestamp_slug=run_timestamp_slug,
            )
            if reused is not None:
                if switching_enabled:
                    updated_meta = dict(reused.metadata) if isinstance(reused.metadata, dict) else {}
                    updated_meta["regime"] = {
                        "enabled": switching_enabled,
                        "label": regime_label,
                        "confidence": regime_confidence,
                        "lookback": regime_lookback_value,
                        "route": "trend" if regime_is_trend else "alternate",
                        "mode_used": regime_mode_used,
                    }
                    reused.metadata = updated_meta
                overlay_context = {
                    "regime_enabled": switching_enabled,
                    "regime_label": regime_label,
                    "regime_confidence": regime_confidence,
                    "regime_route": "trend" if regime_is_trend else "alternate",
                    "entry_model_vol_by_horizon": {},
                    "vol_ref": None,
                    "performance_patches_enabled": performance_patches_enabled,
                    "price_frame": price_frame,
                }
                reused.payload = RegimeRiskOverlay().apply(reused.payload or {}, overlay_context)
                reused.payload = PerformanceOverlay().apply(reused.payload or {}, overlay_context)
                return reused

        features = prepare_features(price_frame)
        target_series = features.target
        last_price = float(target_series.iloc[-1])
        if last_price <= 0:
            raise ValueError("Close price must be positive to compute log transforms.")
        # Modeling price uses the last close; execution price can be live when provided.
        execution_price_value = last_price
        execution_price_source_value = "close"
        execution_price_override: Optional[float] = None
        if execution_price is not None:
            try:
                execution_price_value = float(execution_price)
            except (TypeError, ValueError) as exc:
                raise ValueError("execution_price must be positive and finite.") from exc
            if not np.isfinite(execution_price_value) or execution_price_value <= 0:
                raise ValueError("execution_price must be positive and finite.")
            execution_price_source_value = (
                execution_price_source if execution_price_source in {"client", "live"} else "client"
            )
            execution_price_override = execution_price_value
        last_log_price = float(np.log(last_price))
        as_of_ts = last_complete_ts or price_frame.index[-1]
        as_of = format_timestamp(as_of_ts)
        spot_entry_price = execution_price_value if trade_mode_normalized == "spot" else last_price
        if trade_mode_normalized == "spot" and use_montecarlo:
            quantiles_anchor = spot_entry_price
        else:
            quantiles_anchor = last_price

        freq = FREQ_MAP.get(timeframe.lower())
        if freq is None:
            raise ValueError(f"No pandas frequency mapping for timeframe '{timeframe}'.")

        regime_fit_seconds: Optional[float] = None
        if switching_enabled and not regime_is_trend and regime_label is not None:
            baseline_model = _select_regime_baseline(
                regime_label,
                config=self._config,
                model_router=self._model_router,
                symbol=symbol,
                timeframe=timeframe,
            )
            if baseline_model is None:
                logger.warning(
                    "Regime routing enabled but no baseline available for %s; using default pipeline.",
                    regime_label,
                )
            else:
                try:
                    if not getattr(baseline_model, "_prefit", False):
                        fit_start = time.perf_counter()
                        baseline_model.fit(target_series)
                        regime_fit_seconds = time.perf_counter() - fit_start
                        if getattr(self._config, "regime_baseline_cache", False):
                            model_choice = getattr(baseline_model, "_regime_choice", "baseline")
                            model_type = _regime_model_type(regime_label, str(model_choice))
                            try:
                                self._model_router.save_model(
                                    model_type,
                                    symbol,
                                    timeframe,
                                    baseline_model,
                                    metadata={"regime": regime_label, "choice": model_choice},
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "Failed to cache regime baseline %s for %s @ %s: %s",
                                    model_type,
                                    symbol,
                                    timeframe,
                                    exc,
                                )
                    else:
                        regime_fit_seconds = 0.0
                    mean_model_override = baseline_model
                    logger.info(
                        "Regime routing enabled: using %s for %s [%s] (regime=%s).",
                        baseline_model.name,
                        symbol,
                        timeframe,
                        regime_label,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Regime baseline %s failed for %s @ %s; using default pipeline. (%s)",
                        getattr(baseline_model, "name", "baseline"),
                        symbol,
                        timeframe,
                        exc,
                    )
                    baseline_model = None

        model_type = select_model_type(timeframe)
        device = resolve_device(self._config.inference_device, model_type)
        mean_model = mean_model_override
        if mean_model is not None:
            durations["mean_model_fit_seconds"] = regime_fit_seconds or 0.0
            logger.info(
                "Using provided mean model (%s) for %s [%s]",
                getattr(mean_model, "name", mean_model.__class__.__name__),
                symbol,
                timeframe,
            )
        else:
            if not force_retrain:
                try:
                    mean_model = self._model_router.load_model(
                        model_type,
                        symbol,
                        timeframe,
                        device=device,
                    )
                except FileNotFoundError:
                    mean_model = None
                if isinstance(mean_model, NHiTSForecaster) and mean_model.requires_retraining():
                    logger.warning(
                        "Cached NHITS model for %s @ %s was trained with covariates; retraining.",
                        symbol,
                        timeframe,
                    )
                    mean_model = None
            if mean_model is None:
                trainer = MEAN_TRAINERS[model_type]
                if force_retrain:
                    logger.info("Force retrain enabled; training %s model for %s [%s]", model_type, symbol, timeframe)
                else:
                    logger.info("Training %s model for %s [%s]", model_type, symbol, timeframe)
                fit_start = time.perf_counter()
                mean_model = trainer(
                    symbol,
                    timeframe,
                    price_frame=price_frame,
                    data_provider=self._data_provider,
                    model_router=self._model_router,
                    device=device,
                    training_config=self._config.training,
                    refresh_data=refresh_data,
                )
                durations["mean_model_fit_seconds"] = time.perf_counter() - fit_start
            else:
                durations["mean_model_fit_seconds"] = 0.0
                logger.info("Loaded cached %s model for %s [%s]", model_type, symbol, timeframe)

        log_returns = get_log_returns(price_frame)
        residuals = prepare_residuals(log_returns)
        garch = vol_model_override
        if garch is not None:
            durations["garch_fit_seconds"] = 0.0
            logger.info("Using provided volatility model for %s [%s]", symbol, timeframe)
        else:
            if not force_retrain:
                try:
                    garch = self._model_router.load_egarch(symbol, timeframe)
                except FileNotFoundError:
                    garch = None
            if garch is None:
                if force_retrain:
                    logger.info("Force retrain enabled; training EGARCH model for %s [%s]", symbol, timeframe)
                else:
                    logger.info("Training EGARCH model for %s [%s]", symbol, timeframe)
                garch_fit_start = time.perf_counter()
                garch = train_egarch(
                    symbol,
                    timeframe,
                    price_frame=price_frame,
                    residuals=residuals,
                    data_provider=self._data_provider,
                    model_router=self._model_router,
                    show_progress=show_progress,
                    refresh_data=refresh_data,
                )
                durations["garch_fit_seconds"] = time.perf_counter() - garch_fit_start
            else:
                durations["garch_fit_seconds"] = 0.0
                logger.info("Loaded cached EGARCH model for %s [%s]", symbol, timeframe)

        horizon_steps = [horizon_to_steps(h, timeframe) for h in horizons]
        max_steps = max(horizon_steps)
        garch_forecast_start = time.perf_counter()
        garch_forecast: EGARCHForecast = garch.forecast(max_steps)
        durations["garch_forecast_seconds"] = time.perf_counter() - garch_forecast_start
        sigma_path = garch_forecast.sigma
        variance_path = garch_forecast.variance
        forecast_method = getattr(garch_forecast, "method", "unknown")
        mc_skew = float(garch_forecast.skew)
        logger.info(
            "EGARCH forecast produced in %.2fs | method=%s | sigma_range=(%.6f, %.6f)",
            durations["garch_forecast_seconds"],
            forecast_method,
            float(sigma_path.min()),
            float(sigma_path.max()),
        )

        mc_simulator = (
            MonteCarloSimulator(
                paths=paths,
                seed=self._config.monte_carlo.seed,
                show_progress=use_montecarlo,
            )
            if use_montecarlo
            else None
        )
        if use_montecarlo:
            logger.info("Monte Carlo enabled | paths=%d seed=%s", paths, self._config.monte_carlo.seed)
        else:
            logger.info("Monte Carlo disabled")
        forward_mc_simulator = None
        if use_montecarlo and trade_mode_normalized == "forward":
            forward_mc_simulator = MonteCarloSimulator(
                paths=paths,
                seed=self._config.monte_carlo.seed,
                show_progress=False,
            )

        risk_engine = RiskEngine(self._config)
        horizon_payload: List[HorizonForecast] = []
        mean_forecasts: Dict[str, pd.DataFrame] = {}
        rolling_predictions: Dict[str, pd.Series] = {}
        rolling_summary: Dict[str, Any] = {}
        horizon_iterable = list(zip(horizons, horizon_steps))
        if rolling_eval:
            steps_eval = rolling_steps or 0
            if steps_eval <= 0 or steps_eval >= len(target_series):
                logger.warning(
                    "Rolling evaluation skipped (requested steps=%s, available history=%s).",
                    steps_eval,
                    len(target_series),
                )
            else:
                try:
                    # Imported lazily to avoid circular imports at module load time.
                    from alphalens_forecast.evaluation import rolling_forecast  # type: ignore

                    train_series = target_series.iloc[:-steps_eval]
                    test_series = target_series.iloc[-steps_eval:]
                    for horizon_hours, steps in horizon_iterable:
                        horizon_label = f"{horizon_hours}h"
                        roll_values = rolling_forecast(
                            mean_model,
                            train_series,
                            test_series,
                            timeframe,
                            horizon=steps,
                            max_steps=steps_eval,
                        )
                        target_index = test_series.sort_index().index[: len(roll_values)]
                        rolling_predictions[horizon_label] = pd.Series(
                            roll_values, index=target_index, name="prediction"
                        )
                    rolling_summary = {
                        "enabled": True,
                        "steps": steps_eval,
                        "train_rows": len(train_series),
                        "test_rows": len(test_series),
                        "horizons": [f"{h}h" for h, _ in horizon_iterable],
                    }
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Rolling evaluation failed; continuing with bulk forecast. (%s)", exc)
        logger.info("Processing %d horizons", len(horizon_iterable))
        bias_corrections: Dict[str, float] = {}
        if getattr(self._config, "bias_correction_enabled", False):
            bias_window = int(getattr(self._config, "bias_correction_window", 0) or 0)
            bias_min_history = int(getattr(self._config, "bias_correction_min_history", 0) or 0)
            if bias_window > 0:
                bias_corrections = _compute_bias_corrections(
                    mean_model=mean_model,
                    target_series=target_series,
                    timeframe=timeframe,
                    horizons=horizon_iterable,
                    window=bias_window,
                    min_history=bias_min_history,
                )
        forecast_loop_start = time.perf_counter()
        for horizon_hours, steps in tqdm(
            horizon_iterable,
            desc="Processing horizons",
            disable=not show_progress,
        ):
            forecast_df = mean_model.forecast(
                steps=steps,
                freq=freq,
                future_regressors=None,
            )

            if "ds" in forecast_df.columns:
                forecast_df = forecast_df.set_index("ds")

            horizon_label = f"{horizon_hours}h"
            horizon_variance = variance_path.iloc[:steps].to_numpy(dtype=float)
            sigma_per_step = sigma_path.iloc[:steps].to_numpy(dtype=float)
            if not np.isfinite(horizon_variance).all() or not np.isfinite(sigma_per_step).all():
                logger.error("Invalid EGARCH path detected for horizon %s.", horizon_hours)
                raise RuntimeError("EGARCH provided invalid variance or sigma values.")
            sigma_h = float(np.sqrt(np.sum(horizon_variance)))
            bias_value = bias_corrections.get(horizon_label)
            if bias_value is not None and np.isfinite(bias_value):
                sigma_clip = float(getattr(self._config, "bias_correction_sigma_clip", 0.0) or 0.0)
                if sigma_clip > 0 and np.isfinite(sigma_h) and sigma_h > 0:
                    bias_value = float(_clamp(float(bias_value), -sigma_clip * sigma_h, sigma_clip * sigma_h))
                adjusted = forecast_df["yhat"].astype(float) - float(bias_value)
                if float(adjusted.iloc[-1]) > 0:
                    forecast_df = forecast_df.copy()
                    forecast_df["yhat"] = adjusted
                else:
                    logger.debug("Skipping bias correction for %s due to non-positive yhat.", horizon_label)

            blend_weight = _regime_blend_weight(
                regime_label,
                regime_confidence,
                self._config,
                horizon_steps=steps,
                max_steps=max_steps,
            )
            if blend_weight is not None and 0.0 < blend_weight < 1.0:
                forecast_df = forecast_df.copy()
                forecast_df["yhat"] = (
                    blend_weight * last_price + (1.0 - blend_weight) * forecast_df["yhat"].astype(float)
                )

            mean_forecasts[horizon_label] = forecast_df.copy()
            if trajectory_recorder is not None:
                trajectory_recorder.add_from_dataframe(
                    horizon_label=horizon_label,
                    forecast_df=forecast_df,
                )

            yhat_series = forecast_df["yhat"].astype(float)
            if not np.isfinite(yhat_series.values).all():
                logger.error("Divergent %s prediction detected; contains NaN/Inf.", mean_model.name)
                raise RuntimeError(f"{mean_model.name} produced invalid forecasts.")
            median_price_estimate = float(forecast_df["yhat"].iloc[-1])
            if median_price_estimate <= 0:
                raise ValueError("Mean model returned a non-positive price forecast.")
            median_log = float(np.log(median_price_estimate))
            drift = (median_log - last_log_price) / steps

            quantiles = compute_student_t_quantiles(
                median_log=median_log,
                sigma=sigma_h,
                dof=garch_forecast.dof,
            )
            median_price = quantiles["p50"]
            p20_price = quantiles["p20"]
            p80_price = quantiles["p80"]

            direction = "long" if median_price >= last_price else "short"
            tp_level = p80_price if direction == "long" else p20_price
            sl_level = p20_price if direction == "long" else p80_price
            if trade_mode_normalized == "spot" and execution_price_override is not None:
                scale = execution_price_value / last_price
                tp_level *= scale
                sl_level *= scale

            probability = None
            if use_montecarlo and mc_simulator is not None:
                mc_start = time.perf_counter()
                step_hours = interval_to_hours(timeframe)
                mc_result = mc_simulator.simulate(
                    current_price=spot_entry_price,
                    drift=drift,
                    sigma=sigma_per_step,
                    dof=garch_forecast.dof,
                    skew=mc_skew,
                    tp=tp_level,
                    sl=sl_level,
                    steps=steps,
                    step_hours=step_hours,
                )
                mc_duration = time.perf_counter() - mc_start
                durations.setdefault("monte_carlo_seconds", 0.0)
                durations["monte_carlo_seconds"] += mc_duration
                probability = mc_result.probability_hit_tp_before_sl
                p20_price = mc_result.quantiles["p20"]
                median_price = mc_result.quantiles["p50"]
                p80_price = mc_result.quantiles["p80"]
                logger.debug(
                    "Monte Carlo horizon=%sh completed in %.2fs | prob=%.4f",
                    horizon_hours,
                    mc_duration,
                    probability,
                )
                if trade_mode_normalized == "forward" and forward_mc_simulator is not None:
                    forward_mc_start = time.perf_counter()
                    entry_price = median_price
                    forward_result = forward_mc_simulator.simulate(
                        current_price=entry_price,
                        drift=drift,
                        sigma=sigma_per_step,
                        dof=garch_forecast.dof,
                        skew=mc_skew,
                        tp=tp_level,
                        sl=sl_level,
                        steps=steps,
                        step_hours=step_hours,
                    )
                    forward_duration = time.perf_counter() - forward_mc_start
                    durations.setdefault("monte_carlo_seconds", 0.0)
                    durations["monte_carlo_seconds"] += forward_duration
                    probability = forward_result.probability_hit_tp_before_sl

            horizon_payload.append(
                HorizonForecast(
                    horizon_label=f"{horizon_hours}h",
                    median=median_price,
                    p20=p20_price,
                    p80=p80_price,
                    sigma=sigma_h,
                    dof=garch_forecast.dof,
                    drift=drift,
                    model_name=mean_model.name,
                    vol_model_name="EGARCH_t",
                    calibrated=True,
                    probability_hit_tp_before_sl=probability,
                    last_price=last_price,
                    execution_price=execution_price_override,
                    quantiles_anchor=quantiles_anchor,
                )
            )
        durations["forecast_loop_seconds"] = time.perf_counter() - forecast_loop_start
        logger.info(
            "Processed horizons in %.2fs | horizons=%d",
            durations["forecast_loop_seconds"],
            len(horizon_payload),
        )

        result_payload = risk_engine.build(
            symbol=symbol,
            as_of=as_of,
            timeframe=timeframe,
            horizons=horizon_payload,
            use_montecarlo=use_montecarlo,
            trade_mode=trade_mode_normalized,
        )
        sigma_by_horizon = {horizon.horizon_label: float(horizon.sigma) for horizon in horizon_payload}
        vol_ref = None
        if sigma_by_horizon:
            sigma_values = [value for value in sigma_by_horizon.values() if np.isfinite(value)]
            if sigma_values:
                vol_ref = float(np.median(sigma_values))
        overlay_context = {
            "regime_enabled": switching_enabled,
            "regime_label": regime_label,
            "regime_confidence": regime_confidence,
            "regime_scores": regime_scores,
            "regime_mode_used": regime_mode_used,
            "regime_route": "trend" if regime_is_trend else "alternate",
            "entry_model_vol_by_horizon": sigma_by_horizon,
            "vol_ref": vol_ref,
            "performance_patches_enabled": performance_patches_enabled,
            "price_frame": price_frame,
        }
        if logger.isEnabledFor(logging.DEBUG) and regime_label is None:
            logger.debug(
                "Trade overlay: missing regime_label for %s [%s] | enabled=%s rows=%d",
                symbol,
                timeframe,
                switching_enabled,
                len(price_frame) if price_frame is not None else -1,
            )
        result_payload = RegimeRiskOverlay().apply(result_payload, overlay_context)
        result_payload = PerformanceOverlay().apply(result_payload, overlay_context)
        result_payload = ContextInsightsOverlay().apply(result_payload, overlay_context)
        if trajectory_recorder is not None:
            result_payload["trajectories"] = trajectory_recorder.to_payload()

        durations["total_seconds"] = time.perf_counter() - run_start
        logger.info(
            "Forecast pipeline completed in %.2fs for %s [%s]",
            durations["total_seconds"],
            symbol,
            timeframe,
        )

        metadata: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizons": list(horizons),
            "timestamp": run_timestamp_iso,
            "timestamp_slug": run_timestamp_slug,
            "as_of": as_of,
            "n_observations": len(price_frame),
            "data_hash": data_hash,
            "use_montecarlo": use_montecarlo,
            "monte_carlo_paths": paths if use_montecarlo else 0,
            "durations": durations,
            "mean_model": summarize_mean_model(mean_model),
            "vol_model": summarize_garch_model(garch, garch_forecast),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
        }
        metadata["regime"] = {
            "enabled": switching_enabled,
            "label": regime_label,
            "confidence": regime_confidence,
            "lookback": regime_lookback_value,
            "route": "trend" if regime_is_trend else "alternate",
            "mode_used": regime_mode_used,
        }
        if rolling_summary:
            metadata["rolling_eval"] = rolling_summary
        metadata["residual_std"] = float(residuals.std(ddof=0))
        metadata["last_price"] = last_price
        metadata["execution_price"] = execution_price_value
        metadata["price_modeling"] = last_price
        metadata["price_execution"] = execution_price_value
        metadata["execution_price_source"] = execution_price_source_value
        metadata["sigma_path_min"] = float(sigma_path.min())
        metadata["sigma_path_max"] = float(sigma_path.max())

        return OrchestrationResult(
            payload=result_payload,
            price_frame=price_frame,
            residuals=residuals,
            mean_model=mean_model,
            vol_model=garch,
            garch_forecast=garch_forecast,
            metadata=metadata,
            predictions=mean_forecasts,
            data_hash=data_hash,
            as_of=as_of,
            durations=durations,
            run_timestamp_iso=run_timestamp_iso,
            run_timestamp_slug=run_timestamp_slug,
            trajectories=trajectory_recorder.to_payload() if trajectory_recorder is not None else [],
            rolling_predictions=rolling_predictions,
        )

    def _reuse_from_store(
        self,
        *,
        model_store: ModelStore,
        symbol: str,
        timeframe: str,
        data_hash: str,
        price_frame: pd.DataFrame,
        durations: Dict[str, float],
        run_timestamp_iso: str,
        run_timestamp_slug: str,
    ) -> Optional[OrchestrationResult]:
        symbol_slug = slugify(symbol)
        timeframe_slug = slugify(timeframe)
        stored: Optional[StoredArtifacts] = model_store.load_latest(symbol_slug, timeframe_slug)
        if not stored:
            return None
        stored_hash = stored.metadata.get("data_hash")
        if stored_hash != data_hash:
            logger.warning(
                "Saved model hash %s does not match current data hash %s; retraining.",
                stored_hash,
                data_hash,
            )
            return None
        logger.info("Reusing cached artifacts for %s [%s] (hash=%s)", symbol, timeframe, data_hash)
        reuse_durations = dict(stored.metadata.get("durations", {}))
        reuse_durations["fetch_seconds"] = durations["fetch_seconds"]
        return OrchestrationResult(
            payload=stored.payload or {},
            price_frame=price_frame,
            residuals=pd.Series(dtype=float),
            mean_model=stored.mean_model,
            vol_model=stored.vol_model,
            garch_forecast=None,
            metadata=stored.metadata,
            predictions={},
            data_hash=data_hash,
            as_of=stored.metadata.get("as_of", format_timestamp(price_frame.index[-1])),
            used_cached_artifacts=True,
            durations=reuse_durations,
            run_timestamp_iso=stored.metadata.get("timestamp", run_timestamp_iso),
            run_timestamp_slug=stored.metadata.get("timestamp_slug", run_timestamp_slug),
            rolling_predictions={},
        )


def forecast_from_series(
    series: pd.Series,
    *,
    model: BaseForecaster,
    timeframe: str,
    horizons: Iterable[int],
    symbol: str = "BTC/USD",
    config: Optional[AppConfig] = None,
    vol_model: Optional[EGARCHVolModel] = None,
    use_montecarlo: Optional[bool] = None,
    paths: Optional[int] = None,
    fit_model: bool = True,
    show_progress: bool = False,
) -> OrchestrationResult:
    """
    Run a forecast using an explicit price series and pre-selected model.

    This helper is intended for near real-time testing/backtesting loops where you
    already loaded the price history (e.g., from a CSV) and want to probe how a
    particular mean model behaves when fed the last ``n`` points. Pass ``fit_model``
    to control whether the helper should call ``model.fit`` before forecasting.
    """
    if series.empty:
        raise ValueError("Price series must contain at least one observation.")

    resolved_config = config or AppConfig()
    resolved_paths = paths if paths is not None else resolved_config.monte_carlo.paths
    resolved_use_mc = use_montecarlo if use_montecarlo is not None else resolved_config.monte_carlo.use_montecarlo

    aligned_series = align_series_to_timeframe(series, timeframe)
    price_frame = series_to_price_frame(aligned_series)
    if fit_model:
        features = prepare_features(price_frame)
        model.fit(features.target, features.regressors)

    provider = _SeriesDataProvider(price_frame)
    temp_dir = Path(tempfile.mkdtemp(prefix="alphalens_manual_models_"))
    try:
        router = ModelRouter(temp_dir)
        engine = ForecastEngine(resolved_config, provider, router)
        return engine.forecast(
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
            paths=resolved_paths,
            use_montecarlo=resolved_use_mc,
            reuse_cached=False,
            model_store=None,
            show_progress=show_progress,
            trajectory_recorder=None,
            price_frame=price_frame,
            mean_model_override=model,
            vol_model_override=vol_model,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = [
    "ForecastEngine",
    "FREQ_MAP",
    "OrchestrationResult",
    "build_future_index",
    "compute_dataframe_hash",
    "compute_student_t_quantiles",
    "format_timestamp",
    "make_run_timestamp",
    "summarize_garch_model",
    "summarize_mean_model",
    "forecast_from_series",
]
