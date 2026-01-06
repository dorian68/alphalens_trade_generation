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
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from alphalens_forecast.backtesting import TrajectoryRecorder
from alphalens_forecast.models.selection import resolve_device, select_model_type
from alphalens_forecast.training import MEAN_TRAINERS, train_egarch
from alphalens_forecast.utils.model_store import ModelStore, StoredArtifacts
from alphalens_forecast.utils.text import slugify
from alphalens_forecast.utils.timeseries import align_series_to_timeframe, series_to_price_frame

logger = logging.getLogger(__name__)

FREQ_MAP: Dict[str, str] = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1h": "1H",
    "2h": "2H",
    "3h": "3H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
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
    q15_log = median_log + dist.ppf(0.15) * sigma
    q50_log = median_log
    q85_log = median_log + dist.ppf(0.85) * sigma
    print(f"q15_log: {float(np.exp(q15_log))} and q15 knowing its 2 times sigma: {float(np.exp(median_log + dist.ppf(0.15) * 2 * sigma))} ")
    print(f"q85_log: {float(np.exp(q85_log))} and q85 knowing its 2 times sigma: {float(np.exp(median_log + dist.ppf(0.85) * 2 * sigma))} ")
    return {
        "p20": float(np.exp(q15_log)),
        "p50": float(np.exp(q50_log)),
        "p80": float(np.exp(q85_log)),
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

        data_hash = compute_dataframe_hash(price_frame)
        logger.debug("Price frame hash: %s", data_hash)
        if model_store and reuse_cached and frame_override is None and not force_retrain:
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
                return reused

        features = prepare_features(price_frame)
        target_series = features.target
        last_price = float(target_series.iloc[-1])
        if last_price <= 0:
            raise ValueError("Close price must be positive to compute log transforms.")
        last_log_price = float(np.log(last_price))
        as_of_ts = last_complete_ts or price_frame.index[-1]
        as_of = format_timestamp(as_of_ts)

        freq = FREQ_MAP.get(timeframe.lower())
        if freq is None:
            raise ValueError(f"No pandas frequency mapping for timeframe '{timeframe}'.")

        model_type = select_model_type(timeframe)
        device = resolve_device(self._config.torch_device, model_type)
        mean_model = mean_model_override
        if mean_model is not None:
            durations["mean_model_fit_seconds"] = 0.0
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

            horizon_variance = variance_path.iloc[:steps].to_numpy(dtype=float)
            sigma_per_step = sigma_path.iloc[:steps].to_numpy(dtype=float)
            if not np.isfinite(horizon_variance).all() or not np.isfinite(sigma_per_step).all():
                logger.error("Invalid EGARCH path detected for horizon %s.", horizon_hours)
                raise RuntimeError("EGARCH provided invalid variance or sigma values.")
            sigma_h = float(np.sqrt(np.sum(horizon_variance)))

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

            probability = None
            if use_montecarlo and mc_simulator is not None:
                mc_start = time.perf_counter()
                step_hours = interval_to_hours(timeframe)
                mc_result = mc_simulator.simulate(
                    current_price=last_price,
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
        if rolling_summary:
            metadata["rolling_eval"] = rolling_summary
        metadata["residual_std"] = float(residuals.std(ddof=0))
        metadata["last_price"] = last_price
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
