"""Helpers for trajectory exports and walk-forward backtesting diagnostics."""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from alphalens_forecast.config import AppConfig
from alphalens_forecast.core import horizon_to_steps
from alphalens_forecast.data import DataProvider
from alphalens_forecast.metrics.direction import (
    direction_accuracy_v1_step,
    direction_accuracy_v2_anchor,
    direction_accuracy_v3_deadzone,
    direction_accuracy_from_series,
    get_deadzone_abs,
    get_deadzone_atr_k,
    get_direction_accuracy_mode,
    use_reporting_extended_metrics,
)
from alphalens_forecast.models import ModelRouter

logger = logging.getLogger(__name__)
_ATR_WINDOW = 14


@dataclass
class ForecastTrajectory:
    """Store step-by-step predictions for a forecast horizon."""

    horizon_label: str
    timestamps: List[pd.Timestamp]
    predictions: List[float]

    def to_series(self) -> pd.Series:
        """Return predictions as a pandas Series."""
        return pd.Series(self.predictions, index=pd.DatetimeIndex(self.timestamps))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise trajectory."""
        return {
            "horizon": self.horizon_label,
            "steps": len(self.predictions),
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "predictions": self.predictions,
        }


class TrajectoryRecorder:
    """Collect trajectories during forecasting for export or backtesting."""

    def __init__(self) -> None:
        self._trajectories: List[ForecastTrajectory] = []

    def add_from_dataframe(
        self,
        horizon_label: str,
        forecast_df: pd.DataFrame,
    ) -> None:
        """Store the sequence of predictions for a given horizon."""
        if "yhat" not in forecast_df.columns:
            raise ValueError("forecast_df must contain a 'yhat' column.")
        series = forecast_df["yhat"].astype(float)
        timestamps = pd.to_datetime(series.index)
        self._trajectories.append(
            ForecastTrajectory(
                horizon_label=horizon_label,
                timestamps=list(timestamps),
                predictions=series.to_numpy(dtype=float).tolist(),
            )
        )

    def to_payload(self) -> List[Dict[str, Any]]:
        """Return serialisable trajectories."""
        return [traj.to_dict() for traj in self._trajectories]

    @property
    def trajectories(self) -> Iterable[ForecastTrajectory]:
        return tuple(self._trajectories)


def evaluate_trajectory(
    actual: pd.Series,
    trajectory: ForecastTrajectory,
    *,
    last_observed: Optional[float] = None,
    deadzone_atr: Optional[float] = None,
) -> Dict[str, float]:
    """Compare a trajectory to realised prices."""
    actual_aligned, predicted = _align_series(actual, trajectory.to_series())
    errors = actual_aligned - predicted
    metrics = {
        "rmse": float(np.sqrt(np.nanmean(np.square(errors)))),
        "mae": float(np.nanmean(np.abs(errors))),
        "direction_accuracy": _direction_accuracy(
            actual_aligned,
            predicted,
            last_observed=last_observed,
            deadzone_atr=deadzone_atr,
        ),
    }
    metrics.update(_trade_metrics(actual_aligned, predicted, last_observed=last_observed))
    if use_reporting_extended_metrics():
        metrics.update(_extended_direction_metrics(actual_aligned, predicted, last_observed, deadzone_atr))
    return metrics


def _coerce_utc_index(series: pd.Series) -> pd.Series:
    """Return a copy of ``series`` with a UTC DatetimeIndex."""
    idx = series.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True, errors="coerce")
    else:
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
    aligned = series.copy()
    aligned.index = idx
    return aligned[~aligned.index.isna()]


def _align_series(actual: pd.Series, predicted: pd.Series) -> tuple[pd.Series, pd.Series]:
    actual_utc = _coerce_utc_index(actual)
    predicted_utc = _coerce_utc_index(predicted)
    idx = actual_utc.index.intersection(predicted_utc.index)
    if idx.empty:
        # Fall back to tz-naive alignment if upstream data dropped timezone info.
        actual_naive = actual_utc.copy()
        predicted_naive = predicted_utc.copy()
        actual_naive.index = actual_naive.index.tz_localize(None)
        predicted_naive.index = predicted_naive.index.tz_localize(None)
        idx = actual_naive.index.intersection(predicted_naive.index)
        if not idx.empty:
            return actual_naive.reindex(idx), predicted_naive.reindex(idx)
        # Final fallback: align by position when timestamps are irreconcilable.
        min_len = min(len(actual_utc), len(predicted_utc))
        if min_len <= 0:
            raise ValueError("No overlapping timestamps between actual and predicted series.")
        logger.warning(
            "No overlapping timestamps between actual and predicted series; "
            "falling back to positional alignment (n=%d).",
            min_len,
        )
        actual_trim = actual_utc.iloc[:min_len]
        predicted_trim = predicted_utc.iloc[:min_len].copy()
        predicted_trim.index = actual_trim.index
        return actual_trim, predicted_trim
    return actual_utc.reindex(idx), predicted_utc.reindex(idx)


def _last_atr(frame: pd.DataFrame, window: int = _ATR_WINDOW) -> Optional[float]:
    required = {"high", "low", "close"}
    if not required.issubset(frame.columns):
        return None
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    value = float(atr.iloc[-1])
    return value if np.isfinite(value) else None


def _direction_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    last_observed: Optional[float] = None,
    deadzone_atr: Optional[float] = None,
) -> float:
    return direction_accuracy_from_series(
        actual,
        predicted,
        last_observed=last_observed,
        mode=get_direction_accuracy_mode(),
        deadzone_abs=get_deadzone_abs(),
        deadzone_atr=deadzone_atr,
    )


def _extended_direction_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    last_observed: Optional[float],
    deadzone_atr: Optional[float],
) -> Dict[str, float]:
    deadzone_abs = get_deadzone_abs()
    metrics: Dict[str, float] = {
        "direction_accuracy_v1": direction_accuracy_v1_step(actual, predicted),
    }
    if last_observed is None or actual.empty or predicted.empty:
        metrics["direction_accuracy_v2"] = float("nan")
        metrics["direction_accuracy_v3"] = float("nan")
        metrics["conditional_direction_accuracy"] = float("nan")
        return metrics
    actual_end = float(actual.iloc[-1])
    pred_end = float(predicted.iloc[-1])
    metrics["direction_accuracy_v2"] = direction_accuracy_v2_anchor(last_observed, actual_end, pred_end)
    metrics["direction_accuracy_v3"] = direction_accuracy_v3_deadzone(
        last_observed,
        actual_end,
        pred_end,
        deadzone_abs=deadzone_abs,
        deadzone_atr=deadzone_atr,
    )
    threshold_set = False
    if deadzone_abs is not None and np.isfinite(deadzone_abs) and deadzone_abs > 0:
        threshold_set = True
    if deadzone_atr is not None and np.isfinite(deadzone_atr) and deadzone_atr > 0:
        threshold_set = True
    metrics["conditional_direction_accuracy"] = (
        metrics["direction_accuracy_v3"] if threshold_set else float("nan")
    )
    return metrics


def _aggregation_metrics() -> List[str]:
    metrics = ["rmse", "mae", "direction_accuracy"]
    if use_reporting_extended_metrics():
        metrics.extend(
            [
                "direction_accuracy_v1",
                "direction_accuracy_v2",
                "direction_accuracy_v3",
                "conditional_direction_accuracy",
                "coverage_50",
                "coverage_80",
                "prob_up_brier",
            ]
        )
    metrics.extend(
        [
            "trade_return",
            "trade_return_pct",
            "trade_win_rate",
        ]
    )
    return metrics


def _trade_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    *,
    last_observed: Optional[float],
) -> Dict[str, float]:
    """Compute simple trade PnL metrics based on anchor-to-horizon direction."""
    if last_observed is None or actual.empty or predicted.empty:
        return {
            "trade_return": float("nan"),
            "trade_return_pct": float("nan"),
            "trade_win_rate": float("nan"),
        }
    try:
        actual_end = float(actual.iloc[-1])
        pred_end = float(predicted.iloc[-1])
        entry = float(last_observed)
    except (TypeError, ValueError):
        return {
            "trade_return": float("nan"),
            "trade_return_pct": float("nan"),
            "trade_win_rate": float("nan"),
        }
    if not np.isfinite(entry) or entry == 0:
        return {
            "trade_return": float("nan"),
            "trade_return_pct": float("nan"),
            "trade_win_rate": float("nan"),
        }
    direction = np.sign(pred_end - entry)
    if direction == 0 or not np.isfinite(actual_end):
        return {
            "trade_return": 0.0,
            "trade_return_pct": 0.0,
            "trade_win_rate": 0.0,
        }
    pnl = direction * (actual_end - entry)
    pnl_pct = pnl / entry
    win = 1.0 if pnl > 0 else 0.0
    return {
        "trade_return": float(pnl),
        "trade_return_pct": float(pnl_pct),
        "trade_win_rate": float(win),
    }


def _regime_key(regime: Dict[str, Any]) -> str:
    if not regime:
        return "unknown"
    enabled = regime.get("enabled", True)
    if enabled is False:
        return "disabled"
    label = regime.get("label")
    if label:
        return str(label)
    route = regime.get("route")
    if route:
        return f"unknown:{route}"
    return "unknown"


@dataclass
class BacktestEvaluation:
    """Metrics for a single (cutoff, horizon) evaluation."""

    cutoff: pd.Timestamp
    horizon: str
    steps: int
    metrics: Dict[str, float]
    prediction_last: float
    actual_last: float
    regime: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cutoff": self.cutoff.isoformat(),
            "horizon": self.horizon,
            "steps": self.steps,
            "metrics": self.metrics,
            "prediction_last": self.prediction_last,
            "actual_last": self.actual_last,
            "regime": self.regime,
        }


@dataclass
class BacktestResult:
    """Container summarising walk-forward performance."""

    evaluations: List[BacktestEvaluation]
    aggregates: Dict[str, Dict[str, float]]
    regime_aggregates: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "aggregates": self.aggregates,
            "regime_aggregates": self.regime_aggregates,
            "evaluations": [entry.to_dict() for entry in self.evaluations],
        }


class _FrozenDataProvider:
    """Minimal provider that serves an in-memory price frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:  # noqa: D401 - signature mirrors DataProvider
        return self._frame


class BacktestRunner:
    """Utility that replays historical windows and scores forecasts."""

    def __init__(self, config: AppConfig, cache_dir: Optional[Path] = None) -> None:
        cache = cache_dir.resolve() if cache_dir else None
        self._config = config
        self._data_provider = DataProvider(config.twelve_data, cache_dir=cache)

    def run(
        self,
        *,
        symbol: str,
        timeframe: str,
        horizons: Sequence[int],
        paths: int,
        use_montecarlo: bool,
        samples: Optional[int] = None,
        stride: Optional[int] = None,
        min_history: int = 500,
        enable_regime_switching: Optional[bool] = None,
        regime_lookback: Optional[int] = None,
        enable_performance_patches: Optional[bool] = None,
    ) -> BacktestResult:
        """Execute a walk-forward backtest."""
        frame = self._data_provider.load_data(symbol, timeframe)
        if len(frame) < min_history:
            raise ValueError(f"Insufficient history for backtest ({len(frame)} < {min_history}).")

        horizon_values = [int(h) for h in horizons]
        horizon_steps = {f"{hours}h": horizon_to_steps(hours, timeframe) for hours in horizon_values}
        max_steps = max(horizon_steps.values())
        stride = max(stride or max_steps, 1)
        min_anchor = max(min_history, max_steps)
        max_anchor = len(frame) - max_steps
        if max_anchor <= min_anchor:
            raise ValueError("Not enough observations to create backtest windows.")

        candidate_indices = list(range(min_anchor, max_anchor + 1, stride))
        if samples and samples > 0:
            anchors = candidate_indices[-samples:]
        else:
            anchors = candidate_indices
        if not anchors:
            raise ValueError("No valid backtest anchors were generated.")

        evaluations: List[BacktestEvaluation] = []
        for anchor in anchors:
            evaluations.extend(
                self._evaluate_window(
                    price_frame=frame,
                    anchor=anchor,
                    symbol=symbol,
                    timeframe=timeframe,
                    horizon_hours=horizon_values,
                    horizon_steps=horizon_steps,
                    paths=paths,
                    use_montecarlo=use_montecarlo,
                    enable_regime_switching=enable_regime_switching,
                    regime_lookback=regime_lookback,
                    enable_performance_patches=enable_performance_patches,
                )
            )

        aggregates = self._aggregate_evaluations(evaluations)
        regime_aggregates = self._aggregate_by_regime(evaluations)
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizons": horizon_values,
            "n_windows": len(anchors),
            "stride_bars": stride,
            "min_history": min_history,
            "max_steps": max_steps,
            "use_montecarlo": use_montecarlo,
            "montecarlo_paths": paths if use_montecarlo else 0,
            "regime_switching": enable_regime_switching if enable_regime_switching is not None else self._config.regime_switching,
            "regime_lookback": regime_lookback if regime_lookback is not None else self._config.regime_lookback,
            "window_start": frame.index[min_anchor].isoformat(),
            "window_end": frame.index[max_anchor].isoformat(),
        }
        performance_meta = (
            enable_performance_patches
            if enable_performance_patches is not None
            else (self._config.performance_patches if self._config.performance_patches else None)
        )
        if performance_meta is not None:
            metadata["performance_patches"] = performance_meta
        return BacktestResult(
            evaluations=evaluations,
            aggregates=aggregates,
            regime_aggregates=regime_aggregates,
            metadata=metadata,
        )

    def _evaluate_window(
        self,
        *,
        price_frame: pd.DataFrame,
        anchor: int,
        symbol: str,
        timeframe: str,
        horizon_hours: Sequence[int],
        horizon_steps: Dict[str, int],
        paths: int,
        use_montecarlo: bool,
        enable_regime_switching: Optional[bool],
        regime_lookback: Optional[int],
        enable_performance_patches: Optional[bool],
    ) -> List[BacktestEvaluation]:
        """Train on history up to ``anchor`` and score predictions."""
        cutoff_idx = anchor
        training_frame = price_frame.iloc[:cutoff_idx].copy()
        future_prices = price_frame["close"].iloc[cutoff_idx : cutoff_idx + max(horizon_steps.values())].copy()
        if future_prices.isna().any():
            future_prices = future_prices.fillna(method="ffill")
        deadzone_atr = None
        deadzone_atr_k = get_deadzone_atr_k()
        if deadzone_atr_k is not None:
            atr_value = _last_atr(training_frame)
            if atr_value is not None:
                deadzone_atr = float(deadzone_atr_k) * atr_value
        provider = _FrozenDataProvider(training_frame)
        temp_dir = Path(tempfile.mkdtemp(prefix="alphalens_bt_"))
        result = None
        try:
            router = ModelRouter(temp_dir)
            trajectory_recorder = TrajectoryRecorder()
            from alphalens_forecast.forecasting import ForecastEngine

            engine = ForecastEngine(self._config, provider, router)
            result = engine.forecast(
                symbol=symbol,
                timeframe=timeframe,
                horizons=horizon_hours,
                paths=paths,
                use_montecarlo=use_montecarlo,
                reuse_cached=False,
                model_store=None,
                show_progress=False,
                trajectory_recorder=trajectory_recorder,
                enable_regime_switching=enable_regime_switching,
                regime_lookback=regime_lookback,
                enable_performance_patches=enable_performance_patches,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        cutoff_ts = training_frame.index[-1]
        last_observed_price = float(training_frame["close"].iloc[-1])
        regime_info = result.metadata.get("regime", {}) if result is not None else {}
        trajectory_map = {traj.horizon_label: traj for traj in trajectory_recorder.trajectories}
        evaluations: List[BacktestEvaluation] = []
        for horizon_label, steps in horizon_steps.items():
            trajectory = trajectory_map.get(horizon_label)
            if trajectory is None:
                continue
            actual_slice = future_prices.iloc[:steps]
            if len(actual_slice) < steps:
                continue
            metrics = evaluate_trajectory(
                actual_slice,
                trajectory,
                last_observed=last_observed_price,
                deadzone_atr=deadzone_atr,
            )
            evaluations.append(
                BacktestEvaluation(
                    cutoff=cutoff_ts,
                    horizon=horizon_label,
                    steps=steps,
                    metrics=metrics,
                    prediction_last=float(trajectory.predictions[-1]),
                    actual_last=float(actual_slice.iloc[-1]),
                    regime=regime_info,
                )
            )
        return evaluations

    @staticmethod
    def _aggregate_evaluations(evaluations: List[BacktestEvaluation]) -> Dict[str, Dict[str, float]]:
        """Aggregate RMSE/MAE/direction accuracy per horizon."""
        aggregates: Dict[str, Dict[str, Any]] = {}
        metrics = _aggregation_metrics()
        for entry in evaluations:
            bucket = aggregates.setdefault(
                entry.horizon,
                {metric: [] for metric in metrics} | {"evaluations": 0},
            )
            bucket["evaluations"] += 1
            for metric in metrics:
                value = entry.metrics.get(metric)
                if value is not None and np.isfinite(value):
                    bucket[metric].append(value)

        summary: Dict[str, Dict[str, float]] = {}
        for horizon, stats in aggregates.items():
            summary[horizon] = {
                metric: (float(np.mean(values)) if values else float("nan"))
                for metric, values in stats.items()
                if metric != "evaluations"
            }
            summary[horizon]["evaluations"] = int(stats.get("evaluations", 0))
        return summary

    @staticmethod
    def _aggregate_by_regime(evaluations: List[BacktestEvaluation]) -> Dict[str, Dict[str, float]]:
        """Aggregate RMSE/MAE/direction accuracy per regime label."""
        aggregates: Dict[str, Dict[str, Any]] = {}
        metrics = _aggregation_metrics()
        for entry in evaluations:
            key = _regime_key(entry.regime)
            bucket = aggregates.setdefault(
                key,
                {metric: [] for metric in metrics} | {"evaluations": 0, "confidence": []},
            )
            bucket["evaluations"] += 1
            for metric in metrics:
                value = entry.metrics.get(metric)
                if value is not None and np.isfinite(value):
                    bucket[metric].append(value)
            confidence = entry.regime.get("confidence") if entry.regime else None
            try:
                confidence_value = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence_value = None
            if confidence_value is not None and np.isfinite(confidence_value):
                bucket["confidence"].append(confidence_value)

        summary: Dict[str, Dict[str, float]] = {}
        for regime_key, stats in aggregates.items():
            summary[regime_key] = {
                metric: (float(np.mean(values)) if values else float("nan"))
                for metric, values in stats.items()
                if metric not in {"evaluations", "confidence"}
            }
            summary[regime_key]["evaluations"] = int(stats.get("evaluations", 0))
            if stats.get("confidence"):
                summary[regime_key]["mean_confidence"] = float(np.mean(stats["confidence"]))
        return summary


__all__ = [
    "ForecastTrajectory",
    "TrajectoryRecorder",
    "evaluate_trajectory",
    "BacktestRunner",
    "BacktestResult",
    "BacktestEvaluation",
]
