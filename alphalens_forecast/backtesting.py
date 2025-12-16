"""Helpers for trajectory exports and walk-forward backtesting diagnostics."""
from __future__ import annotations

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
from alphalens_forecast.models import ModelRouter


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


def evaluate_trajectory(actual: pd.Series, trajectory: ForecastTrajectory) -> Dict[str, float]:
    """Compare a trajectory to realised prices."""
    actual_aligned, predicted = _align_series(actual, trajectory.to_series())
    errors = actual_aligned - predicted
    return {
        "rmse": float(np.sqrt(np.nanmean(np.square(errors)))),
        "mae": float(np.nanmean(np.abs(errors))),
        "direction_accuracy": _direction_accuracy(actual_aligned, predicted),
    }


def _align_series(actual: pd.Series, predicted: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = actual.index.intersection(predicted.index)
    if idx.empty:
        raise ValueError("No overlapping timestamps between actual and predicted series.")
    return actual.reindex(idx), predicted.reindex(idx)


def _direction_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    actual_diff = actual.diff().to_numpy(dtype=float)[1:]
    predicted_diff = predicted.diff().to_numpy(dtype=float)[1:]
    valid = np.logical_and(np.isfinite(actual_diff), np.isfinite(predicted_diff))
    if not valid.any():
        return float("nan")
    return float(np.mean(np.sign(actual_diff[valid]) == np.sign(predicted_diff[valid])))


@dataclass
class BacktestEvaluation:
    """Metrics for a single (cutoff, horizon) evaluation."""

    cutoff: pd.Timestamp
    horizon: str
    steps: int
    metrics: Dict[str, float]
    prediction_last: float
    actual_last: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cutoff": self.cutoff.isoformat(),
            "horizon": self.horizon,
            "steps": self.steps,
            "metrics": self.metrics,
            "prediction_last": self.prediction_last,
            "actual_last": self.actual_last,
        }


@dataclass
class BacktestResult:
    """Container summarising walk-forward performance."""

    evaluations: List[BacktestEvaluation]
    aggregates: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "aggregates": self.aggregates,
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
                )
            )

        aggregates = self._aggregate_evaluations(evaluations)
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
            "window_start": frame.index[min_anchor].isoformat(),
            "window_end": frame.index[max_anchor].isoformat(),
        }
        return BacktestResult(evaluations=evaluations, aggregates=aggregates, metadata=metadata)

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
    ) -> List[BacktestEvaluation]:
        """Train on history up to ``anchor`` and score predictions."""
        cutoff_idx = anchor
        training_frame = price_frame.iloc[:cutoff_idx].copy()
        future_prices = price_frame["close"].iloc[cutoff_idx : cutoff_idx + max(horizon_steps.values())].copy()
        if future_prices.isna().any():
            future_prices = future_prices.fillna(method="ffill")
        provider = _FrozenDataProvider(training_frame)
        temp_dir = Path(tempfile.mkdtemp(prefix="alphalens_bt_"))
        try:
            router = ModelRouter(temp_dir)
            trajectory_recorder = TrajectoryRecorder()
            from alphalens_forecast.forecasting import ForecastEngine

            engine = ForecastEngine(self._config, provider, router)
            engine.forecast(
                symbol=symbol,
                timeframe=timeframe,
                horizons=horizon_hours,
                paths=paths,
                use_montecarlo=use_montecarlo,
                reuse_cached=False,
                model_store=None,
                show_progress=False,
                trajectory_recorder=trajectory_recorder,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        cutoff_ts = training_frame.index[-1]
        trajectory_map = {traj.horizon_label: traj for traj in trajectory_recorder.trajectories}
        evaluations: List[BacktestEvaluation] = []
        for horizon_label, steps in horizon_steps.items():
            trajectory = trajectory_map.get(horizon_label)
            if trajectory is None:
                continue
            actual_slice = future_prices.iloc[:steps]
            if len(actual_slice) < steps:
                continue
            metrics = evaluate_trajectory(actual_slice, trajectory)
            evaluations.append(
                BacktestEvaluation(
                    cutoff=cutoff_ts,
                    horizon=horizon_label,
                    steps=steps,
                    metrics=metrics,
                    prediction_last=float(trajectory.predictions[-1]),
                    actual_last=float(actual_slice.iloc[-1]),
                )
            )
        return evaluations

    @staticmethod
    def _aggregate_evaluations(evaluations: List[BacktestEvaluation]) -> Dict[str, Dict[str, float]]:
        """Aggregate RMSE/MAE/direction accuracy per horizon."""
        aggregates: Dict[str, Dict[str, Any]] = {}
        for entry in evaluations:
            bucket = aggregates.setdefault(
                entry.horizon,
                {"rmse": [], "mae": [], "direction_accuracy": [], "evaluations": 0},
            )
            bucket["evaluations"] += 1
            for metric in ("rmse", "mae", "direction_accuracy"):
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


__all__ = [
    "ForecastTrajectory",
    "TrajectoryRecorder",
    "evaluate_trajectory",
    "BacktestRunner",
    "BacktestResult",
    "BacktestEvaluation",
]
