"""Automatic hyperparameter tuning for Darts N-HiTS models."""
from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import NHiTSModel
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuningMetrics:
    """Container for evaluation metrics collected during tuning."""

    rmse: float
    mape: float
    directional_accuracy: float

    @property
    def score(self) -> float:
        """Combined objective: minimize RMSE while maximizing directional accuracy."""
        directional = 0.0 if np.isnan(self.directional_accuracy) else self.directional_accuracy
        return self.rmse - directional


def evaluate_forecast(true: TimeSeries, predicted: TimeSeries) -> TuningMetrics:
    """Compute RMSE, MAPE, and directional accuracy for aligned true/predicted series."""
    if true.n_components != 1 or predicted.n_components != 1:
        raise ValueError("Only univariate TimeSeries instances are supported.")
    df = pd.concat(
        [true.pd_series(), predicted.pd_series()],
        axis=1,
        join="inner",
    ).dropna()
    df.columns = ["true", "pred"]
    if df.empty:
        raise ValueError("No overlapping timestamps between true and predicted series.")

    true_vals = df["true"].to_numpy(dtype=float)
    pred_vals = df["pred"].to_numpy(dtype=float)
    errors = pred_vals - true_vals
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_errors = np.abs(errors / true_vals)
    mape = float(np.nanmean(pct_errors) * 100.0)
    if np.isnan(mape):
        mape = float("inf")

    if len(df) > 1:
        true_diff = np.diff(true_vals)
        pred_diff = np.diff(pred_vals)
        directional = float(np.mean(np.sign(pred_diff) == np.sign(true_diff)))
    else:
        directional = float("nan")

    return TuningMetrics(rmse=rmse, mape=mape, directional_accuracy=directional)


def auto_tune_nhits(
    series: TimeSeries,
    horizon: int,
    candidate_params: Optional[Iterable[Mapping[str, Any]] | Mapping[str, Sequence[Any]]] = None,
) -> Tuple[NHiTSModel, pd.DataFrame]:
    """Run a robust N-HiTS hyperparameter search and return the best refitted model and metrics."""
    if not isinstance(series, TimeSeries):
        raise TypeError("series must be a darts.TimeSeries.")
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    param_sets = _prepare_candidate_params(horizon, candidate_params)
    if not param_sets:
        raise ValueError("No candidate hyper-parameters were provided.")

    train_series, val_series = _temporal_split(series, validation_ratio=0.2)
    if len(val_series) == 0:
        raise ValueError("Validation split is empty; provide a longer history.")

    records: List[Dict[str, Any]] = []
    best_record: Optional[Dict[str, Any]] = None
    progress_bar = tqdm(param_sets, desc="N-HiTS tuning", unit="config")
    for params in progress_bar:
        params = dict(params)
        record = {**params}
        try:
            _validate_window_lengths(params, train_series)
            model = _build_model(params, enable_early_stopping=True)
            model.fit(train_series, val_series=val_series, verbose=False)
            predictions = model.predict(len(val_series))
            metrics = evaluate_forecast(val_series, predictions)
            record.update(
                {
                    "rmse": metrics.rmse,
                    "mape": metrics.mape,
                    "directional_accuracy": metrics.directional_accuracy,
                    "score": metrics.score,
                    "status": "ok",
                    "error": None,
                }
            )
            if best_record is None or metrics.score < best_record["score"]:
                best_record = {**record}
                progress_bar.set_postfix({"rmse": f"{metrics.rmse:.4f}", "dir": f"{metrics.directional_accuracy:.3f}"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Configuration failed: %s | err=%s", params, exc)
            record.update(
                {
                    "rmse": np.inf,
                    "mape": np.inf,
                    "directional_accuracy": np.nan,
                    "score": np.inf,
                    "status": "error",
                    "error": str(exc),
                }
            )
        records.append(record)

    results_df = pd.DataFrame.from_records(records)
    successful = results_df[results_df["status"] == "ok"].sort_values("score")
    if best_record is None or successful.empty:
        raise RuntimeError("Tuning failed for all configurations. Review the logs for details.")

    best_params = {key: best_record[key] for key in _param_names(param_sets)}
    best_params["n_epochs"] = 50
    logger.info("Retraining best configuration on the full series: %s", best_params)
    best_model = _build_model(best_params, enable_early_stopping=False)
    best_model.fit(series, verbose=False)
    arranged_results = (
        results_df.sort_values(by=["status", "score"], ascending=[False, True]).reset_index(drop=True)
    )
    return best_model, arranged_results


def _prepare_candidate_params(
    horizon: int,
    candidate_params: Optional[Iterable[Mapping[str, Any]] | Mapping[str, Sequence[Any]]],
) -> List[Dict[str, Any]]:
    if candidate_params is None:
        grid: Dict[str, Sequence[Any]] = {
            "input_chunk_length": [32, 64, 128],
            "output_chunk_length": [max(horizon, 1), max(2 * horizon, 1)],
            "n_epochs": [5, 20],
            "dropout": [0.0, 0.1, 0.2],
            "batch_size": [16, 32],
            "learning_rate": [1e-3, 5e-4, 1e-4],
        }
        return _expand_grid(grid)
    if isinstance(candidate_params, Mapping):
        return _expand_grid(candidate_params)
    return [dict(params) for params in candidate_params]


def _expand_grid(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values_product = itertools.product(*(grid[key] for key in keys))
    return [dict(zip(keys, combo)) for combo in values_product]


def _param_names(param_sets: List[Dict[str, Any]]) -> List[str]:
    if not param_sets:
        return []
    return sorted(param_sets[0].keys())


def _temporal_split(series: TimeSeries, validation_ratio: float) -> Tuple[TimeSeries, TimeSeries]:
    split_idx = max(1, int(len(series) * (1 - validation_ratio)))
    split_idx = min(split_idx, len(series) - 1)
    split_point = series.time_index[split_idx]
    train, val = series.split_before(split_point)
    return train, val


def _build_model(params: Mapping[str, Any], *, enable_early_stopping: bool) -> NHiTSModel:
    trainer_kwargs: Dict[str, Any] = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    if enable_early_stopping:
        trainer_kwargs["callbacks"] = [
            EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-4, mode="min"),
        ]
    learning_rate = params.get("learning_rate", 1e-3)
    return NHiTSModel(
        input_chunk_length=int(params["input_chunk_length"]),
        output_chunk_length=int(params["output_chunk_length"]),
        n_epochs=int(params["n_epochs"]),
        dropout=float(params.get("dropout", 0.0)),
        batch_size=int(params.get("batch_size", 32)),
        loss_fn=SmoothL1Loss(),
        optimizer_cls=Adam,
        optimizer_kwargs={"lr": float(learning_rate)},
        random_state=42,
        pl_trainer_kwargs=trainer_kwargs,
    )


def _validate_window_lengths(params: Mapping[str, Any], train_series: TimeSeries) -> None:
    input_len = int(params["input_chunk_length"])
    if input_len >= len(train_series):
        raise ValueError(
            f"input_chunk_length={input_len} exceeds available training history ({len(train_series)} timesteps)."
        )


# Example usage (commented out to keep the module import-safe):
# from darts.datasets import AirPassengersDataset
# series = AirPassengersDataset().load()
# best_model, tuning_results = auto_tune_nhits(series, horizon=12, candidate_params=None)
# print(tuning_results.head())
