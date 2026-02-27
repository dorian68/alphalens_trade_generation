"""Performance-report helpers for AlphaLens forecasts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from alphalens_forecast.metrics.direction import (
    brier_score_direction,
    coverage_from_mc,
    direction_accuracy_from_series,
    direction_accuracy_v1_step,
    direction_accuracy_v2_anchor,
    direction_accuracy_v3_deadzone,
    get_deadzone_abs,
    get_deadzone_atr_k,
    get_direction_accuracy_mode,
    reporting_extended_metrics_default_on,
    use_reporting_same_metrics,
)


def _align_series(*series: pd.Series) -> Iterable[pd.Series]:
    """Align multiple series to their common index."""
    if not series:
        return tuple()
    common_index = series[0].index
    for ser in series[1:]:
        common_index = common_index.intersection(ser.index)
    if common_index.empty:
        raise ValueError("Series do not share a common index; cannot build report.")
    return tuple(ser.reindex(common_index) for ser in series)


def _safe_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute descriptive stats with finite fallbacks."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


@dataclass
class PerformanceReport:
    """Structured performance summary."""

    metrics: Dict[str, float]
    coverage: Dict[str, float]
    residuals: Dict[str, float]
    volatility: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "metrics": self.metrics,
            "coverage": self.coverage,
            "residuals": self.residuals,
            "volatility": self.volatility,
            "metadata": self.metadata,
            "notes": self.notes,
        }


def generate_performance_report(
    *,
    actual: pd.Series,
    predicted: pd.Series,
    quantiles: Optional[Dict[str, pd.Series]] = None,
    residuals: Optional[pd.Series] = None,
    sigma: Optional[pd.Series] = None,
    last_observed: Optional[float] = None,
    pred_samples: Optional[np.ndarray] = None,
    atr: Optional[pd.Series] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PerformanceReport:
    """
    Build a consolidated performance/diagnostic report for a forecast.

    Parameters
    ----------
    actual:
        Series of realised prices.
    predicted:
        Series of model predictions aligned to ``actual``.
    quantiles:
        Optional mapping containing lower/upper scenarios (expects keys ``p20``/``p80``).
    residuals:
        Optional residual series. If omitted, residuals are computed as ``actual - predicted``.
    sigma:
        Optional volatility path (per-step sigma).
    last_observed:
        Optional anchor price used for v2/v3 direction accuracy.
    pred_samples:
        Optional Monte Carlo samples for coverage/brier metrics.
    atr:
        Optional ATR series used to scale deadzone thresholds.
    metadata:
        Optional context (model type, timeframe, training info, etc.).
    """
    actual_aligned, predicted_aligned = _align_series(actual, predicted)
    residual_series = residuals if residuals is not None else (actual_aligned - predicted_aligned)

    errors = (actual_aligned - predicted_aligned).to_numpy(dtype=float)
    actual_values = actual_aligned.to_numpy(dtype=float)
    safe_actual = actual_values.copy()
    safe_actual[np.isclose(safe_actual, 0.0)] = np.nan
    metrics = {
        "rmse": float(np.sqrt(np.nanmean(np.square(errors)))),
        "mae": float(np.nanmean(np.abs(errors))),
        "mape": float(np.nanmean(np.abs(errors / safe_actual))) * 100.0,
    }

    notes: list[str] = []
    deadzone_abs = get_deadzone_abs()
    deadzone_atr = None
    deadzone_atr_k = get_deadzone_atr_k()
    if deadzone_atr_k is not None and atr is not None and not atr.empty:
        atr_aligned = atr.reindex(actual_aligned.index).astype(float)
        atr_value = float(atr_aligned.iloc[-1])
        if np.isfinite(atr_value):
            deadzone_atr = float(deadzone_atr_k) * atr_value
    mode = get_direction_accuracy_mode()
    if use_reporting_same_metrics():
        direction_accuracy = direction_accuracy_from_series(
            actual_aligned,
            predicted_aligned,
            last_observed=last_observed,
            mode=mode,
            deadzone_abs=deadzone_abs,
            deadzone_atr=deadzone_atr,
        )
        if mode != "v1" and last_observed is None:
            direction_accuracy = direction_accuracy_v1_step(actual_aligned, predicted_aligned)
            notes.append("direction_accuracy uses v1 (last_observed unavailable for v2/v3).")
    else:
        direction_accuracy = direction_accuracy_v1_step(actual_aligned, predicted_aligned)
    metrics["direction_accuracy"] = direction_accuracy

    coverage = {}
    if quantiles and {"p20", "p80"} <= quantiles.keys():
        lower, upper = _align_series(actual, quantiles["p20"], quantiles["p80"])[1:]
        in_band = (actual_aligned >= lower) & (actual_aligned <= upper)
        coverage["p20_p80_band"] = float(in_band.mean())
        coverage["p20_breach"] = float((actual_aligned < lower).mean())
        coverage["p80_breach"] = float((actual_aligned > upper).mean())
    else:
        notes.append("Quantile coverage unavailable (p20/p80 missing).")

    residual_array = residual_series.reindex(actual_aligned.index).to_numpy(dtype=float)
    residual_stats = _safe_stats(residual_array)
    residual_stats["skew"] = float(pd.Series(residual_array).skew(skipna=True))
    residual_stats["kurtosis"] = float(pd.Series(residual_array).kurtosis(skipna=True))

    volatility_stats: Dict[str, float]
    if sigma is not None:
        sigma_aligned = sigma.reindex(actual_aligned.index, method="ffill")
        volatility_stats = _safe_stats(sigma_aligned.to_numpy(dtype=float))
    else:
        volatility_stats = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        notes.append("Sigma path unavailable; volatility stats omitted.")

    if reporting_extended_metrics_default_on():
        metrics["direction_accuracy_v1"] = direction_accuracy_v1_step(actual_aligned, predicted_aligned)
        if last_observed is not None and not actual_aligned.empty and not predicted_aligned.empty:
            actual_end = float(actual_aligned.iloc[-1])
            pred_end = float(predicted_aligned.iloc[-1])
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
        else:
            metrics["direction_accuracy_v2"] = float("nan")
            metrics["direction_accuracy_v3"] = float("nan")
            metrics["conditional_direction_accuracy"] = float("nan")
        if pred_samples is not None and not actual_aligned.empty:
            samples = np.asarray(pred_samples, dtype=float).reshape(-1)
            actual_end = float(actual_aligned.iloc[-1])
            metrics["coverage_50"] = coverage_from_mc(actual_end, samples, alpha=0.5)
            metrics["coverage_80"] = coverage_from_mc(actual_end, samples, alpha=0.2)
            if last_observed is not None and np.isfinite(last_observed):
                prob_up = float(np.nanmean(samples > float(last_observed)))
                metrics["prob_up_brier"] = brier_score_direction(prob_up, actual_end > float(last_observed))
        elif "p20_p80_band" in coverage:
            metrics["coverage_80"] = float(coverage["p20_p80_band"])

    return PerformanceReport(
        metrics=metrics,
        coverage=coverage,
        residuals=residual_stats,
        volatility=volatility_stats,
        metadata=metadata or {},
        notes=notes,
    )
