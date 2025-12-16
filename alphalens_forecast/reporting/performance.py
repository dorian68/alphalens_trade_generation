"""Performance-report helpers for AlphaLens forecasts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


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

    actual_diff = actual_aligned.diff().to_numpy(dtype=float)[1:]
    pred_diff = predicted_aligned.diff().to_numpy(dtype=float)[1:]
    valid = np.logical_and(np.isfinite(actual_diff), np.isfinite(pred_diff))
    direction_accuracy = float(np.mean(np.sign(actual_diff[valid]) == np.sign(pred_diff[valid]))) if valid.any() else float("nan")
    metrics["direction_accuracy"] = direction_accuracy

    coverage = {}
    notes: list[str] = []
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

    return PerformanceReport(
        metrics=metrics,
        coverage=coverage,
        residuals=residual_stats,
        volatility=volatility_stats,
        metadata=metadata or {},
        notes=notes,
    )
