"""Directional and coverage metrics for evaluation/reporting.

Modes:
- v1: step-to-step directional agreement (trajectory shape).
- v2: anchor-to-horizon direction (trade decision aligned).
- v3: v2 with a deadzone / significance filter.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import numpy as np
import pandas as pd

DirectionAccuracyMode = Literal["v1", "v2", "v3"]
_DEFAULT_DIRECTION_MODE: DirectionAccuracyMode = "v3"
_DEFAULT_DEADZONE_ATR_K = 0.75


def get_direction_accuracy_mode() -> DirectionAccuracyMode:
    """Resolve the direction-accuracy mode from the environment."""
    raw = os.getenv("DIRECTION_ACCURACY_MODE")
    if raw:
        value = raw.strip().lower()
        if value in {"v1", "v2", "v3"}:
            return value  # type: ignore[return-value]
    legacy = os.getenv("DIRECTION_ACCURACY_V2")
    if legacy and legacy.strip().lower() in {"1", "true", "yes", "y"}:
        return "v2"
    return _DEFAULT_DIRECTION_MODE


def get_deadzone_abs() -> Optional[float]:
    """Return the absolute deadzone threshold when configured."""
    raw = os.getenv("DIRECTION_DEADZONE_ABS")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


def get_deadzone_atr_k() -> Optional[float]:
    """Return the ATR multiplier deadzone threshold when configured."""
    raw = os.getenv("DIRECTION_DEADZONE_ATR_K")
    if raw is None:
        return float(_DEFAULT_DEADZONE_ATR_K)
    try:
        value = float(raw)
    except ValueError:
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value


def use_reporting_same_metrics() -> bool:
    """Return True when reporting should use the same mode as backtesting."""
    raw = os.getenv("REPORTING_USE_SAME_METRICS")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def use_reporting_extended_metrics() -> bool:
    """Return True when reporting should emit extended metrics."""
    raw = os.getenv("REPORTING_EXTENDED_METRICS")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def reporting_extended_metrics_default_on() -> bool:
    """Return True by default; allow opt-out via REPORTING_EXTENDED_METRICS=0."""
    raw = os.getenv("REPORTING_EXTENDED_METRICS")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def direction_accuracy_v1_step(actual: pd.Series, pred: pd.Series) -> float:
    """
    Step-to-step directional agreement.

    Matches legacy behaviour: compares sign(diff) per step, includes sign(0)=0,
    returns NaN when fewer than 2 points or no valid diffs exist.
    """
    actual_diff = actual.diff().to_numpy(dtype=float)[1:]
    pred_diff = pred.diff().to_numpy(dtype=float)[1:]
    valid = np.logical_and(np.isfinite(actual_diff), np.isfinite(pred_diff))
    if not valid.any():
        return float("nan")
    return float(np.mean(np.sign(actual_diff[valid]) == np.sign(pred_diff[valid])))


def direction_accuracy_v2_anchor(last_observed: float, actual_end: float, pred_end: float) -> float:
    """
    Anchor-to-horizon direction accuracy.

    Direction is sign(end - last_observed). Returns NaN on non-finite inputs.
    """
    if not np.isfinite(last_observed) or not np.isfinite(actual_end) or not np.isfinite(pred_end):
        return float("nan")
    return float(np.sign(actual_end - last_observed) == np.sign(pred_end - last_observed))


def direction_accuracy_v3_deadzone(
    last_observed: float,
    actual_end: float,
    pred_end: float,
    deadzone_abs: Optional[float] = None,
    deadzone_atr: Optional[float] = None,
) -> float:
    """
    Anchor-to-horizon direction accuracy with a deadzone filter.

    If |actual_end - last_observed| is below the deadzone threshold, returns NaN.
    """
    if not np.isfinite(last_observed) or not np.isfinite(actual_end) or not np.isfinite(pred_end):
        return float("nan")
    threshold = _resolve_deadzone_threshold(deadzone_abs, deadzone_atr)
    if threshold is not None:
        if abs(actual_end - last_observed) < threshold:
            return float("nan")
    return float(np.sign(actual_end - last_observed) == np.sign(pred_end - last_observed))


def direction_accuracy_from_series(
    actual: pd.Series,
    pred: pd.Series,
    *,
    last_observed: Optional[float] = None,
    mode: Optional[DirectionAccuracyMode] = None,
    deadzone_abs: Optional[float] = None,
    deadzone_atr: Optional[float] = None,
) -> float:
    """Compute direction accuracy using the selected mode."""
    mode_value = mode or get_direction_accuracy_mode()
    if mode_value == "v1":
        return direction_accuracy_v1_step(actual, pred)
    if last_observed is None:
        return float("nan")
    if actual.empty or pred.empty:
        return float("nan")
    actual_end = float(actual.iloc[-1])
    pred_end = float(pred.iloc[-1])
    if mode_value == "v2":
        return direction_accuracy_v2_anchor(last_observed, actual_end, pred_end)
    if mode_value == "v3":
        return direction_accuracy_v3_deadzone(
            last_observed,
            actual_end,
            pred_end,
            deadzone_abs=deadzone_abs,
            deadzone_atr=deadzone_atr,
        )
    return float("nan")


def coverage_from_mc(actual_end: float, pred_samples: np.ndarray, alpha: float) -> float:
    """
    Return 1 if actual_end is inside the (1-alpha) interval from samples.
    """
    if not np.isfinite(actual_end):
        return float("nan")
    if pred_samples is None:
        return float("nan")
    samples = np.asarray(pred_samples, dtype=float)
    if samples.size == 0:
        return float("nan")
    if alpha <= 0.0 or alpha >= 1.0:
        return float("nan")
    q_low, q_high = np.nanquantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    if not np.isfinite(q_low) or not np.isfinite(q_high):
        return float("nan")
    return float(q_low <= actual_end <= q_high)


def brier_score_direction(prob_up: float, actual_direction_up: bool) -> float:
    """Brier score for a binary up/down direction forecast."""
    try:
        p = float(prob_up)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(p):
        return float("nan")
    outcome = 1.0 if actual_direction_up else 0.0
    return float((p - outcome) ** 2)


def _resolve_deadzone_threshold(
    deadzone_abs: Optional[float],
    deadzone_atr: Optional[float],
) -> Optional[float]:
    candidates = []
    if deadzone_abs is not None and np.isfinite(deadzone_abs):
        candidates.append(float(deadzone_abs))
    if deadzone_atr is not None and np.isfinite(deadzone_atr):
        candidates.append(float(deadzone_atr))
    if not candidates:
        return None
    threshold = max(candidates)
    if threshold < 0:
        return None
    return threshold
