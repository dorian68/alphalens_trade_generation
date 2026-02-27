"""Post-processing utilities for HMM regime labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

REGIME_TREND_UP = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGE = "RANGE"
REGIME_BREAKOUT = "BREAKOUT_VOL_EXPANSION"
REGIME_STRESS_CHOP = "STRESS_CHOP"

REGIME_ORDER = (
    REGIME_TREND_UP,
    REGIME_TREND_DOWN,
    REGIME_RANGE,
    REGIME_BREAKOUT,
    REGIME_STRESS_CHOP,
)


@dataclass
class SmoothConfig:
    vote_window: int = 5
    persist_bars: int = 3
    switch_penalty: float = 0.10
    prob_margin: float = 0.15
    breakout_hold_bars: int = 2
    stress_drift_max: float = 0.03


def smooth_regimes(
    labels: np.ndarray,
    probs: Optional[np.ndarray],
    config: SmoothConfig,
    features: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Smooth regime labels with persistence, majority vote, and breakout/stress overrides.

    Parameters
    ----------
    labels:
        Base per-bar labels (argmax of probabilities).
    probs:
        Per-bar probabilities aligned with REGIME_ORDER.
    config:
        Smoothing configuration.
    features:
        Optional dict of auxiliary features. Expected keys when available:
        - breakout_confirmed: boolean array
        - slope_norm: float array
        - stress_eligible: boolean array
    """
    if labels is None:
        return labels
    base = np.asarray(labels, dtype=object)
    n = len(base)
    if n == 0:
        return base

    features = features or {}
    breakout_confirmed = np.asarray(features.get("breakout_confirmed", np.zeros(n, dtype=bool)))
    slope_norm = features.get("slope_norm")
    stress_eligible = features.get("stress_eligible")

    if slope_norm is not None:
        slope_norm = np.asarray(slope_norm, dtype=float)
    if stress_eligible is not None:
        stress_eligible = np.asarray(stress_eligible, dtype=bool)

    locked = np.zeros(n, dtype=bool)
    if breakout_confirmed is not None and breakout_confirmed.any():
        hold = max(1, int(config.breakout_hold_bars))
        for idx in np.where(breakout_confirmed)[0]:
            end = min(n, idx + hold)
            locked[idx:end] = True

    smoothed = base.copy()
    smoothed[locked] = REGIME_BREAKOUT

    label_to_idx = None
    if probs is not None:
        label_to_idx = {label: i for i, label in enumerate(REGIME_ORDER)}
        probs = np.asarray(probs, dtype=float)

    prev_label = smoothed[0]
    for t in range(1, n):
        if locked[t]:
            smoothed[t] = REGIME_BREAKOUT
            prev_label = smoothed[t]
            continue

        # majority vote over recent window
        window = max(1, int(config.vote_window))
        start = max(0, t - window + 1)
        window_labels = base[start : t + 1]
        candidate = _mode_label(window_labels, prev_label)

        if candidate != prev_label:
            persist_ok = _run_length(base, t, candidate, int(config.persist_bars)) >= int(config.persist_bars)
            margin_ok = False
            penalty_ok = True
            if probs is not None and label_to_idx:
                cand_idx = label_to_idx.get(candidate)
                prev_idx = label_to_idx.get(prev_label)
                if cand_idx is not None and prev_idx is not None:
                    cand_prob = probs[t, cand_idx]
                    prev_prob = probs[t, prev_idx]
                    margin_ok = (cand_prob - prev_prob) >= float(config.prob_margin)
                    penalty = float(config.switch_penalty)
                    if penalty > 0:
                        penalty_ok = (cand_prob - penalty) >= prev_prob
            if not (persist_ok or margin_ok) or not penalty_ok:
                candidate = prev_label

        # stress-chop gating
        if candidate == REGIME_STRESS_CHOP:
            if stress_eligible is not None and not stress_eligible[t]:
                candidate = prev_label
            if slope_norm is not None and np.isfinite(slope_norm[t]):
                if abs(slope_norm[t]) >= float(config.stress_drift_max):
                    candidate = REGIME_TREND_UP if slope_norm[t] > 0 else REGIME_TREND_DOWN

        smoothed[t] = candidate
        prev_label = candidate

    return smoothed


def _mode_label(window_labels: np.ndarray, prev_label: str) -> str:
    counts: Dict[str, int] = {}
    for label in window_labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    max_count = max(counts.values())
    winners = [label for label, count in counts.items() if count == max_count]
    if len(winners) == 1:
        return winners[0]
    if prev_label in winners:
        return prev_label
    return str(window_labels[-1])


def _run_length(labels: np.ndarray, idx: int, target: str, max_len: int) -> int:
    run = 1
    for j in range(idx - 1, max(-1, idx - max_len), -1):
        if str(labels[j]) == target:
            run += 1
        else:
            break
    return run

