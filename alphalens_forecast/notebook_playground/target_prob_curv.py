"""Backward-compatible re-export for target probability curve utilities."""
from __future__ import annotations

from alphalens_forecast.core.target_prob_curve import (
    TPFindConfig,
    TargetProbSurface,
    TargetProbabilityCurve,
)

__all__ = ["TargetProbabilityCurve", "TPFindConfig", "TargetProbSurface"]
