"""Documented training cadence for each model family."""
from __future__ import annotations

TRAINING_FREQUENCIES = {
    "nhits": {
        "frequency": "every 2 days",
        "rationale": "high-frequency data drifts quickly; retrain every 48-72h.",
    },
    "neuralprophet": {
        "frequency": "weekly",
        "rationale": "captures intraday seasonality; weekly refresh balances cost and drift.",
    },
    "prophet": {
        "frequency": "weekly",
        "rationale": "lower frequency so weekly retraining is sufficient.",
    },
    "egarch": {
        "frequency": "daily",
        "rationale": "volatility regimes move fast; recalibrate once per trading day.",
    },
    "tft": {
        "frequency": "weekly",
        "rationale": "experimental deep architecture, align with NeuralProphet cadence.",
    },
}

__all__ = ["TRAINING_FREQUENCIES"]
