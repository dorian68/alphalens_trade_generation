# AlphaLens model selection (timeframe -> model)

Source of truth: `alphalens_forecast/models/selection.py` (select_model_type).

Auto-selection rules (as implemented in `select_model_type`):
- <= 240 min -> nhits
- > 240 min -> prophet

Examples:
- 15min -> nhits
- 1h -> nhits
- 4h -> nhits
- 1d -> prophet

Notes:
- The API/CLI calls `select_model_type` when `model_type` is not provided
  (see `inference_api.py` and `alphalens_forecast/main.py`).
- You can override by passing `model_type` explicitly. Allowed: nhits, neuralprophet, prophet, tft.
- EGARCH is loaded separately for volatility; it is not part of this mean-model selection.
