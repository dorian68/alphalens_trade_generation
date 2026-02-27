# AlphaLens Forecast

AlphaLens Forecast is a hybrid trading-forecast stack that combines statistical and neural mean models with EGARCH volatility and a Monte Carlo risk layer to emit AlphaLens-ready trade payloads.

## Table of Contents
1. [Overview](#overview)
2. [Feature Highlights](#feature-highlights)
3. [Architecture and Data Flow](#architecture-and-data-flow)
4. [Model Selection and Mean Models](#model-selection-and-mean-models)
5. [Volatility Model (EGARCH)](#volatility-model-egarch)
6. [Quantiles, TP/SL, Risk and Sizing](#quantiles-tpsl-risk-and-sizing)
7. [Monte Carlo Simulation](#monte-carlo-simulation)
8. [Trade Modes and Execution Price](#trade-modes-and-execution-price)
9. [Repository Layout](#repository-layout)
10. [Installation and Setup](#installation-and-setup)
11. [Configuration](#configuration)
12. [CLI Usage](#cli-usage)
13. [Inference API (HTTP)](#inference-api-http)
14. [Surface API (Target Probability)](#surface-api-target-probability)
15. [Data Provider and Caching](#data-provider-and-caching)
16. [Artifacts and Model Storage](#artifacts-and-model-storage)
17. [Backtesting and Evaluation](#backtesting-and-evaluation)
18. [Custom Integrations and Utilities](#custom-integrations-and-utilities)
19. [Scripts](#scripts)
20. [Development and Testing](#development-and-testing)
21. [Troubleshooting](#troubleshooting)
22. [License](#license)

## Overview
AlphaLens Forecast solves end-to-end intraday swing forecasting:

1. Fetch OHLCV via Twelve Data or supply a pre-aligned pandas series.
2. Select a mean model based on timeframe heuristics or explicit overrides.
3. Fit or load an EGARCH volatility model with Student-t innovations.
4. Produce horizon-level price quantiles, then compute TP/SL and risk metrics.
5. Optionally run a Monte Carlo engine to refine TP/SL probabilities and quantiles.
6. Emit a JSON payload with direction, confidence, TP, SL, risk/reward, and sizing.

## Feature Highlights
- Dynamic model routing based on timeframe in `alphalens_forecast/models/selection.py`.
- Support for NHITS, NeuralProphet, Prophet, and experimental TFT models.
- EGARCH volatility forecasting with Student-t distribution and skew estimates.
- Monte Carlo engine for TP/SL hit probability and final-price quantiles.
- RiskEngine payloads with direction, confidence, risk/reward, and position sizing.
- Data caching for Twelve Data with on-demand refresh and range queries.
- Walk-forward backtesting, trajectory export, and performance reports.
- S3-backed model storage for production inference, with local fallback for dev.

## Architecture and Data Flow
1. `DataProvider` loads OHLCV from Twelve Data and merges with cached CSVs.
2. The most recent in-progress candle can be dropped to avoid lookahead.
3. Feature engineering prepares targets and regressors for mean models.
4. `ForecastEngine` loads cached models or trains mean and EGARCH models.
5. EGARCH produces a sigma path, skew, and degrees of freedom for Student-t.
6. Forecasts are produced for each horizon and converted to log-price quantiles.
7. `RiskEngine` converts quantiles into TP/SL, risk/reward, and sizing.
8. Optional Monte Carlo simulation refines quantiles and hit probabilities.
9. Artifacts and summaries can be persisted via `ModelStore` and `ModelRouter`.

## Model Selection and Mean Models
The automatic selection rules are enforced by `alphalens_forecast/models/selection.py`:

- `<= 240min` uses `nhits`.
- `> 240min` uses `prophet`.

Examples:
- `15min` -> `nhits`
- `1h` -> `nhits`
- `4h` -> `nhits`
- `1d` -> `prophet`

Overrides:
- The CLI and API call `select_model_type` when `model_type` is not provided.
- You can override explicitly with `model_type` in the API, or `--eval-model-type` in the CLI.
- Allowed labels: `nhits`, `neuralprophet`, `prophet`, `tft`.
- EGARCH is loaded separately for volatility; it is not part of mean-model selection.

Supported timeframes are those in `FREQ_MAP` in `alphalens_forecast/forecasting.py`:
`1min`, `5min`, `15min`, `30min`, `45min`, `1h`, `2h`, `3h`, `4h`, `6h`, `8h`, `12h`, `1day`.

## Volatility Model (EGARCH)
- EGARCH is trained on log returns of the price series.
- The model outputs a per-step sigma path and a Student-t `dof` and `skew`.
- The forecasted variance path is aggregated per horizon as:
  `sigma_h = sqrt(sum(variance_path[:steps]))`.
- Volatility is used for quantiles, Monte Carlo draws, and position sizing.

## Quantiles, TP/SL, Risk and Sizing
The default quantile and TP/SL logic is implemented in `alphalens_forecast/forecasting.py` and `alphalens_forecast/core/risk_engine.py`.

Quantiles (log-price, Student-t):
- `median_log = log(median_price)` from the mean model.
- `p20 = exp(median_log + t.ppf(0.15) * sigma_h)`
- `p50 = exp(median_log)`
- `p80 = exp(median_log + t.ppf(0.85) * sigma_h)`

Notes:
- The helper `compute_student_t_quantiles` uses 0.15 and 0.85 but labels them as `p20` and `p80` for backward compatibility.

Direction:
- `long` if `median >= last_price`.
- `short` otherwise.

TP/SL mapping:
- `long` -> `TP = p80`, `SL = p20`.
- `short` -> `TP = p20`, `SL = p80`.

Scaling in spot mode:
- If `trade_mode=spot` and an `execution_price` is provided, TP/SL are scaled by
  `scale = entry_price / quantiles_anchor`.
- `quantiles_anchor` defaults to the last close, or the spot entry price when Monte Carlo is enabled.

Risk/reward:
- If `trade_mode=forward` or an explicit `execution_price` exists:
  `risk_reward = abs(tp - entry_price) / abs(entry_price - sl)`.
- Otherwise, the median forecast is used as the reference for the ratio.

Probability fallback:
- If Monte Carlo is disabled, probability is approximated as a linear ratio across the TP/SL span.
- The reference price is `entry_price` in forward mode, `execution_price` if provided, otherwise `median`.

Confidence:
- `confidence = 2 * CDF_t(1, dof) - 1`, clamped to `[0, 1]`.
- If `dof <= 2`, confidence is set to `0.0`.

Position sizing:
- `position_size = min(target_volatility / sigma_h, max_position)`.

## Monte Carlo Simulation
The Monte Carlo engine in `alphalens_forecast/core/montecarlo.py` simulates log-price paths with heavy tails:

- Innovations are drawn from a Student-t distribution with `dof`.
- Optional skew is applied by scaling positive and negative shocks differently.
- Innovations are normalized to zero mean and unit variance.
- A variance-normalization term `t_scale = sqrt((dof - 2) / dof)` keeps scale consistent.
- Per-step sigma is scaled by `sqrt(step_hours)` and applied to shocks.

Outputs per horizon:
- `probability_hit_tp_before_sl`.
- `prob_sl_before_tp`.
- `expected_pnl` based on TP/SL hits and terminal price if neither hits.
- Final-price quantiles `p20`, `p50`, `p80`.

When `trade_mode=forward`, a second Monte Carlo run is executed with entry price set to the median forecast.

## Trade Modes and Execution Price
`trade_mode` is `spot` or `forward`:

- `spot` uses the last close as entry price unless an `execution_price` is provided.
- `forward` uses the median forecast as a conditional entry price.

Execution price sources:
- CLI: execution price is not directly exposed, but can be set via API or by extending the CLI.
- Inference API: `live_price` or `execution_price` in the request body.
- If no execution price is supplied, the inference API attempts to fetch a live price from Twelve Data.

## Repository Layout
```
alphalens_forecast/
├── api/                  # FastAPI surface endpoint for TP/SL target probability curves
├── backtesting.py        # Trajectory recorder and walk-forward evaluation
├── config.py             # AppConfig and environment variable parsing
├── core/                 # Feature engineering, Monte Carlo, risk, probability curve helpers
├── data/                 # Twelve Data client and caching provider
├── forecasting.py        # ForecastEngine orchestration + forecast_from_series
├── models/               # Model definitions, router, selection, TFT experiments
├── reporting/            # Performance report builder
├── risk/                 # TP/SL analysis helpers (quantiles, sensitivity, interpretability)
├── storage/              # S3 store wrapper
├── training.py           # Offline training entrypoints
├── training_schedule.py  # Recommended retrain cadence
├── utils/                # Model store, text/timeseries utilities, Twelve Data client
└── main.py               # CLI entrypoint

scripts and runners
├── inference_api.py      # HTTP inference service for EC2
├── run_forecast.py       # Scripted forecast runner (no CLI)
└── target_prob_curb_runner.py  # Scripted TP/SL surface generator

config/
└── instruments.yml       # Optional instrument/timeframe roster
```

## Installation and Setup
Prereqs: Python 3.9+, git, pip/venv (or conda). Some libraries require native build tools.

```bash
git clone https://github.com/<org>/alphalens_forecast.git
cd alphalens_forecast
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

GPU support (optional):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
export TORCH_DEVICE=cuda
```

## Configuration
AlphaLens reads environment variables from the shell, `.env`, and `alphalens_forecast/.env`.

Example `.env`:
```env
TWELVE_DATA_API_KEY=your_key_here
DEFAULT_SYMBOL=BTC/USD
DEFAULT_INTERVAL=15min
MC_PATHS=3000
TARGET_ANNUAL_VOL=0.2
CONFIDENCE_THRESHOLD=0.6
MAX_POSITION_SIZE=1.0
```

Core variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TWELVE_DATA_API_KEY` | Twelve Data API key. | (empty) |
| `TWELVE_DATA_BASE_URL` | Twelve Data API endpoint. | `https://api.twelvedata.com/time_series` |
| `DEFAULT_SYMBOL` | Default symbol for CLI and scripts. | `BTC/USD` |
| `DEFAULT_INTERVAL` | Default timeframe. | `15min` |
| `DATA_OUTPUT_SIZE` | Max bars requested per API call. | `5000` |
| `TD_RETRY_ATTEMPTS` | Twelve Data retry attempts. | `3` |
| `TD_RETRY_BACKOFF` | Retry backoff multiplier. | `1.5` |
| `USE_MONTECARLO` | Enable Monte Carlo in pipeline. | `true` |
| `MC_PATHS` | Monte Carlo paths. | `3000` |
| `MC_SEED` | Monte Carlo RNG seed. | unset |
| `TARGET_ANNUAL_VOL` | Target volatility for sizing. | `0.20` |
| `CONFIDENCE_THRESHOLD` | Confidence filter threshold. | `0.60` |
| `MAX_POSITION_SIZE` | Cap on position sizing. | `1.0` |
| `DEFAULT_TIMEFRAME` | Risk engine default timeframe. | `15min` |
| `TORCH_DEVICE` | Torch device (`cpu`, `cuda`). | `cpu` |
| `TRAIN_NUM_WORKERS` | DataLoader workers for training. | `0` |
| `TRAIN_PIN_MEMORY` | Pin memory for Torch training. | `false` |
| `TRAIN_PERSISTENT_WORKERS` | Persistent workers in training. | `false` |
| `ALPHALENS_MODEL_DIR` | Base directory for local model artifacts. | `models` |
| `ALPHALENS_MODEL_BUCKET` | S3 bucket for model artifacts. | (empty) |
| `ALPHALENS_MODEL_PREFIX` | S3 prefix for artifacts. | (empty) |
| `ALPHALENS_S3_ONLY` | Require S3 models only. | `false` |
| `ALPHALENS_REQUIRE_S3` | Alias for S3-only mode. | `false` |
| `ALPHALENS_API_HOST` | Inference API host. | `0.0.0.0` |
| `ALPHALENS_API_PORT` | Inference API port. | `8000` |
| `ALPHALENS_LOG_LEVEL` | Inference API log level. | `INFO` |
| `ALPHALENS_DATA_CACHE_DIR` | Inference API data cache override. | (empty) |
| `ALPHALENS_CACHE_DIR` | Inference API cache alias. | (empty) |

Optional instrument universe file:
```yaml
defaults:
  horizons: [3, 6, 12, 24]
instruments:
  - symbol: BTC/USD
    timeframes: [15min, 1h, 4h]
    horizons: [3, 6, 12, 24]
  - symbol: EUR/USD
    timeframes: [15min, 1h]
```

## CLI Usage
Primary entrypoint:
```bash
python -m alphalens_forecast.main --symbol BTC/USD --timeframe 15min --horizons 3 6 12 24
```

Common modes:
- Live forecast is the default; JSON is printed or persisted with `--output`.
- Backtest mode uses `--backtest` to run walk-forward evaluation.
- Eval-only mode uses `--eval-only` to generate a multi-step forecast from an existing model.
- Report-only mode uses `--report-only` and `--report-input` to generate performance metrics.

Key flags:

| Flag | Description |
|------|-------------|
| `--paths` | Monte Carlo paths. |
| `--no-montecarlo` | Disable Monte Carlo simulation. |
| `--save-models` | Persist trained models and manifests. |
| `--model-dir` | Directory for model artifacts. |
| `--data-cache-dir` | Override OHLCV cache directory. |
| `--reuse-model` | Reuse cached models when data hash matches. |
| `--force-retrain` | Ignore cached models and retrain. |
| `--refresh-data` | Force data refresh before training. |
| `--trajectory-output` | Save per-step prediction trajectories. |
| `--eval-only` | Run mean-model evaluation only. |
| `--eval-model-type` | Override mean model for eval (`nhits`, `neuralprophet`, `prophet`). |
| `--eval-steps` | Steps to forecast in eval mode. |
| `--backtest` | Run walk-forward backtest. |
| `--backtest-samples` | Number of windows to evaluate. |
| `--backtest-stride` | Bars between backtest windows. |
| `--backtest-min-history` | Minimum history before first window. |
| `--report-only` | Build a report from saved forecast JSON. |
| `--report-input` | Forecast JSON path for report mode. |
| `--report-actual-input` | CSV of realised prices for report mode. |

Example backtest:
```bash
python -m alphalens_forecast.main --symbol EUR/USD --timeframe 1h --horizons 6 12 24 --backtest --backtest-samples 8
```

Example eval-only run:
```bash
python -m alphalens_forecast.main --symbol BTC/USD --timeframe 15min --eval-only --eval-model-type nhits --eval-steps 48
```

## Inference API (HTTP)
The inference API runs from `inference_api.py` and enforces S3-only model storage.

Start the server:
```bash
python inference_api.py
```

Routes:
- `GET /` or `GET /health` returns status, supported timeframes, and model types.
- `POST /forecast` runs a forecast and returns payload + metadata.

Example cURL calls:
```bash
API=http://127.0.0.1:8000

# Health check
curl -s "$API/health" | jq .
```

```bash
API=http://127.0.0.1:8000

# Basic forecast (defaults to config when fields omitted)
curl -s "$API/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "horizons": [6, 12, 24],
    "use_montecarlo": true,
    "paths": 3000,
    "trade_mode": "spot",
    "include_metadata": true,
    "include_predictions": false
  }' | jq .
```

```bash
API=http://127.0.0.1:8000

# Override mean model, provide live price, include model status
curl -s "$API/forecast?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USD",
    "timeframe": "15min",
    "horizons": [3, 6, 12],
    "model_type": "nhits",
    "trade_mode": "forward",
    "live_price": 51234.5,
    "include_predictions": true,
    "include_model_info": true,
    "force_retrain": false,
    "refresh_data": false
  }' | jq .
```

Request fields for `POST /forecast`:

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Trading symbol (e.g., `BTC/USD`). |
| `timeframe` | string | Timeframe (must be in `FREQ_MAP`). |
| `horizons` | list[int] | Horizon hours. Defaults to config. |
| `use_montecarlo` | bool | Enable Monte Carlo. |
| `paths` | int | Monte Carlo paths. |
| `model_type` | string | Override mean model (`nhits`, `neuralprophet`, `prophet`, `tft`). |
| `trade_mode` | string | `spot` or `forward`. |
| `live_price` | float | Use as execution price. |
| `execution_price` | float | Legacy alias for `live_price`. |
| `include_predictions` | bool | Include per-step predictions. |
| `include_metadata` | bool | Include run metadata. |
| `include_model_info` | bool | Include model load status. |
| `force_retrain` | bool | Force model retraining. |
| `refresh_data` | bool | Refresh OHLCV before training. |

If `live_price` or `execution_price` are not provided, the API attempts to fetch a live price from Twelve Data. If that fails, it falls back to the last close.

Example request:
```json
{
  "symbol": "EUR/USD",
  "timeframe": "1h",
  "horizons": [6, 12, 24],
  "use_montecarlo": true,
  "paths": 3000,
  "trade_mode": "spot",
  "include_metadata": true,
  "include_predictions": false
}
```

Example response shape:
```json
{
  "ok": true,
  "status": "ok",
  "message": "forecast completed",
  "request_id": "...",
  "timestamp": "...",
  "request": {"symbol": "EUR/USD", "timeframe": "1h", "horizons": [6, 12, 24]},
  "warnings": [],
  "data": {
    "payload": {"symbol": "EUR/USD", "asOf": "...", "horizons": ["..."]},
    "as_of": "...",
    "data_hash": "...",
    "durations": {"total_seconds": 12.3},
    "metadata": {"mean_model": {"name": "..."}},
    "model_status": {"mean": {"loaded": true}, "vol": {"loaded": true}},
    "total_seconds": 12.3
  }
}
```

## Surface API (Target Probability)
The surface API in `alphalens_forecast/api/surface_api.py` builds TP/SL surfaces by solving for TP levels that meet target hit probabilities.

Run locally:
```bash
uvicorn alphalens_forecast.api.surface_api:app --reload --host 0.0.0.0 --port 8000
```

Important request fields:
- `symbol`, `timeframe`.
- `entry_price` (optional override; default last close).
- `horizon_hours` or `steps`.
- `paths`, `dof`, `skew`.
- `direction` (`long` or `short`).
- `methodology` (`legacy` uses sigma_ref from EWMA, `research` uses sample mean/std).
- `target_prob` range spec (min, max, steps).
- `sl_sigma` range spec (min, max, steps).

Response includes:
- `sigma_ref` and `atr` for context.
- `surface.target_probs`, `surface.sl_sigma`, and `surface.tp_sigma` grids.

## Data Provider and Caching
- `DataProvider` caches OHLCV under `alphalens_forecast/data/cache/{symbol}/{timeframe}.csv`.
- Symbol normalization converts `BASE_QUOTE` to `BASE/QUOTE` for Twelve Data calls when safe.
- Auto-refresh optionally requests missing bars and merges them into cache.
- Range queries are supported with `start`, `end`, and `range_cache` options.
- `range_cache` can be `none` (default), `merge`, or `separate`.

## Artifacts and Model Storage
Two storage layers are used:

1. `ModelRouter` handles model training artifacts per model type and symbol/timeframe.
2. `ModelStore` writes consolidated artifacts and payloads for audits.

Local model layout:
- `models/{model_type}/{symbol_slug}/{timeframe_slug}/` with `model.*`, `metadata.json`, and `metrics.json`.

S3 integration:
- `ALPHALENS_MODEL_BUCKET` and `ALPHALENS_MODEL_PREFIX` enable S3 artifact storage.
- `ALPHALENS_S3_ONLY` or `ALPHALENS_REQUIRE_S3` forces S3-only loading.
- The inference API enforces S3-only mode by default.

## Backtesting and Evaluation
- `BacktestRunner` performs walk-forward evaluation on historical windows.
- Metrics include RMSE, MAE, and direction accuracy per horizon.
- Direction accuracy modes: `v1` = step-to-step agreement, `v2` = anchor-to-horizon, `v3` = v2 with a deadzone (default: v1 via `DIRECTION_ACCURACY_MODE`).
- `TrajectoryRecorder` exports per-step predictions for each horizon.
- `reporting/performance.py` aggregates metrics and coverage for p20/p80 bounds.
- Reports include extended direction/coverage metrics by default (set `REPORTING_EXTENDED_METRICS=0` to disable).

## Custom Integrations and Utilities
- `forecast_from_series` lets you bypass the data provider and run a forecast from a pandas Series.
- `risk/sl_tp_analysis.py` provides `compute_sl_tp_from_quantiles`, `analyze_sl_tp_sensitivity`, and `interpret_sl_tp`.
- Utilities in `alphalens_forecast/utils` provide text normalization, time alignment, and S3 helpers.

## Scripts
- `run_forecast.py` runs a full forecast without CLI flags using a `RunOverrides` dataclass.
- `target_prob_curb_runner.py` generates TP/SL probability surfaces without FastAPI.
- `inference_api.py` starts the HTTP inference server.

## Development and Testing
- There is no formal test suite; validate with `--backtest`, `--eval-only`, and notebooks.
- Suggested tooling: `ruff` and `black` for formatting and linting.
- Heavy dependencies (Prophet, Torch, Darts) may require native build tools.

## Troubleshooting
- Missing Twelve Data API key will return HTTP 401 from data fetches.
- Prophet build issues typically require `cmdstanpy` toolchain or prebuilt wheels.
- CUDA out of memory can be mitigated by lowering `MC_PATHS` or running on CPU.
- Model reuse mismatch can be fixed by clearing `models/` or disabling `--reuse-model`.
- Inference API missing models can be resolved by training with `--save-models` and pushing to S3.
- Backtest failures often mean insufficient history; increase `DATA_OUTPUT_SIZE` or lower horizons.

## License
MIT (customize as needed).
