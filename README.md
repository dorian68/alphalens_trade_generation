# AlphaLens Forecast

AlphaLens Forecast is a hybrid trading-forecast stack that combines Prophet, NeuralProphet, N-HiTS, TFT, EGARCH volatility modelling, and a Monte Carlo/Risk engine to ship consistent AlphaLens payloads for discretionary or systematic desks.

## Table of Contents
1. [Overview](#overview)
2. [Feature Highlights](#feature-highlights)
3. [Tech Stack](#tech-stack)
4. [Repository Layout](#repository-layout)
5. [Installation & Setup](#installation--setup)
6. [Configuration](#configuration)
7. [Data Flow & Runtime Pipeline](#data-flow--runtime-pipeline)
8. [CLI Usage](#cli-usage)
9. [Model Training & Scheduling](#model-training--scheduling)
10. [Monte Carlo & Risk Controls](#monte-carlo--risk-controls)
11. [Backtesting, Evaluation & Reporting](#backtesting-evaluation--reporting)
12. [Custom Integrations & Utilities](#custom-integrations--utilities)
13. [Surface API](#surface-api)
14. [Data Provider & Artifacts](#data-provider--artifacts)
15. [Development & Testing](#development--testing)
16. [Troubleshooting](#troubleshooting)
17. [License](#license)

## Overview
AlphaLens Forecast solves end-to-end intraday swing forecasting:

- Fetch OHLCV via Twelve Data (or pre-aligned custom series) and persist it locally.
- Choose the mean-model automatically based on timeframe heuristics (NHITS < 30m, NeuralProphet intraday, Prophet 4h+; TFT optional).
- Fit or load an EGARCH(1,1) volatility model (Student-t innovations) for fat-tail awareness.
- Run a Monte Carlo engine that evaluates per-horizon TP/SL probabilities, quantiles, and path-wise diagnostics.
- Feed everything into the RiskEngine to emit direction, confidence, position sizing, take-profit/stop-loss, and risk/reward.
- Optionally persist artifacts, export trajectories, or run walk-forward diagnostics and performance reports.

## Feature Highlights
- **Dynamic model routing:** `alphalens_forecast/models/selection.py` chooses NHITS/NeuralProphet/Prophet based on cadence. Override with `--eval-model-type`.
- **ModelRouter-aware storage:** `models/{model}/{symbol}/{timeframe}` layout plus metadata/manifest tracking and re-use switches (`--save-models`, `--reuse-model`).
- **TFT experimentation:** `alphalens_forecast/models/tft_model.py` exposes a Temporal Fusion Transformer path for longer context, sequence-to-sequence experimentation.
- **Volatility-aware Monte Carlo:** Monte Carlo simulator (`alphalens_forecast/core/montecarlo.py`) leverages EGARCH sigma paths and Student-t draws to compute hit probabilities and quantiles.
- **Full diagnostics:** `TrajectoryRecorder` for per-horizon paths, walk-forward backtests (`alphalens_forecast/backtesting.py`), and PDF-ready performance reports (`alphalens_forecast/reporting`).
- **Risk orchestration:** `alphalens_forecast/core/risk_engine.py` produces ready-to-trade payloads with risk-reward, probability of success, and volatility-scaling sizing.
- **TP/SL analysis helpers:** `alphalens_forecast/risk/sl_tp_analysis.py` exposes quantile-based SL/TP, sensitivity sweeps, optional Monte Carlo consistency checks, and interpretability metrics.
- **Cache-first data access:** The `DataProvider` keeps `alphalens_forecast/data/cache/{symbol}/{timeframe}.csv` synced, so CLI calls stay cheap and reproducible.

## Tech Stack
- **Core libraries:** pandas, numpy, scipy, tqdm, matplotlib.
- **Forecasting:** Darts (NHITS), NeuralProphet, Prophet, TFT, PyTorch Forecasting, torch.
- **Volatility:** `arch` for EGARCH.
- **Configuration:** `python-dotenv`, YAML.
- **CLI/reporting:** argparse, JSON, dataclasses.
See `requirements.txt` for exact versions; GPU acceleration is optional (`TORCH_DEVICE=cuda`).

## Repository Layout
```
alphalens_forecast/
├── core/                 # Feature engineering, Monte Carlo, risk, volatility helpers
├── risk/                 # TP/SL analysis helpers (quantiles, sensitivity, interpretability)
├── data/                 # Twelve Data client and caching provider
├── forecasting.py        # ForecastEngine orchestration + forecast_from_series
├── backtesting.py        # Trajectory recorder and walk-forward evaluation
├── reporting/            # Performance report builder
├── models/               # Model definitions, router, selection and TFT experiments
├── training.py           # Offline training entrypoints for NHITS/NeuralProphet/Prophet/EGARCH
├── training_schedule.py  # Documented cadences per model family
├── utils/                # Model store, text/timeseries utilities, Twelve Data client
├── config.py             # AppConfig + instrument universe helpers
├── main.py               # CLI entrypoint (live forecast, backtest, eval, report)
├── test.py               # Example holdout evaluation workflow
├── requirements.txt
└── README.md

config/
└── instruments.yml       # Optional instrument/timeframe roster (fallback provided)
```

## Installation & Setup
Prereqs: Python 3.9+, git, pip/venv (or conda). Prophet/PyTorch may require system build tools (gcc, pystan toolchain) and optionally CUDA for GPU.

```bash
git clone https://github.com/<org>/alphalens_forecast.git
cd alphalens_forecast
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to use a GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA tag
export TORCH_DEVICE=cuda
```

## Configuration
AlphaLens reads environment variables from the shell, `.env`, and `alphalens_forecast/.env`. Create a `.env` file at the project root:

```env
TWELVE_DATA_API_KEY=your_key_here
DEFAULT_SYMBOL=BTC/USD
DEFAULT_INTERVAL=15min
MC_PATHS=3000
TARGET_ANNUAL_VOL=0.2
CONFIDENCE_THRESHOLD=0.6
MAX_POSITION_SIZE=1.0
```

| Variable | Description | Default |
|----------|-------------|---------|
| `TWELVE_DATA_API_KEY` | Twelve Data key used by `DataProvider`. | (empty) |
| `TWELVE_DATA_BASE_URL` | Override API endpoint. | `https://api.twelvedata.com/time_series` |
| `DEFAULT_SYMBOL` | CLI fallback symbol. | `BTC/USD` |
| `DEFAULT_INTERVAL` | CLI fallback timeframe. | `15min` |
| `DATA_OUTPUT_SIZE` | Max bars requested per call. | `5000` |
| `TD_RETRY_ATTEMPTS` / `TD_RETRY_BACKOFF` | API retry policy. | `3 / 1.5` |
| `USE_MONTECARLO` | Toggle Monte Carlo block. | `true` |
| `MC_PATHS` / `MC_SEED` | Monte Carlo draws and RNG seed. | `3000 / unset` |
| `TARGET_ANNUAL_VOL` | Volatility target for sizing. | `0.20` |
| `CONFIDENCE_THRESHOLD` | Signal publish threshold. | `0.60` |
| `MAX_POSITION_SIZE` | Cap on position sizing. | `1.0` |
| `DEFAULT_TIMEFRAME` | Risk engine fallback timeframe. | `15min` |
| `TORCH_DEVICE` | Torch device (`cpu`, `cuda`). | `cpu` |
| `ALPHALENS_MODEL_BUCKET` | S3 bucket for model artifacts. | (empty) |
| `ALPHALENS_MODEL_PREFIX` | Optional S3 prefix for model artifacts. | (empty) |
| `ALPHALENS_S3_ONLY` / `ALPHALENS_REQUIRE_S3` | Require models from S3 only; disables local fallback and requires S3 for uploads. | `false` |


S3-only mode (production):

```env
ALPHALENS_MODEL_BUCKET=your_bucket
ALPHALENS_MODEL_PREFIX=production
ALPHALENS_S3_ONLY=true
```

When enabled, the runtime will only load models present in S3 (no local fallback) and requires S3 for uploads. Missing models can be retrained and then persisted back to S3, but requests still fail if S3 is unavailable.

Optional: populate `config/instruments.yml` to control the universe:

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
If absent, the app falls back to a default G10 + BTC/ETH list.

## Data Flow & Runtime Pipeline
1. **DataProvider** (`alphalens_forecast/data/provider.py`) fetches OHLCV via Twelve Data, merges it with cached CSVs (`alphalens_forecast/data/cache`), and returns a pandas frame containing close, log returns, etc.
2. **Feature Engineering** (`alphalens_forecast/core/feature_engineering.py`) assembles regressors/targets for the chosen mean model.
3. **Model selection/training** happens through `ForecastEngine`. It loads cached models from `ModelRouter` or trains new ones via `train_nhits`, `train_neuralprophet`, `train_prophet`, or `train_egarch`.
4. **Monte Carlo Simulator** draws Student-t shocks scaled by EGARCH sigma to test TP vs SL hit order and compute quantiles (`p20`, `p50`, `p80`).
5. **RiskEngine** translates the statistical forecasts into AlphaLens payloads with direction, TP, SL, probability of success, risk/reward, and volatility-targeted sizing.
6. **Reporting/Artifacts** optionally persist manifests, JSON payloads, trajectory exports, and run summaries for auditability (`utils/model_store.py`).

## CLI Usage
Primary entrypoint:
```bash
python -m alphalens_forecast.main --symbol BTC/USD --timeframe 15min --horizons 3 6 12 24
```

Mode highlights:

| Mode | Flags | Notes |
|------|-------|-------|
| Live forecast | *(default)* | Streams JSON payload to stdout. Use `--output` to persist. |
| Backtest | `--backtest [--backtest-samples 8 --backtest-output backtest.json --backtest-stride 96]` | Walk-forward retraining with RMSE/MAE/direction accuracy per horizon. |
| Eval-only | `--eval-only --eval-model-type nhits --eval-steps 48 --eval-output eval.json` | Load an existing model and generate multi-step trajectories without Monte Carlo. |
| Report-only | `--report-only --report-input forecast.json --report-actual-input realised.csv --report-output perf.json` | Builds a coverage/metrics report without re-running forecasts. |
| Artifact persistence | `--save-models --model-dir /mnt/artifacts --reuse-model` | Persist/reuse trained models + manifests and emit run summaries. |
| Trajectory export | `--trajectory-output traj.json` | Stores per-step predictions for downstream plotting. |

Other useful flags:
- `--paths 5000` to change Monte Carlo draws.
- `--no-montecarlo` to skip the simulator when you only need deterministic quantiles.
- `--data-cache-dir /mnt/cache_fx` to relocate the OHLCV cache.
- `--log-level DEBUG --debug` for verbose tracing.

The CLI prints a JSON payload matching the AlphaLens schema (`symbol`, `asOf`, per-horizon forecast, TP/SL, risk-reward, confidence, `position_size`).

## Model Training & Scheduling
- Reusable training helpers live in `alphalens_forecast/training.py` (`train_nhits`, `train_neuralprophet`, `train_prophet`, `train_egarch`).
- Cadence recommendations sit in `alphalens_forecast/training_schedule.py` (NHITS ~48-72h, NeuralProphet/Prophet weekly, EGARCH daily, TFT weekly).
- Kick off offline refreshes via cron, AWS Batch, or notebooks:

```python
from alphalens_forecast.training import train_nhits, train_neuralprophet, train_prophet, train_egarch

symbol = "EUR/USD"
for timeframe in ("15min", "1h", "4h"):
    train_nhits(symbol, timeframe)
    train_neuralprophet(symbol, timeframe)
    train_prophet(symbol, timeframe)
    train_egarch(symbol, timeframe)
```

Saved artifacts are loaded automatically by `ModelRouter` when the CLI runs. Combine with `--reuse-model` to skip retraining if the input data hash matches the stored manifest.

## Monte Carlo & Risk Controls
- Monte Carlo settings derive from `MC_PATHS`, `MC_SEED`, and CLI overrides. Each path draws Student-t shocks scaled by EGARCH sigma to test TP vs SL hit order and compute quantiles (`p20`, `p50`, `p80`).
- RiskEngine consumes `HorizonForecast` objects, computes direction (`median >= last_price` => long), risk-reward ratios, TP/SL, hit probabilities, and volatility-targeted `position_size` capped by `MAX_POSITION_SIZE`.
- Confidence is expressed as the probability the Student-t error stays within ±1σ, so low `dof` outputs get penalised automatically.

## Backtesting, Evaluation & Reporting
- `--backtest` enables walk-forward evaluation with configurable stride/anchors (`alphalens_forecast/backtesting.py`). Output JSON includes per-horizon metrics and aggregate statistics; pass `--backtest-output` to persist.
- `TrajectoryRecorder` captures per-step predictions for each horizon. Combine with `--trajectory-output` or consume programmatically for plotting or holdout comparison.
- `alphalens_forecast/reporting/performance.py` exposes `generate_performance_report`, which aligns actual vs predicted series, computes RMSE/MAE/MAPE/direction accuracy, coverage stats for `p20/p80`, residual skew/kurtosis, and volatility summaries. Use `--report-only` to drive it from saved JSON files.

## Custom Integrations & Utilities
- `forecast_from_series` lets you bypass the data provider and supply a pre-aligned pandas `Series` (e.g., Kaggle BTC dataset). You can pass pre-fitted models (`mean_model`, `vol_model`) or let it fit on the fly.
- `TrajectoryRecorder` + `evaluate_trajectory` (see `alphalens_forecast/test.py`) show how to inspect holdout windows, plot predicted vs realised prices, and compute RMSE/MAE/direction accuracy per horizon.
- Utilities in `alphalens_forecast/utils` include `ModelStore` (persist artifacts & manifests), `series_to_price_frame`/`align_series_to_timeframe`, and a lightweight Twelve Data REST client.

TP/SL analysis helpers live in `alphalens_forecast/risk/sl_tp_analysis.py` for quantile-based SL/TP, sensitivity sweeps, optional Monte Carlo checks, and interpretability helpers.

```python
from alphalens_forecast.risk.sl_tp_analysis import (
    compute_sl_tp_from_quantiles,
    analyze_sl_tp_sensitivity,
)

sl_tp = compute_sl_tp_from_quantiles(
    last_price=last_price,
    median_log_price=median_log,
    sigma_h=sigma_h,
    dof=dof,
    direction="long",
    q_low=0.2,
    q_high=0.8,
)

sweep = analyze_sl_tp_sensitivity(
    {"last_price": last_price, "median_log_price": median_log, "sigma_h": sigma_h, "dof": dof, "horizon": steps},
    vary="dof",
    grid=[4, 6, 8, 12],
)
```

## Surface API
Run the surface API locally:

```bash
uvicorn alphalens_forecast.api.surface_api:app --reload --host 0.0.0.0 --port 8000
```

Example request with an explicit entry_price (omit entry_price to use the latest close):

```bash
curl -X POST http://localhost:8000/surface \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EUR/USD","timeframe":"4h","horizon_hours":48,"entry_price":1.165,"target_prob":{"min":0.2,"max":0.8,"steps":7},"sl_sigma":{"min":0.5,"max":2.0,"steps":7}}'
```

## Data Provider & Artifacts
- Default cache: `alphalens_forecast/data/cache/{symbol}/{timeframe}.csv`. Override with `--data-cache-dir` or `ALPHALENS_DATA_CACHE`.
- Model artifacts: `models/{model_type}/{symbol_slug}/{timeframe_slug}/model.*` (NeuralProphet emits `.np`, other models default to `.pth`). EGARCH checkpoints live alongside mean models.
- Lightning logs: `lightning_logs/` (if you train TFT or other PyTorch Lightning models).
- Monte Carlo/forecast outputs can be stored via `--output`, `--trajectory-output`, and run summaries created by `write_run_summary` in `main.py`.

## Development & Testing
- There is no strict test suite yet; validation is performed via `--backtest`, `--eval-only`, and manual notebooks (`alphalens_forecast/test.py`). Consider adding `pytest` coverage around `ForecastEngine` and `RiskEngine` when integrating in larger systems.
- Linting/formatting is not enforced, but sticking to `ruff`/`black` defaults keeps contributions consistent.
- Heavy dependencies (Prophet, Torch, Darts) may need additional system packages (gcc, g++, `cmdstanpy` toolchain). On Debian/Ubuntu: `sudo apt install build-essential python3-dev`.

## Troubleshooting
- **Missing Twelve Data API key:** set `TWELVE_DATA_API_KEY` or expect HTTP 401 responses when fetching OHLCV.
- **Prophet build issues:** ensure `cmdstanpy` toolchain is installed or use the precompiled wheels available for manylinux.
- **CUDA out of memory:** lower `MC_PATHS`, reduce horizons, or run on CPU by setting `TORCH_DEVICE=cpu`.
- **Model reuse mismatch:** delete stale entries under `models/` or disable `--reuse-model` until the cache is refreshed.
- **S3-only mode errors:** ensure `ALPHALENS_MODEL_BUCKET` is set, AWS credentials are available, and `boto3` is installed; confirm the model artifacts exist under `s3://<bucket>/<prefix>/<symbol>/<timeframe>/<model_type>/`.
- **Not enough history for backtests:** extend `DATA_OUTPUT_SIZE`, refresh the cache (`--data-cache-dir ... --backtest --backtest-min-history <bigger>`), or reduce horizons.

## License
MIT (customise as needed for your organisation).
