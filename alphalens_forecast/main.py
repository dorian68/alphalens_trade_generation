"""Command-line entry point for AlphaLens hybrid forecasting."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from alphalens_forecast.config import AppConfig, get_config
from alphalens_forecast.data import DataProvider
from alphalens_forecast.forecasting import FREQ_MAP, ForecastEngine, OrchestrationResult
from alphalens_forecast.models import ModelRouter
from alphalens_forecast.reporting import generate_performance_report
from alphalens_forecast.backtesting import BacktestRunner, TrajectoryRecorder
from alphalens_forecast.models.selection import select_model_type
from alphalens_forecast.utils.model_store import ModelStore, StoredArtifacts
from alphalens_forecast.utils.text import slugify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("alphalens_forecast")


def _str_to_bool(value: Optional[str]) -> bool:
    """Parse common string representations of booleans."""
    if value is None:
        return True
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean.")


def configure_logging(log_level: str, debug: bool) -> None:
    """Configure root logger level once CLI arguments are known."""
    level = logging.DEBUG if debug else getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logger.setLevel(level)
    logger.debug("Logging configured | level=%s debug=%s", logging.getLevelName(level), debug)


def resolve_model_directory(candidate: Optional[str]) -> Path:
    """Resolve the directory used to persist trained models with AWS-safe defaults."""
    if candidate:
        base = Path(candidate).expanduser()
    else:
        default_root = Path(os.environ.get("ALPHALENS_MODEL_DIR", "models"))
        base = default_root
    if base.is_absolute():
        resolved = base
    else:
        resolved = Path("/tmp") / base if "AWS_EXECUTION_ENV" in os.environ else Path.cwd() / base
    resolved.mkdir(parents=True, exist_ok=True)
    logger.debug("Resolved model directory: %s", resolved)
    return resolved


def format_bytes(num_bytes: int) -> str:
    """Render file size in human readable units."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def parse_args(config: AppConfig) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AlphaLens Forecasting CLI")
    parser.add_argument("--symbol", default=config.twelve_data.symbol, help="Trading symbol to fetch.")
    parser.add_argument(
        "--timeframe",
        default=config.twelve_data.interval,
        help="Twelve Data timeframe (e.g., 15min, 1h).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=config.monte_carlo.horizon_hours,
        help="Forecast horizons in hours.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=config.monte_carlo.paths,
        help="Monte Carlo paths.",
    )
    parser.add_argument(
        "--no-montecarlo",
        action="store_true",
        help="Disable Monte Carlo simulation.",
    )
    parser.add_argument(
        "--save-models",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Persist trained models and metadata (default: False).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory where trained models and manifests are written.",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        default=None,
        help="Directory where OHLCV cache files are stored (default: alphalens_forecast/data/cache).",
    )
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Reuse the most recent saved models when data hash matches.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file for the resulting JSON payload.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose DEBUG logging output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--report-input",
        type=str,
        default=None,
        help="Path to a forecast JSON file (with ds/median/p20/p80) to evaluate.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Optional output path for the generated performance report.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate a performance report from --report-input and exit.",
    )
    parser.add_argument(
        "--report-actual-input",
        type=str,
        default=None,
        help="Optional CSV providing realised prices (datetime, close) for report mode.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=str,
        default=None,
        help="Optional path to write per-step trajectory forecasts for each horizon.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load an existing mean model and produce multi-step forecasts for evaluation.",
    )
    parser.add_argument(
        "--eval-model-type",
        type=str,
        default=None,
        help="Model type label to evaluate (nhits, neuralprophet, prophet). Defaults to timeframe-based selection.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=24,
        help="Number of forward steps to forecast when using --eval-only.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Optional JSON path for evaluation forecasts (otherwise printed).",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a walk-forward backtest instead of generating a live forecast.",
    )
    parser.add_argument(
        "--backtest-samples",
        type=int,
        default=10,
        help="Number of evaluation windows to simulate when using --backtest (default: 10).",
    )
    parser.add_argument(
        "--backtest-stride",
        type=int,
        default=None,
        help="Number of bars between backtest windows (defaults to the largest horizon length).",
    )
    parser.add_argument(
        "--backtest-min-history",
        type=int,
        default=500,
        help="Minimum number of observations before the first backtest window.",
    )
    parser.add_argument(
        "--backtest-output",
        type=str,
        default=None,
        help="Optional JSON path for the backtest report (otherwise printed).",
    )
    return parser.parse_args()


def orchestrate(
    config: AppConfig,
    args: argparse.Namespace,
    *,
    model_store: Optional[ModelStore],
    model_dir: Path,
    trajectory_recorder: Optional[TrajectoryRecorder] = None,
) -> OrchestrationResult:
    """Run the full forecasting workflow and return enriched artifacts."""
    cache_dir = Path(args.data_cache_dir).expanduser() if args.data_cache_dir else None
    data_provider = DataProvider(config.twelve_data, cache_dir=cache_dir)
    model_router = ModelRouter(model_dir)
    engine = ForecastEngine(config, data_provider, model_router)
    use_montecarlo = not args.no_montecarlo and config.monte_carlo.use_montecarlo
    return engine.forecast(
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizons=args.horizons,
        paths=args.paths,
        use_montecarlo=use_montecarlo,
        reuse_cached=args.reuse_model,
        model_store=model_store,
        show_progress=True,
        trajectory_recorder=trajectory_recorder,
    )


def persist_artifacts(
    model_store: ModelStore,
    model_dir: Path,
    symbol: str,
    timeframe: str,
    result: OrchestrationResult,
) -> Optional[StoredArtifacts]:
    """Persist trained models, their metadata, and manifest files."""
    if result.mean_model is None or result.vol_model is None:
        logger.warning("Artifacts not saved because models are unavailable.")
        return None

    symbol_slug = slugify(symbol)
    timeframe_slug = slugify(timeframe)
    prefix = f"{symbol_slug}_{timeframe_slug}_{result.run_timestamp_slug}"

    metadata = dict(result.metadata)
    metadata["model_dir"] = str(model_dir)
    metadata["payload_hash"] = hashlib.sha256(
        json.dumps(result.payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    try:
        stored = model_store.save(
            prefix=prefix,
            mean_model=result.mean_model,
            vol_model=result.vol_model,
            metadata=metadata,
            payload=result.payload,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist model artifacts for %s [%s]", symbol, timeframe)
        return None

    try:
        model_size = stored.model_path.stat().st_size
        manifest_size = stored.manifest_path.stat().st_size
        logger.info(
            "Model artifact saved to %s (%s)",
            stored.model_path,
            format_bytes(model_size),
        )
        logger.info(
            "Manifest saved to %s (%s)",
            stored.manifest_path,
            format_bytes(manifest_size),
        )
    except OSError as exc:
        logger.warning("Unable to determine artifact file sizes: %s", exc)

    return stored


def write_run_summary(
    model_dir: Path,
    symbol: str,
    timeframe: str,
    horizons: Iterable[int],
    data_hash: Optional[str],
    start_time: datetime,
    end_time: datetime,
    success: bool,
    use_montecarlo: bool,
    artifact_paths: Optional[StoredArtifacts],
    reuse_applied: bool,
) -> None:
    """Emit a JSON run summary next to the saved artifacts."""
    duration = (end_time - start_time).total_seconds()
    summary = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": list(horizons),
        "start_time": start_time.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "end_time": end_time.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "duration_seconds": duration,
        "success": success,
        "reuse_cached_artifacts": reuse_applied,
        "use_montecarlo": use_montecarlo,
        "data_hash": data_hash,
    }
    if artifact_paths is not None:
        summary["artifacts"] = {
            "model_path": str(artifact_paths.model_path),
            "manifest_path": str(artifact_paths.manifest_path),
        }

    filename = (
        f"{slugify(symbol)}_{slugify(timeframe)}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_run_summary.json"
    )
    path = model_dir / filename
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Run summary written to %s", path)
    except OSError:
        logger.exception("Failed to write run summary to %s", path)


def main() -> None:
    """Entrypoint."""
    config = get_config()
    args = parse_args(config)
    configure_logging(args.log_level, args.debug)
    logger.debug("Arguments: %s", vars(args))

    if args.backtest and (args.eval_only or args.report_only):
        raise ValueError("Backtest mode cannot be combined with --eval-only or --report-only.")
    if args.eval_only and args.report_only:
        raise ValueError("Cannot enable both --eval-only and --report-only.")
    if args.backtest:
        run_backtest_mode(config=config, args=args)
        return
    if args.eval_only:
        run_eval_mode(config=config, args=args)
        return
    if args.report_input and args.report_only:
        run_report_mode(config=config, args=args)
        return

    model_dir = resolve_model_directory(args.model_dir)
    model_store: Optional[ModelStore] = None
    if args.save_models or args.reuse_model:
        model_store = ModelStore(model_dir, logger)
    trajectory_recorder = TrajectoryRecorder() if args.trajectory_output else None

    wall_start = datetime.now(timezone.utc)
    try:
        result = orchestrate(
            config,
            args,
            model_store=model_store,
            model_dir=model_dir,
            trajectory_recorder=trajectory_recorder,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Forecast pipeline failed")
        if args.save_models and model_store is not None:
            write_run_summary(
                model_dir=model_dir,
                symbol=args.symbol,
                timeframe=args.timeframe,
                horizons=args.horizons,
                data_hash=None,
                start_time=wall_start,
                end_time=datetime.now(timezone.utc),
                success=False,
                use_montecarlo=not args.no_montecarlo and config.monte_carlo.use_montecarlo,
                artifact_paths=None,
                reuse_applied=False,
            )
        raise

    artifact_paths = None
    if args.save_models and model_store and not result.used_cached_artifacts:
        artifact_paths = persist_artifacts(
            model_store=model_store,
            model_dir=model_dir,
            symbol=args.symbol,
            timeframe=args.timeframe,
            result=result,
        )
    elif args.save_models and result.used_cached_artifacts:
        logger.info("Reuse flag active and cached artifacts served; skipping save.")

    if args.save_models:
        write_run_summary(
            model_dir=model_dir,
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizons=args.horizons,
            data_hash=result.data_hash,
            start_time=wall_start,
            end_time=datetime.now(timezone.utc),
            success=True,
            use_montecarlo=result.metadata.get("use_montecarlo", False),
            artifact_paths=artifact_paths,
            reuse_applied=result.used_cached_artifacts,
        )

    json_output = json.dumps(result.payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(json_output)
        logger.info("Results written to %s", args.output)
    else:
        print(json_output)

    if args.trajectory_output:
        if not result.trajectories:
            logger.warning("Trajectory output requested but no trajectories were captured.")
        else:
            Path(args.trajectory_output).write_text(
                json.dumps(result.trajectories, indent=2),
                encoding="utf-8",
            )
            logger.info("Trajectory data written to %s", args.trajectory_output)


def run_report_mode(config: AppConfig, args: argparse.Namespace) -> None:
    """Generate a performance report from an existing forecast file."""
    if not args.report_input:
        raise ValueError("--report-input is required for report mode.")
    actual_series = _load_actual_series(config, args)
    payload = _load_forecast_payload(Path(args.report_input))
    predicted_series, quantiles = _extract_forecast_series(payload)
    overlap = actual_series.index.intersection(predicted_series.index)
    if overlap.empty:
        raise ValueError(
            "No overlap between realised prices and forecast timestamps. "
            "Provide a CSV with realised closes via --report-actual-input or rerun once the forecast horizon has elapsed."
        )
    metadata = {
        "symbol": payload.get("symbol", args.symbol),
        "timeframe": payload.get("timeframe", args.timeframe),
        "source_file": args.report_input,
        "n_predictions": len(predicted_series),
    }
    report = generate_performance_report(
        actual=actual_series,
        predicted=predicted_series,
        quantiles=quantiles,
        metadata=metadata,
    )
    report_json = json.dumps(report.to_dict(), indent=2, default=str)
    if args.report_output:
        Path(args.report_output).write_text(report_json, encoding="utf-8")
        logger.info("Performance report written to %s", args.report_output)
    else:
        print(report_json)


def run_eval_mode(config: AppConfig, args: argparse.Namespace) -> None:
    """Load an existing mean model checkpoint and emit multi-step forecasts."""
    model_type = (args.eval_model_type or select_model_type(args.timeframe)).lower()
    model_dir = resolve_model_directory(args.model_dir)
    router = ModelRouter(model_dir)
    try:
        model = router.load_model(model_type, args.symbol, args.timeframe)
    except FileNotFoundError:
        model = None
    if model is None:
        raise FileNotFoundError(
            f"No cached {model_type} model found for {args.symbol} @ {args.timeframe}. "
            "Train it first via the standard CLI."
        )
    provider = DataProvider(config.twelve_data)
    price_frame = provider.load_data(args.symbol, args.timeframe)
    freq = FREQ_MAP.get(args.timeframe.lower())
    if freq is None:
        raise ValueError(f"No pandas frequency mapping for timeframe '{args.timeframe}'.")
    steps = max(1, int(args.eval_steps))
    forecast_df = model.forecast(steps=steps, freq=freq, future_regressors=None)
    if "ds" not in forecast_df.columns:
        forecast_df = forecast_df.reset_index().rename(columns={"index": "ds"})
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], utc=True)
    predictions = [
        {"ds": ts.isoformat(), "yhat": float(val)}
        for ts, val in zip(forecast_df["ds"], forecast_df["yhat"])
    ]
    recent_actual = price_frame["close"].tail(max(steps, 1))
    recent_actual_payload = [
        {"ds": idx.isoformat(), "close": float(value)}
        for idx, value in recent_actual.items()
    ]
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "model_type": model_type,
        "steps": steps,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "predictions": predictions,
        "recent_actual": recent_actual_payload,
        "model_path": str(router.get_model_path(model_type, args.symbol, args.timeframe)),
    }
    if args.eval_output:
        Path(args.eval_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Evaluation forecast written to %s", args.eval_output)
    else:
        print(json.dumps(payload, indent=2))


def run_backtest_mode(config: AppConfig, args: argparse.Namespace) -> None:
    """Execute a walk-forward backtest and emit aggregated metrics."""
    cache_dir = Path(args.data_cache_dir).expanduser() if args.data_cache_dir else None
    runner = BacktestRunner(config, cache_dir=cache_dir)
    use_montecarlo = not args.no_montecarlo and config.monte_carlo.use_montecarlo
    samples = args.backtest_samples if args.backtest_samples and args.backtest_samples > 0 else None
    result = runner.run(
        symbol=args.symbol,
        timeframe=args.timeframe,
        horizons=args.horizons,
        paths=args.paths,
        use_montecarlo=use_montecarlo,
        samples=samples,
        stride=args.backtest_stride,
        min_history=max(1, args.backtest_min_history),
    )
    payload = json.dumps(result.to_dict(), indent=2)
    if args.backtest_output:
        Path(args.backtest_output).write_text(payload, encoding="utf-8")
        logger.info("Backtest report written to %s", args.backtest_output)
    else:
        print(payload)


def _load_forecast_payload(path: Path):
    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"]
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            text = path.read_text(encoding=encoding)
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc
            continue
    raise ValueError(f"Unable to read forecast payload from {path}: {last_error}") from last_error


def _extract_forecast_series(payload: dict) -> tuple[pd.Series, dict[str, pd.Series]]:
    trajectories = payload.get("trajectories")
    if trajectories:
        return _build_from_trajectories(trajectories)
    entries = payload.get("forecasts")
    if entries:
        return _build_from_forecast_entries(entries)
    raise ValueError(
        "Forecast payload must contain either 'trajectories' (preferred) or 'forecasts' entries. "
        "Rerun the forecast with --trajectory-output or ensure the JSON includes ds/p20/p80 fields."
    )


def _build_from_trajectories(entries: list) -> tuple[pd.Series, dict[str, pd.Series]]:
    timestamps = []
    medians = []
    for entry in entries:
        if not entry.get("timestamps") or not entry.get("predictions"):
            continue
        ts_list = pd.to_datetime(entry["timestamps"], utc=True)
        preds = np.asarray(entry["predictions"], dtype=float)
        timestamps.extend(ts_list)
        medians.extend(preds)
    if not timestamps:
        raise ValueError("No trajectory datapoints found in payload.")
    predicted = pd.Series(medians, index=pd.DatetimeIndex(timestamps)).sort_index()
    # Trajectories do not include quantiles; signal absence.
    return predicted, {}


def _load_actual_series(config: AppConfig, args: argparse.Namespace) -> pd.Series:
    """Return realised price series for reporting."""
    if args.report_actual_input:
        csv_path = Path(args.report_actual_input)
        if not csv_path.exists():
            raise FileNotFoundError(f"Actual CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
        if "close" not in df.columns:
            raise ValueError("Actual CSV must contain a 'close' column.")
        return df["close"].astype(float)
    provider = DataProvider(config.twelve_data)
    frame = provider.load_data(args.symbol, args.timeframe)
    return frame["close"].astype(float)


def _build_from_forecast_entries(entries: list) -> tuple[pd.Series, dict[str, pd.Series]]:
    timestamps = []
    medians = []
    p20_values = []
    p80_values = []
    for entry in entries:
        try:
            ts = pd.to_datetime(entry["ds"], utc=True)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid forecast entry: {entry}") from exc
        timestamps.append(ts)
        medians.append(float(entry.get("median") or entry.get("yhat") or entry.get("medianPrice")))
        p20_values.append(float(entry.get("p20")))
        p80_values.append(float(entry.get("p80")))
    if not timestamps:
        raise ValueError("No forecast datapoints found in payload.")
    predicted = pd.Series(medians, index=pd.DatetimeIndex(timestamps)).sort_index()
    quantiles = {
        "p20": pd.Series(p20_values, index=pd.DatetimeIndex(timestamps)).sort_index(),
        "p80": pd.Series(p80_values, index=pd.DatetimeIndex(timestamps)).sort_index(),
    }
    return predicted, quantiles


if __name__ == "__main__":
    main()
