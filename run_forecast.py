"""Ready-to-run AlphaLens forecast script (no CLI)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphalens_forecast.config import AppConfig, get_config
from alphalens_forecast.data import DataProvider
from alphalens_forecast.forecasting import ForecastEngine, OrchestrationResult
from alphalens_forecast.models import ModelRouter
from alphalens_forecast.utils.model_store import ModelStore, StoredArtifacts
from alphalens_forecast.utils.text import slugify

logger = logging.getLogger("alphalens_runner")


@dataclass(frozen=True)
class RunOverrides:
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    horizons: Optional[List[int]] = None
    paths: Optional[int] = None
    use_montecarlo: Optional[bool] = None
    trade_mode: str = "spot"
    reuse_cached: bool = False
    save_artifacts: bool = False
    model_dir: Optional[str] = None
    data_cache_dir: Optional[str] = None
    output_path: Optional[str] = "forecast_payload.json"
    show_progress: bool = False
    log_level: str = "INFO"


# Edit these overrides to customize a run.
RUN_OVERRIDES = RunOverrides(
    # symbol="EUR/USD",
    # timeframe="15min",
    # horizons=[3, 6, 12, 24],
    # paths=3000,
    # use_montecarlo=True,
    # output_path="forecast_payload.json",
)


@dataclass(frozen=True)
class RunSettings:
    symbol: str
    timeframe: str
    horizons: List[int]
    paths: int
    use_montecarlo: bool
    trade_mode: str
    reuse_cached: bool
    save_artifacts: bool
    model_dir: Optional[Path]
    data_cache_dir: Optional[Path]
    output_path: Optional[Path]
    show_progress: bool
    log_level: str


def _coerce_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return Path(value).expanduser()


def _normalize_horizons(values: List[int]) -> List[int]:
    return [int(value) for value in values]


def _validate_settings(settings: RunSettings) -> None:
    if not settings.symbol:
        raise ValueError("symbol must not be empty.")
    if not settings.timeframe:
        raise ValueError("timeframe must not be empty.")
    if not settings.horizons:
        raise ValueError("horizons must not be empty.")
    if any(value <= 0 for value in settings.horizons):
        raise ValueError("horizons must be positive integers.")
    if settings.paths <= 0:
        raise ValueError("paths must be positive.")
    if settings.trade_mode not in {"spot", "forward"}:
        raise ValueError("trade_mode must be 'spot' or 'forward'.")


def resolve_settings(config: AppConfig, overrides: RunOverrides) -> RunSettings:
    horizons = overrides.horizons or list(config.monte_carlo.horizon_hours)
    trade_mode = (overrides.trade_mode or "spot").strip().lower()
    settings = RunSettings(
        symbol=overrides.symbol or config.twelve_data.symbol,
        timeframe=overrides.timeframe or config.twelve_data.interval,
        horizons=_normalize_horizons(horizons),
        paths=overrides.paths if overrides.paths is not None else config.monte_carlo.paths,
        use_montecarlo=(
            overrides.use_montecarlo
            if overrides.use_montecarlo is not None
            else config.monte_carlo.use_montecarlo
        ),
        trade_mode=trade_mode,
        reuse_cached=bool(overrides.reuse_cached),
        save_artifacts=bool(overrides.save_artifacts),
        model_dir=_coerce_path(overrides.model_dir),
        data_cache_dir=_coerce_path(overrides.data_cache_dir),
        output_path=_coerce_path(overrides.output_path),
        show_progress=bool(overrides.show_progress),
        log_level=overrides.log_level or "INFO",
    )
    _validate_settings(settings)
    return settings


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(level)


def resolve_model_directory(candidate: Optional[Path]) -> Path:
    if candidate is not None:
        base = candidate
    else:
        base = Path(os.environ.get("ALPHALENS_MODEL_DIR", "models"))
    if base.is_absolute():
        resolved = base
    else:
        resolved = Path("/tmp") / base if "AWS_EXECUTION_ENV" in os.environ else Path.cwd() / base
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def persist_artifacts(
    model_store: ModelStore,
    model_dir: Path,
    symbol: str,
    timeframe: str,
    result: OrchestrationResult,
) -> Optional[StoredArtifacts]:
    if result.used_cached_artifacts:
        logger.info("Cached artifacts reused; skipping save.")
        return None
    if result.mean_model is None or result.vol_model is None:
        logger.warning("Artifacts not saved because models are unavailable.")
        return None

    timestamp_slug = result.run_timestamp_slug or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"{slugify(symbol)}_{slugify(timeframe)}_{timestamp_slug}"

    metadata = dict(result.metadata)
    metadata["model_dir"] = str(model_dir)
    metadata["payload_hash"] = hashlib.sha256(
        json.dumps(result.payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    stored = model_store.save(
        prefix=prefix,
        mean_model=result.mean_model,
        vol_model=result.vol_model,
        metadata=metadata,
        payload=result.payload,
    )
    logger.info("Artifacts saved to %s and %s", stored.model_path, stored.manifest_path)
    return stored


def write_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Payload written to %s", path)


def run_forecast(settings: RunSettings, config: AppConfig) -> OrchestrationResult:
    model_dir = resolve_model_directory(settings.model_dir)
    data_provider = DataProvider(
        config.twelve_data,
        cache_dir=settings.data_cache_dir,
        auto_refresh=True,
    )
    model_router = ModelRouter(model_dir)
    model_store = None
    if settings.save_artifacts or settings.reuse_cached:
        model_store = ModelStore(model_dir, logger)

    engine = ForecastEngine(config, data_provider, model_router)
    result = engine.forecast(
        symbol=settings.symbol,
        timeframe=settings.timeframe,
        horizons=settings.horizons,
        paths=settings.paths,
        use_montecarlo=settings.use_montecarlo,
        trade_mode=settings.trade_mode,
        reuse_cached=settings.reuse_cached,
        model_store=model_store,
        show_progress=settings.show_progress,
    )

    if settings.save_artifacts and model_store is not None:
        persist_artifacts(
            model_store=model_store,
            model_dir=model_dir,
            symbol=settings.symbol,
            timeframe=settings.timeframe,
            result=result,
        )

    if settings.output_path is not None:
        write_payload(settings.output_path, result.payload)

    return result


def main() -> OrchestrationResult:
    config = get_config()
    settings = resolve_settings(config, RUN_OVERRIDES)
    configure_logging(settings.log_level)
    logger.info(
        "Running forecast for %s [%s] horizons=%s",
        settings.symbol,
        settings.timeframe,
        ",".join(str(value) for value in settings.horizons),
    )
    result = run_forecast(settings, config)
    logger.info("Forecast complete.")
    return result


if __name__ == "__main__":
    main()
