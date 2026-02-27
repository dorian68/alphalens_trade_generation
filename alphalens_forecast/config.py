"""Application configuration for the AlphaLens forecasting system."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml

from dotenv import load_dotenv

_PACKAGE_ROOT = Path(__file__).resolve().parent
load_dotenv()
load_dotenv(_PACKAGE_ROOT / ".env")


def _env_bool(key: str, default: bool = True) -> bool:
    """Safely parse a boolean flag from the environment."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _env_float(key: str, default: float) -> float:
    """Safely parse a float from the environment."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """Safely parse an int from the environment."""
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class TwelveDataConfig:
    """Configuration for Twelve Data API access."""

    api_key: str = os.getenv("TWELVE_DATA_API_KEY", "")
    base_url: str = os.getenv(
        "TWELVE_DATA_BASE_URL",
        "https://api.twelvedata.com/time_series",
    )
    symbol: str = os.getenv("DEFAULT_SYMBOL", "BTC/USD")
    interval: str = os.getenv("DEFAULT_INTERVAL", "15min")
    output_size: int = _env_int("DATA_OUTPUT_SIZE", 5000)
    retry_attempts: int = _env_int("TD_RETRY_ATTEMPTS", 3)
    retry_backoff: float = _env_float("TD_RETRY_BACKOFF", 1.5)


@dataclass
class MonteCarloConfig:
    """Configuration values for the Monte Carlo simulator."""

    use_montecarlo: bool = _env_bool("USE_MONTECARLO", True)
    paths: int = _env_int("MC_PATHS", 3000)
    horizon_hours: List[int] = field(
        default_factory=lambda: [3, 6, 12, 24]
    )
    seed: Optional[int] = (
        _env_int("MC_SEED", 0) if os.getenv("MC_SEED") is not None else None
    )


@dataclass
class RiskConfig:
    """Risk management parameters."""

    target_volatility: float = _env_float("TARGET_ANNUAL_VOL", 0.20)
    confidence_threshold: float = _env_float("CONFIDENCE_THRESHOLD", 0.60)
    max_position: float = _env_float("MAX_POSITION_SIZE", 1.0)


@dataclass
class TrainingConfig:
    """Runtime settings for Torch-backed training data loading."""

    num_workers: int = _env_int("TRAIN_NUM_WORKERS", 0)
    pin_memory: bool = _env_bool("TRAIN_PIN_MEMORY", False)
    persistent_workers: bool = _env_bool("TRAIN_PERSISTENT_WORKERS", False)


@dataclass
class AppConfig:
    """Top-level configuration container."""

    twelve_data: TwelveDataConfig = field(default_factory=TwelveDataConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    s3_only: bool = (
        _env_bool("ALPHALENS_S3_ONLY", False)
        or _env_bool("ALPHALENS_REQUIRE_S3", False)
    )
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "15min")
    torch_device: str = os.getenv("ALPHALENS_DEVICE") or os.getenv("TORCH_DEVICE", "cpu")
    inference_device: str = os.getenv("INFERENCE_DEVICE", "auto")
    regime_switching: bool = _env_bool("ALPHALENS_REGIME_SWITCHING", False)
    regime_lookback: int = _env_int("ALPHALENS_REGIME_LOOKBACK", 200)
    performance_patches: bool = _env_bool("ALPHALENS_PERFORMANCE_PATCHES", True)
    regime_mode: str = os.getenv("ALPHALENS_REGIME_MODE", "hmm")
    regime_score_smoothing: bool = _env_bool("ALPHALENS_REGIME_SCORE_SMOOTHING", False)
    regime_score_smoothing_alpha: float = _env_float("ALPHALENS_REGIME_SCORE_SMOOTHING_ALPHA", 0.15)
    regime_walk_forward_calib: bool = _env_bool("ALPHALENS_REGIME_WALK_FORWARD_CALIB", False)
    regime_calib_window: int = _env_int("ALPHALENS_REGIME_CALIB_WINDOW", 200)
    regime_calib_min_history: int = _env_int("ALPHALENS_REGIME_CALIB_MIN_HISTORY", 50)
    regime_calib_trend_quantile: float = _env_float("ALPHALENS_REGIME_CALIB_TREND_Q", 0.7)
    regime_calib_range_quantile: float = _env_float("ALPHALENS_REGIME_CALIB_RANGE_Q", 0.3)
    regime_calib_chop_quantile: float = _env_float("ALPHALENS_REGIME_CALIB_CHOP_Q", 0.7)
    regime_calib_vol_expansion_quantile: float = _env_float("ALPHALENS_REGIME_CALIB_VOL_EXP_Q", 0.7)
    regime_vol_mom_confirm: bool = _env_bool("ALPHALENS_REGIME_VOL_MOM_CONFIRM", False)
    regime_volume_window: int = _env_int("ALPHALENS_REGIME_VOLUME_WINDOW", 50)
    regime_volume_z_threshold: float = _env_float("ALPHALENS_REGIME_VOLUME_Z_THRESHOLD", 1.0)
    regime_volume_ratio_threshold: float = _env_float("ALPHALENS_REGIME_VOLUME_RATIO_THRESHOLD", 1.1)
    regime_momentum_window: int = _env_int("ALPHALENS_REGIME_MOMENTUM_WINDOW", 14)
    regime_momentum_threshold: float = _env_float("ALPHALENS_REGIME_MOMENTUM_THRESHOLD", 0.3)
    regime_range_model: str = os.getenv("ALPHALENS_RANGE_MODEL", "mean_reversion")
    regime_stress_model: str = os.getenv("ALPHALENS_STRESS_MODEL", "flat")
    regime_breakout_model: str = os.getenv("ALPHALENS_BREAKOUT_MODEL", "momentum")
    regime_baseline_cache: bool = _env_bool("ALPHALENS_REGIME_BASELINE_CACHE", False)
    regime_per_instrument_models: bool = _env_bool("ALPHALENS_REGIME_PER_INSTRUMENT_MODELS", False)
    bias_correction_enabled: bool = _env_bool("ALPHALENS_BIAS_CORRECTION", True)
    bias_correction_window: int = _env_int("ALPHALENS_BIAS_CORRECTION_WINDOW", 48)
    bias_correction_min_history: int = _env_int("ALPHALENS_BIAS_CORRECTION_MIN_HISTORY", 120)
    bias_correction_sigma_clip: float = _env_float("ALPHALENS_BIAS_CORRECTION_SIGMA_CLIP", 2.0)
    regime_blend_enabled: bool = _env_bool("ALPHALENS_REGIME_BLEND", True)
    regime_blend_range: float = _env_float("ALPHALENS_REGIME_BLEND_RANGE", 0.8)
    regime_blend_trend: float = _env_float("ALPHALENS_REGIME_BLEND_TREND", 0.2)
    regime_blend_breakout: float = _env_float("ALPHALENS_REGIME_BLEND_BREAKOUT", 0.2)
    regime_blend_confidence_alpha: float = _env_float("ALPHALENS_REGIME_BLEND_CONF_ALPHA", 0.5)
    regime_blend_horizon_alpha: float = _env_float("ALPHALENS_REGIME_BLEND_HORIZON_ALPHA", 0.3)


@dataclass
class InstrumentDefinition:
    """Describe an instrument and its supported horizons/timeframes."""

    symbol: str
    timeframes: List[str]
    horizons: List[int]


@dataclass
class InstrumentUniverse:
    """Container with all configured instruments."""

    instruments: List[InstrumentDefinition]
    default_horizons: List[int]


def _instrument_config_path() -> Path:
    root = _PACKAGE_ROOT.parent
    return root / "config" / "instruments.yml"


@lru_cache(maxsize=1)
def get_instrument_universe() -> InstrumentUniverse:
    """Load instrument/timeframe configuration from YAML."""
    path = _instrument_config_path()
    instruments: List[InstrumentDefinition] = []
    default_horizons = [3, 6, 12, 24]
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        default_horizons = payload.get("defaults", {}).get("horizons", default_horizons)
        for entry in payload.get("instruments", []):
            instruments.append(
                InstrumentDefinition(
                    symbol=entry["symbol"],
                    timeframes=entry.get("timeframes", ["15min"]),
                    horizons=entry.get("horizons", default_horizons),
                )
            )
    else:
        fallback_symbols = [
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
            "AUD/USD",
            "USD/CAD",
            "USD/CHF",
            "BTC/USD",
            "ETH/USD",
        ]
        instruments = [
            InstrumentDefinition(symbol=symbol, timeframes=["15min", "1h", "4h"], horizons=default_horizons)
            for symbol in fallback_symbols
        ]
    return InstrumentUniverse(instruments=instruments, default_horizons=default_horizons)


def get_config() -> AppConfig:
    """Return a fresh copy of the application configuration."""
    return AppConfig()


__all__ = [
    "AppConfig",
    "InstrumentDefinition",
    "InstrumentUniverse",
    "TwelveDataConfig",
    "MonteCarloConfig",
    "RiskConfig",
    "TrainingConfig",
    "get_instrument_universe",
    "get_config",
]
