"""Persistence utilities for HMM regime models (local + optional S3)."""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from alphalens_forecast.regime_detection.deterministic import HMMRegimeModel, RegimeConfig
from alphalens_forecast.storage.s3_store import S3Store, S3UnavailableError
from alphalens_forecast.utils.text import slugify

logger = logging.getLogger(__name__)


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _default_base_dir() -> Path:
    package_root = Path(__file__).resolve().parents[2]
    repo_models = (package_root / "models").resolve()
    cwd_models = (Path.cwd() / "models").resolve()
    if repo_models.exists():
        return repo_models
    return cwd_models


def _filter_config_state(state: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(RegimeConfig.__dataclass_fields__.keys())
    return {key: value for key, value in state.items() if key in allowed}


class HMMRegimeStore:
    """Persist and load HMMRegimeModel artifacts using the models directory layout."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is not None:
            resolved = Path(base_dir).expanduser().resolve()
        else:
            env_dir = os.environ.get("ALPHALENS_MODEL_DIR")
            resolved = Path(env_dir).expanduser().resolve() if env_dir else _default_base_dir()
        self._base_dir = resolved / "regime_hmm"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._s3_only = _env_bool("ALPHALENS_S3_ONLY", False) or _env_bool("ALPHALENS_REQUIRE_S3", False)
        self._s3_store = S3Store.from_env(logger)
        if self._s3_only and self._s3_store is None:
            raise RuntimeError("S3-only mode enabled but ALPHALENS_MODEL_BUCKET is not set.")

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def _s3_symbol(self, symbol: str) -> str:
        cleaned = "".join(ch for ch in symbol if ch.isalnum())
        return cleaned or slugify(symbol)

    def _s3_prefix(self, symbol: str, timeframe: str) -> str:
        return f"{self._s3_symbol(symbol)}/{slugify(timeframe)}/regime_hmm"

    def _s3_model_key(self, symbol: str, timeframe: str) -> str:
        return f"{self._s3_prefix(symbol, timeframe)}/model.pkl"

    def _s3_metadata_key(self, symbol: str, timeframe: str) -> str:
        return f"{self._s3_prefix(symbol, timeframe)}/metadata.json"

    def get_model_dir(self, symbol: str, timeframe: str) -> Path:
        return self._base_dir / slugify(symbol) / slugify(timeframe)

    def get_model_path(self, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(symbol, timeframe) / "model.pkl"

    def get_metadata_path(self, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(symbol, timeframe) / "metadata.json"

    def save(
        self,
        symbol: str,
        timeframe: str,
        model: HMMRegimeModel,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        model_dir = self.get_model_dir(symbol, timeframe)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.get_model_path(symbol, timeframe)

        payload = {
            "model": model.model,
            "mean": model.mean,
            "std": model.std,
            "regime_map": model.regime_map,
            "obs_cols": list(model.obs_cols),
            "config": asdict(model.config),
            "lookback": model.lookback,
        }
        with open(model_path, "wb") as handle:
            pickle.dump(payload, handle)

        manifest = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": "regime_hmm",
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metadata": metadata or {},
            "model_file": model_path.name,
        }
        with open(self.get_metadata_path(symbol, timeframe), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        if self._s3_store is not None:
            try:
                self._s3_store.upload_file(model_path, self._s3_model_key(symbol, timeframe))
                self._s3_store.upload_file(
                    self.get_metadata_path(symbol, timeframe),
                    self._s3_metadata_key(symbol, timeframe),
                )
            except S3UnavailableError as exc:
                if self._s3_only:
                    raise
                logger.warning("S3 unavailable; skipping HMM upload. (%s)", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("S3 upload failed for HMM regime model: %s", exc)
                if self._s3_only:
                    raise

        logger.info("Saved HMM regime model for %s @ %s to %s", symbol, timeframe, model_path)
        return model_path

    def load(self, symbol: str, timeframe: str) -> Optional[HMMRegimeModel]:
        model_dir = self.get_model_dir(symbol, timeframe)
        model_path = self.get_model_path(symbol, timeframe)
        metadata_path = self.get_metadata_path(symbol, timeframe)

        if self._s3_store is not None:
            try:
                model_exists = self._s3_store.exists(self._s3_model_key(symbol, timeframe))
                metadata_exists = self._s3_store.exists(self._s3_metadata_key(symbol, timeframe))
            except S3UnavailableError as exc:
                if self._s3_only:
                    raise
                logger.warning("S3 unavailable; using local HMM cache only. (%s)", exc)
                model_exists = False
                metadata_exists = False
            if model_exists:
                try:
                    self._s3_store.download_file(self._s3_model_key(symbol, timeframe), model_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to download HMM model from S3: %s", exc)
                    if self._s3_only:
                        raise
            if metadata_exists:
                try:
                    self._s3_store.download_file(self._s3_metadata_key(symbol, timeframe), metadata_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to download HMM metadata from S3: %s", exc)
                    if self._s3_only:
                        raise
            if self._s3_only and not model_exists:
                raise FileNotFoundError(
                    f"No trained HMM regime model available for {symbol} @ {timeframe}."
                )

        if not model_path.exists():
            if self._s3_only:
                raise FileNotFoundError(
                    f"No trained HMM regime model available for {symbol} @ {timeframe}."
                )
            return None

        with open(model_path, "rb") as handle:
            payload = pickle.load(handle)

        config_state = payload.get("config", {})
        if isinstance(config_state, dict):
            config_state = _filter_config_state(config_state)
            config = RegimeConfig(**config_state)
        else:
            config = RegimeConfig()

        regime_map = payload.get("regime_map", {})
        if isinstance(regime_map, dict):
            regime_map = {int(k): str(v) for k, v in regime_map.items()}

        model = HMMRegimeModel(
            model=payload.get("model"),
            mean=payload.get("mean"),
            std=payload.get("std"),
            regime_map=regime_map,
            obs_cols=tuple(payload.get("obs_cols", ())),
            config=config,
            lookback=int(payload.get("lookback", config.lookback)),
        )
        logger.info("Loaded HMM regime model for %s @ %s from %s", symbol, timeframe, model_path)
        return model


def save_hmm_regime_model(
    symbol: str,
    timeframe: str,
    model: HMMRegimeModel,
    *,
    store: Optional[HMMRegimeStore] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    store = store or HMMRegimeStore()
    return store.save(symbol, timeframe, model, metadata=metadata)


def load_hmm_regime_model(
    symbol: str,
    timeframe: str,
    *,
    store: Optional[HMMRegimeStore] = None,
) -> Optional[HMMRegimeModel]:
    store = store or HMMRegimeStore()
    return store.load(symbol, timeframe)


__all__ = ["HMMRegimeStore", "save_hmm_regime_model", "load_hmm_regime_model"]
