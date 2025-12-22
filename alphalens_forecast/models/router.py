"""Model routing utilities."""
from __future__ import annotations

import importlib
import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from alphalens_forecast.storage.s3_store import S3Store, S3UnavailableError
from alphalens_forecast.utils.text import slugify

logger = logging.getLogger(__name__)


def _emit_message(message: str) -> None:
    """Emit console output without breaking tqdm progress bars when available."""
    try:
        from tqdm import tqdm  # type: ignore

        tqdm.write(message)
    except Exception:
        print(message)


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_class(class_path: str) -> type:
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid class path '{class_path}'")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def save_model_safely(
    model: Any,
    model_path: Path,
    model_type: str,
    symbol: str,
    timeframe: str,
    logger: logging.Logger,
) -> Tuple[Path, str]:
    """
    Save a model safely without using pickle, preventing memory crashes on EC2.
    Tries NeuralProphet native .save() first, then falls back to torch.save(state_dict).
    """
    target = Path(model_path)
    save_method = getattr(model, "save", None)
    if callable(save_method):
        try:
            actual = save_method(target)
            saved_path = Path(actual) if actual is not None else target
            return saved_path, "native"
        except NotImplementedError:
            # Some forecasters inherit BaseForecaster.save without implementing it.
            # Fall back to state_dict without changing training behavior.
            pass
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to save model for %s model for %s @ %s: %s",
                model_type,
                symbol,
                timeframe,
                exc,
            )
            raise

    state_dict_method = getattr(model, "state_dict", None)
    if callable(state_dict_method):
        state_path = Path(f"{target}.pth")
        try:
            torch.save(state_dict_method(), state_path)
            return state_path, "state_dict"
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to save state_dict for %s model for %s @ %s: %s",
                model_type,
                symbol,
                timeframe,
                exc,
            )
            raise

    raise AttributeError(
        f"Model {model_type} for {symbol} @ {timeframe} does not expose save() or state_dict()."
    )


def _default_base_dir() -> Path:
    """Return the best-effort default models directory."""
    package_root = Path(__file__).resolve().parents[2]
    repo_models = (package_root / "models").resolve()
    cwd_models = (Path.cwd() / "models").resolve()
    if repo_models.exists():
        return repo_models
    return cwd_models


class _S3ModelBackend:
    """S3-backed model store with deterministic local cache handling."""

    def __init__(self, s3_store: Optional[S3Store], s3_only: bool, logger: logging.Logger) -> None:
        self._s3_store = s3_store
        self._s3_only = s3_only
        self._logger = logger

    def ensure_cached(
        self,
        *,
        model_type: str,
        symbol: str,
        timeframe: str,
        model_dir: Path,
        prefix: str,
        model_key: str,
        metadata_key: str,
        metrics_key: str,
        metadata_path: Path,
    ) -> bool:
        if self._s3_store is None:
            if self._s3_only:
                raise RuntimeError(
                    "S3-only mode enabled but no S3 model store is configured."
                )
            return False
        try:
            model_exists = self._s3_store.exists(model_key)
            metadata_exists = self._s3_store.exists(metadata_key)
        except S3UnavailableError as exc:
            if self._s3_only:
                raise
            self._logger.warning("S3 unavailable; using local models only. (%s)", exc)
            return False
        if not model_exists:
            self._logger.warning(
                "Model artifact missing in S3 for %s @ %s (%s).",
                symbol,
                timeframe,
                model_type,
            )
            if self._s3_only:
                raise FileNotFoundError(
                    f"No trained model available for {symbol} @ {timeframe} ({model_type})."
                )
            return False
        model_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {}
        metadata_changed = False
        if metadata_exists:
            temp_path = metadata_path.with_suffix(".s3.tmp")
            self._s3_store.download_file(metadata_key, temp_path)
            remote_text = ""
            try:
                remote_text = temp_path.read_text(encoding="utf-8")
            except OSError as exc:
                metadata_changed = True
                self._logger.warning(
                    "Failed to read S3 metadata for %s @ %s: %s",
                    symbol,
                    timeframe,
                    exc,
                )
            local_text = None
            if metadata_path.exists():
                try:
                    local_text = metadata_path.read_text(encoding="utf-8")
                except OSError:
                    local_text = None
            if local_text is not None and local_text == remote_text and remote_text:
                self._logger.info(
                    "Reusing cached metadata for %s @ %s (%s).",
                    symbol,
                    timeframe,
                    model_type,
                )
            else:
                metadata_changed = True
                if remote_text:
                    try:
                        metadata_path.write_text(remote_text, encoding="utf-8")
                    except OSError as exc:
                        self._logger.warning(
                            "Failed to write metadata cache for %s @ %s: %s",
                            symbol,
                            timeframe,
                            exc,
                        )
                    else:
                        self._logger.info(
                            "Fetched metadata from S3 for %s @ %s (%s).",
                            symbol,
                            timeframe,
                            model_type,
                        )
            try:
                temp_path.unlink()
            except OSError:
                pass
            if remote_text:
                try:
                    manifest = json.loads(remote_text) or {}
                except json.JSONDecodeError as exc:
                    metadata_changed = True
                    self._logger.warning(
                        "Failed to parse S3 metadata for %s @ %s: %s",
                        symbol,
                        timeframe,
                        exc,
                    )
                    manifest = {}
        else:
            metadata_changed = True
            self._logger.warning(
                "Metadata missing in S3 for %s @ %s (%s).",
                symbol,
                timeframe,
                model_type,
            )

        model_file = manifest.get("model_file") or "model.pkl"
        local_model_path = model_dir / model_file
        if metadata_changed or not local_model_path.exists():
            self._s3_store.download_file(model_key, local_model_path)
            self._logger.info(
                "Fetched model artifact from S3 for %s @ %s (%s).",
                symbol,
                timeframe,
                model_type,
            )
        else:
            self._logger.info(
                "Reusing cached model artifact for %s @ %s (%s).",
                symbol,
                timeframe,
                model_type,
            )

        metrics_exists = False
        try:
            metrics_exists = self._s3_store.exists(metrics_key)
        except S3UnavailableError as exc:
            if self._s3_only:
                raise
            self._logger.warning("S3 metrics check failed; skipping. (%s)", exc)
        if metrics_exists:
            metrics_path = model_dir / "metrics.json"
            if metadata_changed or not metrics_path.exists():
                self._s3_store.download_file(metrics_key, metrics_path)
                self._logger.info(
                    "Fetched metrics from S3 for %s @ %s (%s).",
                    symbol,
                    timeframe,
                    model_type,
                )
            else:
                self._logger.info(
                    "Reusing cached metrics for %s @ %s (%s).",
                    symbol,
                    timeframe,
                    model_type,
                )

        try:
            remote_keys = self._s3_store.list(prefix)
        except S3UnavailableError as exc:
            if self._s3_only:
                raise
            self._logger.warning("S3 list failed; skipping auxiliary sync. (%s)", exc)
            remote_keys = []

        extra_keys = [
            key for key in remote_keys
            if key not in {"model.pkl", "metadata.json", "metrics.json"}
        ]
        downloaded = 0
        for rel_key in extra_keys:
            local_path = model_dir / rel_key
            if metadata_changed or not local_path.exists():
                remote_path = f"{prefix}/{rel_key}"
                self._s3_store.download_file(remote_path, local_path)
                downloaded += 1
        if extra_keys:
            if downloaded:
                self._logger.info(
                    "Fetched %d auxiliary artifact(s) from S3 for %s @ %s (%s).",
                    downloaded,
                    symbol,
                    timeframe,
                    model_type,
                )
            else:
                self._logger.info(
                    "Reusing cached auxiliary artifacts for %s @ %s (%s).",
                    symbol,
                    timeframe,
                    model_type,
                )

        if self._s3_only:
            allowed = {model_file}
            if metadata_exists:
                allowed.add("metadata.json")
            if metrics_exists:
                allowed.add("metrics.json")
            allowed.update(extra_keys)
            removed = 0
            for path in model_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(model_dir).as_posix()
                if rel_path in allowed:
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError as exc:
                    self._logger.warning(
                        "Failed to remove stale cache file %s: %s",
                        path,
                        exc,
                    )
            if removed:
                self._logger.info(
                    "Removed %d stale cache file(s) for %s @ %s (%s).",
                    removed,
                    symbol,
                    timeframe,
                    model_type,
                )

        return True

    def upload(
        self,
        *,
        model_dir: Path,
        saved_path: Path,
        model_type: str,
        symbol: str,
        timeframe: str,
        prefix: str,
        model_key: str,
        metadata_key: str,
        metrics_key: str,
        metadata_path: Path,
    ) -> None:
        if self._s3_store is None:
            if self._s3_only:
                raise RuntimeError(
                    "S3-only mode enabled but no S3 model store is configured."
                )
            return
        try:
            self._s3_store.upload_file(saved_path, model_key)
            if metadata_path.exists():
                self._s3_store.upload_file(metadata_path, metadata_key)
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                self._s3_store.upload_file(metrics_path, metrics_key)
            for path in model_dir.rglob("*"):
                if not path.is_file():
                    continue
                if path == saved_path:
                    continue
                rel_path = path.relative_to(model_dir).as_posix()
                if rel_path in {"metadata.json", "metrics.json"}:
                    continue
                key = f"{prefix}/{rel_path}"
                self._s3_store.upload_file(path, key)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "S3 upload failed for %s @ %s (%s): %s",
                symbol,
                timeframe,
                model_type,
                exc,
            )
            if self._s3_only:
                raise


class ModelRouter:
    """
    Persist and load models using the convention ``models/{type}/{symbol}/{tf}``.

    This provides a single choke point so both training jobs and CLI inference
    store assets identically. Metadata is stored alongside the pickle to help
    orchestration layers reason about freshness.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is not None:
            resolved = Path(base_dir).expanduser().resolve()
        else:
            env_dir = os.environ.get("ALPHALENS_MODEL_DIR")
            resolved = (
                Path(env_dir).expanduser().resolve()
                if env_dir
                else _default_base_dir()
            )
        self._base_dir = resolved
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._s3_only = _env_bool("ALPHALENS_S3_ONLY", False) or _env_bool(
            "ALPHALENS_REQUIRE_S3",
            False,
        )
        self._s3_store = S3Store.from_env(logger)
        if self._s3_only and self._s3_store is None:
            raise RuntimeError(
                "S3-only mode enabled but ALPHALENS_MODEL_BUCKET is not set."
            )
        self._s3_backend = _S3ModelBackend(self._s3_store, self._s3_only, logger)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def s3_only(self) -> bool:
        return self._s3_only

    def get_model_dir(self, model_type: str, symbol: str, timeframe: str) -> Path:
        """Return the directory that hosts the pickle/manifest for the model."""
        return (
            self._base_dir
            / model_type.lower()
            / slugify(symbol)
            / slugify(timeframe)
        )

    def get_model_path(self, model_type: str, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(model_type, symbol, timeframe) / "model"

    def _legacy_model_path(self, model_type: str, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(model_type, symbol, timeframe) / "model.pkl"

    def get_metadata_path(self, model_type: str, symbol: str, timeframe: str) -> Path:
        return self.get_model_dir(model_type, symbol, timeframe) / "metadata.json"

    def _s3_symbol(self, symbol: str) -> str:
        cleaned = "".join(ch for ch in symbol if ch.isalnum())
        return cleaned or slugify(symbol)

    def _s3_prefix(self, model_type: str, symbol: str, timeframe: str) -> str:
        return f"{self._s3_symbol(symbol)}/{slugify(timeframe)}/{model_type.lower()}"

    def _s3_model_key(self, model_type: str, symbol: str, timeframe: str) -> str:
        return f"{self._s3_prefix(model_type, symbol, timeframe)}/model.pkl"

    def _s3_metadata_key(self, model_type: str, symbol: str, timeframe: str) -> str:
        return f"{self._s3_prefix(model_type, symbol, timeframe)}/metadata.json"

    def _s3_metrics_key(self, model_type: str, symbol: str, timeframe: str) -> str:
        return f"{self._s3_prefix(model_type, symbol, timeframe)}/metrics.json"

    def _upload_to_s3(
        self,
        model_dir: Path,
        saved_path: Path,
        model_type: str,
        symbol: str,
        timeframe: str,
    ) -> None:
        self._s3_backend.upload(
            model_dir=model_dir,
            saved_path=saved_path,
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            prefix=self._s3_prefix(model_type, symbol, timeframe),
            model_key=self._s3_model_key(model_type, symbol, timeframe),
            metadata_key=self._s3_metadata_key(model_type, symbol, timeframe),
            metrics_key=self._s3_metrics_key(model_type, symbol, timeframe),
            metadata_path=self.get_metadata_path(model_type, symbol, timeframe),
        )

    def _sync_from_s3(self, model_type: str, symbol: str, timeframe: str, model_dir: Path) -> bool:
        return self._s3_backend.ensure_cached(
            model_type=model_type,
            symbol=symbol,
            timeframe=timeframe,
            model_dir=model_dir,
            prefix=self._s3_prefix(model_type, symbol, timeframe),
            model_key=self._s3_model_key(model_type, symbol, timeframe),
            metadata_key=self._s3_metadata_key(model_type, symbol, timeframe),
            metrics_key=self._s3_metrics_key(model_type, symbol, timeframe),
            metadata_path=self.get_metadata_path(model_type, symbol, timeframe),
        )

    def save_model(
        self,
        model_type: str,
        symbol: str,
        timeframe: str,
        model: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Persist a fitted model and optional metadata."""
        model_dir = self.get_model_dir(model_type, symbol, timeframe)
        model_dir.mkdir(parents=True, exist_ok=True)
        _emit_message(
            f"[ModelRouter] Starting save for {model_type} model ({symbol} @ {timeframe}) into {model_dir}"
        )
        if hasattr(model, "save_artifacts"):
            try:
                model.save_artifacts(model_dir)
                _emit_message(
                    f"[ModelRouter] Auxiliary artifacts saved for {model_type} ({symbol} @ {timeframe})"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Model %s for %s @ %s failed to save auxiliary artifacts: %s",
                    model_type,
                    symbol,
                    timeframe,
                    exc,
                )
                _emit_message(
                    f"[ModelRouter] WARNING: save_artifacts failed for {model_type} ({symbol} @ {timeframe}): {exc}"
                )
        model_path = model_dir / "model"
        saved_path, storage_format = save_model_safely(
            model,
            model_path,
            model_type,
            symbol,
            timeframe,
            logger,
        )
        _emit_message(
            f"[ModelRouter] Core model persisted for {model_type} ({symbol} @ {timeframe}) to {saved_path}"
        )
        manifest = {
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": model_type,
            "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metadata": metadata or {},
            "storage_format": storage_format,
            "model_file": saved_path.relative_to(model_dir).as_posix(),
            "class_path": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        }
        with open(self.get_metadata_path(model_type, symbol, timeframe), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        _emit_message(
            f"[ModelRouter] Metadata persisted for {model_type} ({symbol} @ {timeframe})"
        )
        self._upload_to_s3(model_dir, saved_path, model_type, symbol, timeframe)
        logger.info(
            "Saved %s model for %s @ %s to %s (%s)",
            model_type,
            symbol,
            timeframe,
            saved_path,
            storage_format,
        )

    def load_model(
        self,
        model_type: str,
        symbol: str,
        timeframe: str,
        *,
        device: Optional[str] = None,
    ) -> Optional[Any]:
        """Return the model if it exists, otherwise None."""
        model_dir = self.get_model_dir(model_type, symbol, timeframe)
        s3_synced = self._sync_from_s3(model_type, symbol, timeframe, model_dir)
        if self._s3_only and not s3_synced:
            raise RuntimeError(
                "S3-only mode enabled but S3 sync did not run for "
                f"{symbol} @ {timeframe} ({model_type})."
            )
        manifest_path = self.get_metadata_path(model_type, symbol, timeframe)
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read manifest for %s @ %s: %s", symbol, timeframe, exc)

        storage_format = manifest.get("storage_format")
        model_file = manifest.get("model_file")
        class_path = manifest.get("class_path")
        model: Any
        if storage_format and model_file and class_path:
            artifact_path = model_dir / model_file
            if not artifact_path.exists():
                if s3_synced:
                    raise FileNotFoundError(
                        f"No trained model available for {symbol} @ {timeframe} ({model_type})."
                    )
                return None
            try:
                model = self._load_from_artifact(
                    class_path,
                    storage_format,
                    artifact_path,
                    device=device,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load %s model for %s @ %s via %s: %s",
                    model_type,
                    symbol,
                    timeframe,
                    storage_format,
                    exc,
                )
                if self._s3_only:
                    raise RuntimeError(
                        f"Failed to load {model_type} model for {symbol} @ {timeframe}."
                    ) from exc
                return None
        else:
            legacy_path = self._legacy_model_path(model_type, symbol, timeframe)
            if not legacy_path.exists():
                if s3_synced:
                    raise FileNotFoundError(
                        f"No trained model available for {symbol} @ {timeframe} ({model_type})."
                    )
                return None
            try:
                with open(legacy_path, "rb") as handle:
                    model = pickle.load(handle)
                logger.info(
                    "Loaded legacy %s model for %s @ %s from %s",
                    model_type,
                    symbol,
                    timeframe,
                    legacy_path,
                )
            except (OSError, pickle.UnpicklingError) as exc:
                logger.warning("Failed to load %s model for %s @ %s: %s", model_type, symbol, timeframe, exc)
                if self._s3_only:
                    raise RuntimeError(
                        f"Failed to load legacy {model_type} model for {symbol} @ {timeframe}."
                    ) from exc
                return None

        logger.info("Loaded %s model for %s @ %s", model_type, symbol, timeframe)
        if hasattr(model, "set_device") and device:
            # Device is an execution concern; set it centrally after load.
            model.set_device(device)
        if hasattr(model, "load_artifacts"):
            try:
                model.load_artifacts(model_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Model %s for %s @ %s failed to load auxiliary artifacts: %s",
                    model_type,
                    symbol,
                    timeframe,
                    exc,
                )
        return model

    def _load_from_artifact(
        self,
        class_path: str,
        storage_format: str,
        artifact_path: Path,
        *,
        device: Optional[str] = None,
    ) -> Any:
        cls = _resolve_class(class_path)
        if storage_format == "native":
            instance = cls.load_native(artifact_path)
            if hasattr(instance, "set_device") and device:
                instance.set_device(device)
            return instance
        if storage_format == "state_dict":
            state = torch.load(artifact_path, map_location="cpu")
            instance = cls()
            if not hasattr(instance, "load_state_dict"):
                raise AttributeError(f"{class_path} lacks load_state_dict()")
            instance.load_state_dict(state)
            if hasattr(instance, "set_device") and device:
                instance.set_device(device)
            return instance
        raise ValueError(f"Unsupported storage format '{storage_format}' for {class_path}")

    # Convenience wrappers for volatility models -------------------------

    def save_egarch(self, symbol: str, timeframe: str, model: Any, metadata: Optional[dict[str, Any]] = None) -> None:
        """Persist EGARCH results under the dedicated namespace."""
        self.save_model("egarch", symbol, timeframe, model, metadata)

    def load_egarch(self, symbol: str, timeframe: str) -> Optional[Any]:
        """Return the EGARCH model if available."""
        return self.load_model("egarch", symbol, timeframe)


__all__ = ["ModelRouter"]
