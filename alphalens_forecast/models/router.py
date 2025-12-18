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

from alphalens_forecast.utils.text import slugify

logger = logging.getLogger(__name__)


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

    @property
    def base_dir(self) -> Path:
        return self._base_dir

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
        print(
            f"[ModelRouter] Starting save for {model_type} model ({symbol} @ {timeframe}) into {model_dir}"
        )
        if hasattr(model, "save_artifacts"):
            try:
                model.save_artifacts(model_dir)
                print(
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
                print(
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
        print(
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
        print(
            f"[ModelRouter] Metadata persisted for {model_type} ({symbol} @ {timeframe})"
        )
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
                return None
        else:
            legacy_path = self._legacy_model_path(model_type, symbol, timeframe)
            if not legacy_path.exists():
                logger.debug("No %s model found for %s @ %s", model_type, symbol, timeframe)
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
