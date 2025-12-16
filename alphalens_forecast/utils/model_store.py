"""Persistence utilities for AlphaLens forecasting artifacts."""
from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class StoredArtifacts:
    """Container for saved model artifacts and metadata."""

    model_path: Path
    manifest_path: Path
    metadata: Dict[str, Any]
    payload: Optional[Dict[str, Any]] = None
    mean_model: Optional[Any] = None
    vol_model: Optional[Any] = None
    timestamp_slug: Optional[str] = None


class ModelStore:
    """Filesystem-backed storage for trained models and manifests."""

    def __init__(self, base_dir: Path, logger: logging.Logger) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger

    def save(
        self,
        prefix: str,
        mean_model: Any,
        vol_model: Any,
        metadata: Dict[str, Any],
        payload: Optional[Dict[str, Any]],
    ) -> StoredArtifacts:
        """Persist models, metadata, and payload to disk."""
        model_path = self.base_dir / f"{prefix}.pth"
        manifest_path = self.base_dir / f"{prefix}.json"
        payload_state = {
            "mean_class": f"{mean_model.__class__.__module__}.{mean_model.__class__.__qualname__}",
            "mean_state": mean_model.state_dict(),
            "vol_class": f"{vol_model.__class__.__module__}.{vol_model.__class__.__qualname__}",
            "vol_state": vol_model.state_dict(),
        }
        torch.save(payload_state, model_path)

        manifest = {
            "metadata": metadata,
            "payload": payload,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return StoredArtifacts(
            model_path=model_path,
            manifest_path=manifest_path,
            metadata=metadata,
            payload=payload,
            timestamp_slug=metadata.get("timestamp_slug"),
        )

    def load_latest(self, symbol_slug: str, timeframe_slug: str) -> Optional[StoredArtifacts]:
        """Load the most recent artifacts for a given symbol/timeframe combination."""
        pattern = f"{symbol_slug}_{timeframe_slug}_*.json"
        manifests = list(self.base_dir.glob(pattern))
        if not manifests:
            self._logger.debug(
                "No stored artifacts found for symbol=%s timeframe=%s in %s",
                symbol_slug,
                timeframe_slug,
                self.base_dir,
            )
            return None

        latest_manifest_path = max(manifests, key=lambda path: path.stat().st_mtime)
        prefix = latest_manifest_path.stem
        model_path = self.base_dir / f"{prefix}.pth"

        try:
            with open(latest_manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.warning("Failed to load manifest %s: %s", latest_manifest_path, exc)
            return None

        metadata = manifest.get("metadata", {})
        payload = manifest.get("payload")
        mean_model = None
        vol_model = None

        if model_path.exists():
            try:
                stored_models = torch.load(model_path, map_location="cpu")
                mean_model = self._instantiate_from_state(
                    stored_models.get("mean_class"),
                    stored_models.get("mean_state"),
                )
                vol_model = self._instantiate_from_state(
                    stored_models.get("vol_class"),
                    stored_models.get("vol_state"),
                )
            except (OSError, RuntimeError, AttributeError) as exc:
                self._logger.warning("Failed to load stored models %s: %s", model_path, exc)
        else:
            self._logger.warning("Model pickle missing for manifest %s", latest_manifest_path)

        artifacts = StoredArtifacts(
            model_path=model_path,
            manifest_path=latest_manifest_path,
            metadata=metadata,
            payload=payload,
            mean_model=mean_model,
            vol_model=vol_model,
            timestamp_slug=metadata.get("timestamp_slug"),
        )

        self._logger.info(
            "Loaded stored artifacts %s (hash=%s)",
            latest_manifest_path,
            metadata.get("data_hash"),
        )
        return artifacts

    def _instantiate_from_state(self, class_path: Optional[str], state: Optional[Dict[str, Any]]) -> Optional[Any]:
        if class_path is None or state is None:
            return None
        try:
            module_path, _, class_name = class_path.rpartition(".")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            instance = cls()
            if not hasattr(instance, "load_state_dict"):
                raise AttributeError(f"{class_path} lacks load_state_dict")
            instance.load_state_dict(state)
            return instance
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to instantiate %s: %s", class_path, exc)
            return None


__all__ = ["ModelStore", "StoredArtifacts"]
