"""Minimal logging utilities to audit DataLoader settings without side effects."""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _normalize_dataloader(dataloader: Any) -> Any:
    if dataloader is None:
        return None
    if isinstance(dataloader, (list, tuple)):
        return dataloader[0] if dataloader else None
    return dataloader


def _resolve_from_trainer(trainer: Any) -> Tuple[Any, Optional[str]]:
    if trainer is None:
        return None, None
    loader = getattr(trainer, "train_dataloader", None)
    if loader is None:
        return None, None
    try:
        resolved = loader() if callable(loader) else loader
    except Exception:  # noqa: BLE001
        return None, None
    return _normalize_dataloader(resolved), "Lightning trainer"


def _resolve_from_model(model: Any) -> Tuple[Any, Optional[str]]:
    if model is None:
        return None, None
    trainer = getattr(model, "trainer", None) or getattr(model, "_trainer", None)
    dataloader, source = _resolve_from_trainer(trainer)
    if dataloader is not None:
        return dataloader, source
    for attr in ("train_dataloader", "_train_dataloader", "train_loader", "_train_loader"):
        loader = getattr(model, attr, None)
        if loader is None:
            continue
        try:
            resolved = loader() if callable(loader) else loader
        except Exception:  # noqa: BLE001
            continue
        dataloader = _normalize_dataloader(resolved)
        if dataloader is not None:
            return dataloader, f"model attribute ({attr})"
    return None, None


def log_dataloader_audit(
    *,
    model_name: str,
    device: str,
    batch_size: Optional[int],
    model: Any = None,
    source_hint: Optional[str] = None,
) -> None:
    """Log DataLoader settings without touching training behavior."""
    dataloader, source = _resolve_from_model(model)
    if source is None:
        source = source_hint or "library internal"
    if dataloader is not None and source_hint:
        source = f"{source_hint} via {source}"

    def _value(name: str) -> str:
        if dataloader is None:
            return "default (library-controlled)"
        value = getattr(dataloader, name, None)
        return "default (library-controlled)" if value is None else str(value)

    dl_batch = getattr(dataloader, "batch_size", None) if dataloader is not None else None
    batch_value = str(dl_batch) if dl_batch is not None else (
        f"{batch_size} (configured)" if batch_size is not None else "default (library-controlled)"
    )

    logger.info("[DataLoader Audit]")
    logger.info("Model: %s", model_name)
    logger.info("Device: %s", device)
    logger.info("Batch size: %s", batch_value)
    logger.info("num_workers: %s", _value("num_workers"))
    logger.info("persistent_workers: %s", _value("persistent_workers"))
    logger.info("pin_memory: %s", _value("pin_memory"))
    logger.info("DataLoader source: %s", source)
