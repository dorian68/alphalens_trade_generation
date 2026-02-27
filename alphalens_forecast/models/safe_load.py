"""Safe torch loading helpers for CPU/GPU compatibility."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


_CUDA_DESERIALIZE_HINTS = (
    "attempting to deserialize object on a cuda device",
    "torch.cuda.is_available() is false",
)


def _is_cuda_deserialize_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(hint in message for hint in _CUDA_DESERIALIZE_HINTS)


def _resolve_effective_device(prefer_device: Optional[str]) -> str:
    if prefer_device is None:
        return "cpu"
    resolved = prefer_device.strip().lower()
    if resolved == "auto":
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return resolved


def safe_torch_load(
    path: str | Path,
    *,
    prefer_device: Optional[str] = "auto",
    **kwargs: Any,
) -> Any:
    """
    Load a Torch artifact with a CPU-safe fallback.

    If prefer_device == "auto", CUDA is used when available.
    If prefer_device == "cpu", map_location is forced to CPU.
    On CUDA deserialization errors, retry once on CPU.
    """
    effective_device = _resolve_effective_device(prefer_device)
    map_location = kwargs.get("map_location")
    if effective_device.startswith("cpu"):
        map_location = torch.device("cpu")
        kwargs["map_location"] = map_location
    else:
        if "map_location" in kwargs and kwargs["map_location"] is None:
            kwargs.pop("map_location", None)

    logger.info(
        "Torch load | path=%s | device=%s | map_location=%s",
        str(path),
        effective_device,
        "cpu" if map_location is not None else "none",
    )
    try:
        return torch.load(path, **kwargs)
    except Exception as exc:  # noqa: BLE001
        if not _is_cuda_deserialize_error(exc):
            raise
        logger.warning(
            "Torch load CUDA deserialize error; retrying on CPU. path=%s",
            str(path),
        )
        kwargs["map_location"] = torch.device("cpu")
        return torch.load(path, **kwargs)


__all__ = ["safe_torch_load"]
