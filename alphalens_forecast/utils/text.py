"""Utility helpers for text normalisation."""
from __future__ import annotations

import re


def slugify(value: str) -> str:
    """
    Convert arbitrary symbol/timeframe strings into filesystem-safe slugs.

    This helper centralises the behaviour so both CLI artifacts and model
    routing share the same naming convention.
    """
    cleaned = re.sub(r"[\\/:?\s]+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "artifact"


__all__ = ["slugify"]
