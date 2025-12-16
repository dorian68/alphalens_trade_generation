"""AlphaLens forecasting package."""

from typing import Any


def main(*args: Any, **kwargs: Any) -> Any:
    """Thin wrapper that defers importing the CLI entrypoint."""
    from .main import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
