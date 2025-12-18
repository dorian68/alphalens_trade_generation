"""Shared helpers for univariate TimeSeries construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # Darts is heavy; defer optional dependency errors until use.
    from darts import TimeSeries
except ImportError:  # pragma: no cover - optional dependency
    TimeSeries = None  # type: ignore[assignment]

TIMEFRAME_FREQ_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "45min": "45min",
    "1h": "1H",
    "2h": "2H",
    "3h": "3H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1day": "1D",
}


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean dataframe with ``datetime``/``close`` columns."""
    if "datetime" not in df.columns:
        raise KeyError("Dataframe must contain a 'datetime' column.")
    if "close" not in df.columns:
        raise KeyError("Dataframe must contain a 'close' column.")
    frame = df.copy()
    frame = frame.dropna(subset=["datetime", "close"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    frame = frame.sort_values("datetime")
    return frame


def build_timeseries(df: pd.DataFrame) -> TimeSeries:
    """
    Build a Darts ``TimeSeries`` using only datetime/close columns.

    Parameters
    ----------
    df:
        Dataframe with ``datetime`` and ``close`` columns.
    """
    if TimeSeries is None:
        raise ImportError("darts is required for build_timeseries; install it from requirements.txt")
    frame = _normalize_dataframe(df)
    return TimeSeries.from_dataframe(frame, time_col="datetime", value_cols="close")


def series_to_dataframe(series: pd.Series) -> pd.DataFrame:
    """Convert a ``pd.Series`` indexed by datetime into a two-column frame."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Target series must have a DatetimeIndex.")
    values = pd.to_numeric(series.astype(float), errors="coerce")
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(values.index, utc=True),
            "close": values.values,
        }
    )
    frame = frame.dropna(subset=["close"])
    frame = frame.sort_values("datetime")
    return frame


def timeseries_to_dataframe(series: TimeSeries, value_column: str) -> pd.DataFrame:
    """Convert a Darts ``TimeSeries`` back into a dataframe."""
    if TimeSeries is None:
        raise ImportError("darts is required for timeseries_to_dataframe; install it from requirements.txt")
    if hasattr(series, "pd_dataframe"):
        df = series.pd_dataframe()
    else:
        df = series.to_dataframe()
    columns = df.columns
    first_column = columns[0] if columns.nlevels == 1 else columns[0]
    values = df.loc[:, [first_column]].copy()
    values.columns = [value_column]
    values = values.reset_index()
    values = values.rename(columns={values.columns[0]: "datetime"})
    values["datetime"] = pd.to_datetime(values["datetime"], utc=True)
    return values


def series_to_price_frame(series: pd.Series) -> pd.DataFrame:
    """
    Convert a close-price series into the enriched price frame expected by the pipeline.

    The resulting dataframe mirrors the Twelve Data payload by computing log prices,
    log returns, and plain returns so downstream feature/volatility steps work unchanged.
    """
    base = series_to_dataframe(series)
    base = base.set_index("datetime")
    base["open"] = base["close"]
    base["high"] = base["close"]
    base["low"] = base["close"]
    base["volume"] = np.nan
    base["log_price"] = np.log(base["close"])
    base["log_return"] = base["log_price"].diff().fillna(0.0)
    base["return"] = base["close"].pct_change().fillna(0.0)
    return base


def align_series_to_timeframe(series: pd.Series, timeframe: str) -> pd.Series:
    """
    Return a resampled copy of ``series`` that matches the requested timeframe cadence.

    The input is left untouched; missing timestamps are forward filled so models that
    expect a regular DatetimeIndex (NHITS/Prophet/etc.) can operate on arbitrary data sources.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Target series must have a DatetimeIndex.")
    freq = TIMEFRAME_FREQ_MAP.get(timeframe.lower())
    if freq is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}' for alignment.")
    ordered = series.sort_index()
    ordered.index = pd.to_datetime(ordered.index, utc=True)
    resampled = ordered.resample(freq).last()
    if resampled.isna().any():
        resampled = resampled.fillna(method="ffill")
    resampled = resampled.dropna()
    if resampled.empty:
        raise ValueError("Aligned series is empty after resampling; check input data.")
    return resampled


__all__ = [
    "build_timeseries",
    "series_to_dataframe",
    "series_to_price_frame",
    "timeseries_to_dataframe",
    "align_series_to_timeframe",
]
