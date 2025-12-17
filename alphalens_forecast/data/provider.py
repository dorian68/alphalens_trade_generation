"""Caching data provider that abstracts Twelve Data fetches."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset

from alphalens_forecast.config import TwelveDataConfig
from alphalens_forecast.utils import TwelveDataClient, TwelveDataError

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
from alphalens_forecast.utils.text import slugify

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Central access point for price data with simple filesystem caching.

    The provider keeps a per-(symbol, timeframe) CSV under ``alphalens_forecast/data/cache`` so
    repeated forecasts reuse local history instead of hammering the API. The
    implementation is storage-agnostic and can be swapped for S3/GCS later
    by overriding ``_read_cache``/``_write_cache``.
    """

    def __init__(
        self,
        config: Optional[TwelveDataConfig] = None,
        cache_dir: Optional[Path] = None,
        client: Optional[TwelveDataClient] = None,
    ) -> None:
        self._config = config or TwelveDataConfig()
        default_cache = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir = Path(default_cache).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or TwelveDataClient(self._config)

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Return the cache path for a (symbol, timeframe) pair."""
        symbol_slug = slugify(symbol)
        timeframe_slug = slugify(timeframe)
        return self._cache_dir / symbol_slug / f"{timeframe_slug}.csv"

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        refresh: bool = False,
        max_points: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Return a historical dataframe, using cache when possible.

        Parameters
        ----------
        symbol, timeframe:
            Series identifier.
        refresh:
            Force a fresh download even if cache exists.
        """
        cache_path = self.get_cache_path(symbol, timeframe)
        if not refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.debug("Serving %s @ %s from cache", symbol, timeframe)
                return cached.tail(max_points) if max_points else cached
        return self.load_latest(symbol, timeframe, persist=True, max_points=max_points)

    def load_latest(
        self,
        symbol: str,
        timeframe: str,
        persist: bool = True,
        max_points: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Download the latest history from the upstream provider.

        When ``persist`` is True the cache is updated after merging with any
        stored history so repeated calls only append new data.
        """
        requested_points = max_points if max_points and max_points > 0 else None
        if requested_points is not None and requested_points > self._config.output_size:
            frame = self._fetch_batched_history(symbol, timeframe, requested_points)
        else:
            frame = self._client.fetch_ohlcv(symbol=symbol, interval=timeframe, output_size=requested_points)
        cache_path = self.get_cache_path(symbol, timeframe)
        cached = self._read_cache(cache_path)
        if cached is not None:
            combined = pd.concat([cached, frame]).sort_index()
            frame = combined[~combined.index.duplicated(keep="last")]
        if max_points is not None and max_points > 0:
            frame = frame.tail(max_points)
        if persist:
            self._write_cache(cache_path, frame)
        return frame

    def _fetch_batched_history(
        self,
        symbol: str,
        timeframe: str,
        max_points: int,
    ) -> pd.DataFrame:
        """Fetch extended history by issuing multiple Twelve Data requests."""
        remaining = max(max_points, 0)
        frames = []
        end_time: Optional[pd.Timestamp] = None
        try:
            freq_offset = to_offset(timeframe)
        except ValueError:
            freq_offset = pd.Timedelta(timeframe)
        while remaining > 0:
            batch_size = min(self._config.output_size, remaining)
            try:
                batch = self._client.fetch_ohlcv(
                    symbol=symbol,
                    interval=timeframe,
                    output_size=batch_size,
                    end_time=end_time,
                )
            except TwelveDataError as exc:
                # Twelve Data responds with "Data not found" once historical limits are reached.
                if frames and "data not found" in str(exc).lower():
                    logger.info("Historical limit reached for %s @ %s; returning %s points.", symbol, timeframe, sum(len(f) for f in frames))
                    break
                raise
            if batch.empty:
                break
            frames.append(batch)
            remaining -= len(batch)
            earliest = batch.index.min()
            if earliest is None or len(batch) == 0:
                break
            end_time = earliest - freq_offset
            if end_time.tzinfo is None:
                end_time = end_time.tz_localize("UTC")
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames).sort_index()
        return combined.tail(max_points)

    def _read_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["datetime"])
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read cache %s: %s; ignoring cache.", path, exc)
            return None
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = frame.copy()
        serialisable.index.name = "datetime"
        try:
            serialisable.to_csv(path)
        except OSError as exc:
            logger.warning("Failed to write cache %s: %s", path, exc)


__all__ = ["DataProvider"]
