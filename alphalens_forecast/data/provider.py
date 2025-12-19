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
        auto_refresh: bool = False,
    ) -> None:
        self._config = config or TwelveDataConfig()
        default_cache = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_dir = Path(default_cache).resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or TwelveDataClient(self._config)
        self._auto_refresh = auto_refresh

    def get_cache_path(self, symbol: str, timeframe: str) -> Path:
        """Return the cache path for a (symbol, timeframe) pair."""
        symbol_slug = slugify(symbol)
        timeframe_slug = slugify(timeframe)
        return self._cache_dir / symbol_slug / f"{timeframe_slug}.csv"

    def _resolve_api_symbol(self, symbol: str) -> str:
        """
        Normalize symbols for Twelve Data without affecting cache naming.

        Twelve Data typically expects FX/crypto pairs formatted as ``BASE/QUOTE``.
        Locally we often pass/cache them as ``BASE_QUOTE``. To avoid breaking
        legacy cache keys while allowing refresh/range fetches, convert the
        pair format only when it is unambiguous.
        """
        raw = (symbol or "").strip()
        if "/" in raw or "_" not in raw:
            return raw
        parts = raw.split("_")
        if len(parts) != 2:
            return raw
        base, quote = parts[0].strip(), parts[1].strip()
        # Heuristic: treat 2-6 char alnum tickers as a pair (EUR_USD, BTC_USD, XLM_USD).
        if not (2 <= len(base) <= 6 and 2 <= len(quote) <= 6):
            return raw
        if not (base.isalnum() and quote.isalnum()):
            return raw
        normalized = f"{base}/{quote}"
        logger.debug("Normalized symbol for API: %s -> %s", raw, normalized)
        return normalized

    def _resolve_step(self, timeframe: str) -> Optional[pd.Timedelta]:
        try:
            offset = to_offset(timeframe)
            step = getattr(offset, "delta", None)
            if step is not None:
                return step
        except ValueError:
            pass
        try:
            return pd.Timedelta(timeframe)
        except (ValueError, TypeError):
            return None

    def _estimate_refresh_points(self, last_ts: pd.Timestamp, timeframe: str) -> Optional[int]:
        step = self._resolve_step(timeframe)
        if step is None or step <= pd.Timedelta(0):
            return None
        if last_ts.tzinfo is None:
            last_ts = last_ts.tz_localize("UTC")
        now_utc = pd.Timestamp.now(tz="UTC")
        if now_utc < last_ts + step:
            return 0
        delta = now_utc - last_ts
        if delta <= pd.Timedelta(0):
            return 0
        return int(delta / step)

    def _refresh_cache_if_stale(
        self,
        symbol: str,
        timeframe: str,
        cached: pd.DataFrame,
        cache_path: Path,
    ) -> pd.DataFrame:
        if cached.empty:
            return cached
        last_ts = cached.index.max()
        if pd.isna(last_ts):
            return cached
        missing = self._estimate_refresh_points(pd.to_datetime(last_ts, utc=True), timeframe)
        if missing is None or missing <= 0:
            return cached
        request_points = max(missing + 2, 1)
        api_symbol = self._resolve_api_symbol(symbol)
        try:
            if request_points > self._config.output_size:
                fetched = self._fetch_batched_history(api_symbol, timeframe, request_points)
            else:
                fetched = self._client.fetch_ohlcv(
                    symbol=api_symbol,
                    interval=timeframe,
                    output_size=request_points,
                )
        except (TwelveDataError, ValueError) as exc:
            logger.warning(
                "Cache refresh failed for %s @ %s: %s; serving stale cache.",
                symbol,
                timeframe,
                exc,
            )
            return cached

        if fetched is None or fetched.empty:
            return cached

        combined = pd.concat([cached, fetched]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        self._write_cache(cache_path, combined)
        logger.info(
            "Cache refreshed for %s @ %s | added=%d rows",
            symbol,
            timeframe,
            len(combined) - len(cached),
        )
        return combined

    def load_data(
        self,
        symbol: str,
        timeframe: str,
        refresh: bool = False,
        max_points: Optional[int] = None,
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        range_cache: str = "none",
    ) -> pd.DataFrame:
        """
        Return a historical dataframe, using cache when possible.

        Parameters
        ----------
        symbol, timeframe:
            Series identifier.
        refresh:
            Force a fresh download even if cache exists.
        start, end:
            Optional datetime bounds (inclusive) for range queries. When either is provided,
            the provider will not overwrite cached CSVs by default (read-only access).
        range_cache:
            Controls persistence behavior for range queries only:
            - ``"none"`` (default): read-only, do not modify cache on disk.
            - ``"merge"``: merge fetched history into the main cache CSV (append-only semantics).
            - ``"separate"``: write the returned subset to a separate cache file that includes the
              requested start/end in its filename. The main cache is left untouched.
        """
        range_start = pd.to_datetime(start, utc=True) if start is not None else None
        range_end = pd.to_datetime(end, utc=True) if end is not None else None
        if range_start is not None and range_end is not None and range_end < range_start:
            raise ValueError("end must be >= start for DataProvider.load_data().")
        is_range_query = range_start is not None or range_end is not None
        range_cache = (range_cache or "none").strip().lower()
        if range_cache not in {"none", "merge", "separate"}:
            raise ValueError("range_cache must be one of: 'none', 'merge', 'separate'.")
        should_merge_cache = is_range_query and range_cache == "merge"
        should_write_range = is_range_query and range_cache == "separate"

        cache_path = self.get_cache_path(symbol, timeframe)
        if not refresh:
            cached = self._read_cache(cache_path)
            if cached is not None:
                if self._auto_refresh and not is_range_query:
                    cached = self._refresh_cache_if_stale(symbol, timeframe, cached, cache_path)
                logger.debug("Serving %s @ %s from cache", symbol, timeframe)
                if is_range_query:
                    start_bound = range_start or cached.index.min()
                    end_bound = range_end or cached.index.max()
                    subset = cached.loc[(cached.index >= start_bound) & (cached.index <= end_bound)]
                    # If the cache fully covers the requested window, serve it directly.
                    if not subset.empty and subset.index.min() <= start_bound and subset.index.max() >= end_bound:
                        logger.info(
                            "Serving %s @ %s range from cache | start=%s end=%s rows=%d",
                            symbol,
                            timeframe,
                            start_bound,
                            end_bound,
                            len(subset),
                        )
                        if should_write_range:
                            self._write_cache(self._get_range_cache_path(symbol, timeframe, start_bound, end_bound), subset)
                        return subset
                    logger.info(
                        "Cache does not cover requested range for %s @ %s | want=%s..%s cache=%s..%s; fetching without persisting.",
                        symbol,
                        timeframe,
                        start_bound,
                        end_bound,
                        cached.index.min(),
                        cached.index.max(),
                    )
                else:
                    return cached.tail(max_points) if max_points else cached

        # Range queries should not overwrite cached CSVs; fetch as needed with persist=False.
        if not is_range_query:
            return self.load_latest(symbol, timeframe, persist=True, max_points=max_points)

        # Fetch history ending at range_end (or latest), merge with cache in-memory, and slice.
        # Persisting is controlled explicitly by ``range_cache`` to avoid accidental overwrites.
        fetched = self.load_latest(
            symbol,
            timeframe,
            persist=False,
            max_points=max_points,
            end_time=range_end,
        )
        cached = self._read_cache(cache_path)
        if cached is not None and not cached.empty:
            combined = pd.concat([cached, fetched]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = fetched

        if combined.empty:
            raise ValueError(f"No data available for {symbol} @ {timeframe}.")

        start_bound = range_start or combined.index.min()
        end_bound = range_end or combined.index.max()
        subset = combined.loc[(combined.index >= start_bound) & (combined.index <= end_bound)]
        if should_merge_cache:
            # Range-cache merge is intentionally append-only: we merge fetched data into the main cache.
            # This is safe for historical backfills and does not rewrite the requested subset elsewhere.
            self._write_cache(cache_path, combined)
            logger.info(
                "Merged range fetch into cache for %s @ %s | cache_rows=%d cache=%s",
                symbol,
                timeframe,
                len(combined),
                cache_path,
            )
        if should_write_range and not subset.empty:
            range_path = self._get_range_cache_path(symbol, timeframe, start_bound, end_bound)
            self._write_cache(range_path, subset)
            logger.info(
                "Persisted range cache for %s @ %s | rows=%d path=%s",
                symbol,
                timeframe,
                len(subset),
                range_path,
            )
        if subset.empty:
            logger.warning(
                "Range query returned no rows for %s @ %s | want=%s..%s available=%s..%s. "
                "Likely causes: stale cache, start/end outside available history, or max_points too small.",
                symbol,
                timeframe,
                start_bound,
                end_bound,
                combined.index.min(),
                combined.index.max(),
            )
        else:
            logger.info(
                "Serving %s @ %s range (cache not updated) | start=%s end=%s rows=%d",
                symbol,
                timeframe,
                start_bound,
                end_bound,
                len(subset),
            )
        return subset.copy()

    def load_latest(
        self,
        symbol: str,
        timeframe: str,
        persist: bool = True,
        max_points: Optional[int] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Download the latest history from the upstream provider.

        When ``persist`` is True the cache is updated after merging with any
        stored history so repeated calls only append new data.
        """
        requested_points = max_points if max_points and max_points > 0 else None
        api_symbol = self._resolve_api_symbol(symbol)
        if requested_points is not None and requested_points > self._config.output_size:
            frame = self._fetch_batched_history(api_symbol, timeframe, requested_points, end_time=end_time)
        else:
            frame = self._client.fetch_ohlcv(
                symbol=api_symbol,
                interval=timeframe,
                output_size=requested_points,
                end_time=end_time,
            )
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
        *,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Fetch extended history by issuing multiple Twelve Data requests."""
        remaining = max(max_points, 0)
        frames = []
        cursor = end_time
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
                    end_time=cursor,
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
            cursor = earliest - freq_offset
            if cursor.tzinfo is None:
                cursor = cursor.tz_localize("UTC")
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

    def _get_range_cache_path(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Path:
        """Return a dedicated filename for range-query caching (does not affect main cache)."""
        cache_dir = self.get_cache_path(symbol, timeframe).parent
        try:
            start_utc = pd.to_datetime(start, utc=True)
        except Exception:  # noqa: BLE001
            start_utc = pd.Timestamp(start).tz_localize("UTC")
        try:
            end_utc = pd.to_datetime(end, utc=True)
        except Exception:  # noqa: BLE001
            end_utc = pd.Timestamp(end).tz_localize("UTC")
        suffix = (
            f"{slugify(timeframe)}__range_"
            f"{start_utc.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{end_utc.strftime('%Y%m%dT%H%M%SZ')}.csv"
        )
        return cache_dir / suffix


__all__ = ["DataProvider"]
