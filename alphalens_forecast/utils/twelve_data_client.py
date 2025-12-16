"""Client utilities for fetching OHLCV data from Twelve Data."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests

from alphalens_forecast.config import TwelveDataConfig

logger = logging.getLogger(__name__)


class TwelveDataError(RuntimeError):
    """Raised when the Twelve Data API cannot return valid data."""


@dataclass
class TimeSeriesRequest:
    """Container for time-series request parameters."""

    symbol: str
    interval: str
    output_size: int


class TwelveDataClient:
    """Thin Twelve Data HTTP client with retry logic and data cleaning."""

    def __init__(
        self,
        config: Optional[TwelveDataConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config or TwelveDataConfig()
        self._session = session or requests.Session()

    def _interval_to_freq(self, interval: str) -> str:
        """Translate Twelve Data interval strings into pandas offsets."""
        mapping: Dict[str, str] = {
            "1min": "1min",
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "45min": "45min",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "12h": "12H",
            "1day": "1D",
        }
        freq = mapping.get(interval.lower())
        if freq is None:
            raise ValueError(f"Unsupported Twelve Data interval: {interval}")
        return freq

    def _build_params(
        self,
        request: TimeSeriesRequest,
        *,
        end_time: Optional[pd.Timestamp] = None,
    ) -> Dict[str, str]:
        """Build request parameters for Twelve Data queries."""
        params = {
            "symbol": request.symbol,
            "interval": request.interval,
            "apikey": self._config.api_key,
            "outputsize": str(request.output_size),
            "format": "JSON",
            "order": "DESC",
        }
        if end_time is not None:
            end_utc = end_time.tz_convert("UTC") if end_time.tzinfo else end_time.tz_localize("UTC")
            params["end_date"] = end_utc.strftime("%Y-%m-%d %H:%M:%S")
        return params

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        output_size: Optional[int] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from the Twelve Data API and return a cleaned DataFrame.

        Parameters
        ----------
        symbol:
            Trading symbol to request. Defaults to the configured symbol.
        interval:
            Twelve Data interval string. Defaults to the configured interval.
        output_size:
            Number of data points to request. Defaults to configured output size.
        end_time:
            Optional UTC timestamp limiting the latest sample returned.

        Returns
        -------
        pandas.DataFrame
            Cleaned OHLCV table indexed by UTC timestamp with computed log returns.

        Raises
        ------
        TwelveDataError
            If the API request fails or returns malformed data.
        """
        request = TimeSeriesRequest(
            symbol=symbol or self._config.symbol,
            interval=interval or self._config.interval,
            output_size=output_size or self._config.output_size,
        )
        params = self._build_params(request, end_time=end_time)
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < self._config.retry_attempts:
            try:
                response = self._session.get(
                    self._config.base_url,
                    params=params,
                    timeout=30,
                )
                if response.status_code != 200:
                    raise TwelveDataError(
                        f"Twelve Data responded with status {response.status_code}"
                    )
                payload = response.json()
                if "values" not in payload:
                    message = payload.get("message") or "No data returned"
                    raise TwelveDataError(message)
                df = self._clean_payload(payload["values"], request.interval)
                return df
            except (requests.RequestException, ValueError, TwelveDataError) as exc:
                last_error = exc
                attempt += 1
                sleep_time = self._config.retry_backoff ** attempt
                logger.warning(
                    "Twelve Data request failed (attempt=%s/%s): %s; retrying in %.1fs",
                    attempt,
                    self._config.retry_attempts,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)

        raise TwelveDataError(
            f"Failed to fetch Twelve Data time-series after "
            f"{self._config.retry_attempts} attempts: {last_error}"
        )

    def _clean_payload(self, values: list, interval: str) -> pd.DataFrame:
        """Transform Twelve Data payload into a processed OHLCV dataframe."""
        if not values:
            raise TwelveDataError("No datapoints received from Twelve Data.")

        df = pd.DataFrame(values)
        required_cols = {"datetime", "open", "high", "low", "close"}
        missing_required = required_cols - set(df.columns)
        if missing_required:
            raise TwelveDataError(f"Missing columns in Twelve Data response: {missing_required}")

        if "volume" not in df.columns:
            df["volume"] = np.nan

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        numeric_cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            numeric_cols.append("volume")
        for column in numeric_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["close"]).sort_values("datetime").set_index("datetime")

        freq = self._interval_to_freq(interval)
        df = df.resample(freq).last().ffill()

        if (df["close"] <= 0).any():
            logger.warning("Non-positive close prices detected; filtering affected rows.")
            df = df[df["close"] > 0]

        df["log_price"] = np.log(df["close"])
        df["log_return"] = df["log_price"].diff().fillna(0.0)
        df["return"] = df["close"].pct_change().fillna(0.0)

        return df
