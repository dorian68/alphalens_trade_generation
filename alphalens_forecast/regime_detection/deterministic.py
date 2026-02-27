"""Deterministic market regime detection (production-safe, rule-based).

Regimes:
- TREND_UP: persistent positive slope with low-to-moderate choppiness.
- TREND_DOWN: persistent negative slope with low-to-moderate choppiness.
- RANGE: low slope, low breakout pressure, and mean-reverting behavior.
- BREAKOUT_VOL_EXPANSION: price breaks a Donchian channel with volatility expanding.
- STRESS_CHOP: high choppiness with volatility stress and frequent reversals.

Features (rolling, deterministic):
- Log returns of close.
- ATR percent (ATR / close).
- Slope of log(close) over a trend window, normalized by ATR percent.
- Donchian breakout strength and ATR percent expansion.
- Choppiness proxy: sum(|returns|) / |sum(returns)|.
- Stress proxies: tail return, candle range expansion, sign-flip rate.

Limitations:
- This is a rule-based heuristic; it does not learn or adapt beyond rolling quantiles.
- Regimes are descriptive labels, not forecasts.

Example:
    >>> from alphalens_forecast.regime_detection.deterministic import (
    ...     DeterministicRegimeDetector,
    ... )
    >>> detector = DeterministicRegimeDetector()
    >>> result = detector.detect(df)
    >>> result.regime, result.confidence

    >>> # Optional Twelve Data fetch (requires TWELVE_DATA_API_KEY)
    >>> result = detector.detect_from_twelve_data(symbol="BTC/USD", interval="1h")

    >>> # Optional Twelve Data fetch via DataProvider (cache-aware)
    >>> result = detector.detect_from_data_provider(symbol="BTC/USD", interval="1h")
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

EPS = 1e-12

logger = logging.getLogger(__name__)

_FEATURE_COLUMNS_BASE = [
    "returns",
    "atr_pct",
    "atr_pct_median",
    "atr_pct_low",
    "atr_pct_high",
    "atr_pct_extreme",
    "slope_norm",
    "breakout_strength",
    "breakout_flag",
    "breakout_direction",
    "vol_expansion",
    "chop_ratio",
    "sign_flip_rate",
    "tail_return",
    "tail_score",
    "candle_range_pct",
    "candle_score",
]

REGIME_TREND_UP = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGE = "RANGE"
REGIME_BREAKOUT = "BREAKOUT_VOL_EXPANSION"
REGIME_STRESS_CHOP = "STRESS_CHOP"

REGIME_ORDER = [
    REGIME_TREND_UP,
    REGIME_TREND_DOWN,
    REGIME_RANGE,
    REGIME_BREAKOUT,
    REGIME_STRESS_CHOP,
]

HMM_OBS_COLS = (
    "slope_norm",
    "chop_ratio",
    "atr_pct",
    "breakout_strength",
    "vol_expansion",
    "sign_flip_rate",
    "tail_score",
    "candle_score",
)

if TYPE_CHECKING:
    from alphalens_forecast.config import TwelveDataConfig
    from alphalens_forecast.data.provider import DataProvider
    from alphalens_forecast.utils.twelve_data_client import TwelveDataClient


@dataclass
class RegimeConfig:
    """Configuration for deterministic regime detection."""

    lookback: int = 200
    enable_walk_forward_calib: bool = False
    calib_window: int = 200
    calib_min_history: int = 50
    calib_trend_quantile: float = 0.7
    calib_range_quantile: float = 0.3
    calib_chop_quantile: float = 0.7
    calib_vol_expansion_quantile: float = 0.7
    enable_score_smoothing: bool = False
    score_smoothing_alpha: float = 0.15
    enable_vol_mom_confirm: bool = False
    volume_window: int = 50
    volume_z_threshold: float = 1.0
    volume_ratio_threshold: float = 1.1
    momentum_window: int = 14
    momentum_threshold: float = 0.3
    momentum_clip: float = 5.0
    breakout_confirm_penalty: float = 0.5
    stress_confirm_penalty: float = 0.7
    regime_mode: str = "hmm"
    hmm_states: int = 5
    hmm_n_iter: int = 100
    hmm_covariance: str = "diag"
    hmm_random_state: int = 7
    atr_window: int = 14
    trend_window: int = 50
    donchian_window: int = 20
    chop_window: int = 50
    stress_window: int = 30
    atr_quantile_window: int = 100
    atr_low_quantile: float = 0.4
    atr_high_quantile: float = 0.9
    atr_extreme_quantile: float = 0.97
    vol_expansion_ratio: float = 1.2
    breakout_buffer: float = 0.0
    trend_threshold: float = 0.06
    stress_drift_max: float = 0.04
    trend_override_threshold: float = 0.06
    breakout_overrides_stress: bool = True
    trend_scale: float = 0.05
    slope_norm_clip: float = 5.0
    range_slope_max: float = 0.03
    breakout_distance_scale: float = 1.5
    chop_clip: float = 10.0
    chop_high_threshold: float = 4.0
    chop_target: float = 2.0
    tail_quantile: float = 0.05
    tail_atr_mult: float = 2.5
    flip_window: int = 20
    flip_threshold: float = 0.4
    candle_quantile: float = 0.9
    hysteresis_margin: float = 0.15
    hysteresis_persist_bars: int = 3
    confidence_temperature: float = 0.35


@dataclass
class RegimeResult:
    """Result of regime detection."""

    regime: str
    confidence: float
    features: Dict[str, float]
    scores: Dict[str, float]
    meta: Dict[str, Any]


@dataclass
class HMMRegimeModel:
    """Container for a fitted HMM regime model."""

    model: Any
    mean: np.ndarray
    std: np.ndarray
    regime_map: Dict[int, str]
    obs_cols: tuple[str, ...]
    config: RegimeConfig
    lookback: int


class DeterministicRegimeDetector:
    """Deterministic regime detector using rolling, rule-based features."""

    def __init__(self, config: Optional[RegimeConfig] = None) -> None:
        self.config = config or RegimeConfig()

    def detect(self, df: pd.DataFrame, previous_regime: Optional[str] = None) -> RegimeResult:
        """Detect the current regime from OHLCV data."""

        cleaned = _prepare_df(df, self.config, tail_only=True)
        if cleaned.empty or len(cleaned) < 2:
            return _insufficient_data_result(self.config, len(cleaned))

        features = _compute_features(cleaned, self.config)
        scores = _compute_scores_with_config(features, self.config)

        features = _sanitize_frame(features)
        scores = _sanitize_frame(scores)

        features_public = _features_for_output(features, scores)

        last_features = features_public.iloc[-1]
        last_scores = scores.iloc[-1]

        scores_values = scores.to_numpy()
        best_idx = int(np.argmax(scores_values[-1]))
        best_regime = scores.columns[best_idx]
        best_score = float(scores_values[-1, best_idx])
        second_best_score = float(_second_best(scores_values[-1]))

        selected_regime = best_regime

        if (self.config.regime_mode or "heuristic").strip().lower() == "hmm":
            base_labels = np.array([scores.columns[i] for i in np.argmax(scores_values, axis=1)], dtype=object)
            smoothed = _apply_hmm_postprocess(base_labels, scores, features, self.config)
            if len(smoothed):
                selected_regime = str(smoothed[-1])

        if previous_regime in last_scores.index and previous_regime is not None:
            prev_score = float(last_scores[previous_regime])
            if best_regime != previous_regime and best_score < prev_score + self.config.hysteresis_margin:
                if not _regime_persisted(scores, best_regime, self.config.hysteresis_persist_bars):
                    selected_regime = previous_regime

        history_ratio = _history_ratio(cleaned, self.config.lookback)
        selected_score = float(last_scores[selected_regime]) if selected_regime in last_scores.index else best_score
        confidence = _compute_confidence(
            selected_regime=selected_regime,
            best_regime=best_regime,
            best_score=best_score,
            second_best_score=second_best_score,
            selected_score=selected_score,
            temperature=self.config.confidence_temperature,
            history_ratio=history_ratio,
        )

        features_dict = {k: float(last_features[k]) for k in features_public.columns}
        scores_dict = {k: float(last_scores[k]) for k in scores.columns}

        meta = _build_meta(self.config, history_ratio, len(cleaned))
        try:
            mode_used = scores.attrs.get("regime_mode_used")
        except Exception:
            mode_used = None
        if mode_used:
            meta["mode_used"] = mode_used

        return RegimeResult(
            regime=selected_regime,
            confidence=confidence,
            features=features_dict,
            scores=scores_dict,
            meta=meta,
        )

    def detect_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-bar regimes for analysis or backtesting."""

        cleaned = _prepare_df(df, self.config, tail_only=False)
        if cleaned.empty or len(cleaned) < 2:
            return pd.DataFrame(columns=["regime", "confidence"], index=cleaned.index)

        features = _compute_features(cleaned, self.config)
        regime_mode = (self.config.regime_mode or "heuristic").strip().lower()
        if regime_mode == "hmm":
            scores = _compute_scores_hmm_walk_forward(features, self.config)
        else:
            scores = _compute_scores_with_config(features, self.config)
        features = _sanitize_frame(features)
        scores = _sanitize_frame(scores)
        features_public = _features_for_output(features, scores)

        scores_values = scores.to_numpy()
        best_idx = np.argmax(scores_values, axis=1)
        best_scores = scores_values[np.arange(scores_values.shape[0]), best_idx]
        second_best = _second_best(scores_values)

        regimes = np.array([scores.columns[i] for i in best_idx], dtype=object)
        if regime_mode == "hmm":
            regimes = _apply_hmm_postprocess(regimes, scores, features, self.config)

        label_to_idx = {label: i for i, label in enumerate(scores.columns)}
        selected_idx = np.array(
            [label_to_idx.get(str(label), int(best_idx[i])) for i, label in enumerate(regimes)],
            dtype=int,
        )
        selected_scores = scores_values[np.arange(scores_values.shape[0]), selected_idx]
        margins = np.where(
            selected_idx == best_idx,
            best_scores - second_best,
            selected_scores - best_scores,
        )
        confidence = _sigmoid(margins / self.config.confidence_temperature) * np.clip(selected_scores, 0.0, 1.0)
        history_ratio = _history_ratio_series(len(scores), self.config.lookback)
        confidence = np.clip(confidence * history_ratio, 0.0, 1.0)

        out = pd.DataFrame(
            {
                "regime": regimes,
                "confidence": confidence,
            },
            index=scores.index,
        )

        for col in scores.columns:
            out[f"score_{col}"] = scores[col].values

        for col in features_public.columns:
            out[f"feat_{col}"] = features_public[col].values

        return out

    def detect_from_twelve_data(
        self,
        symbol: str,
        interval: str,
        *,
        output_size: Optional[int] = None,
        end_time: Optional[pd.Timestamp] = None,
        twelve_data_config: Optional["TwelveDataConfig"] = None,
        client: Optional["TwelveDataClient"] = None,
    ) -> RegimeResult:
        """
        Fetch OHLCV from Twelve Data and detect the current regime.

        This helper is optional and does not enable caching or persistence by default.
        """
        frame = fetch_ohlcv_twelve_data(
            symbol=symbol,
            interval=interval,
            output_size=output_size,
            end_time=end_time,
            twelve_data_config=twelve_data_config,
            client=client,
        )
        return self.detect(frame)

    def detect_from_data_provider(
        self,
        symbol: str,
        interval: str,
        *,
        max_points: Optional[int] = None,
        end_time: Optional[pd.Timestamp] = None,
        provider: Optional["DataProvider"] = None,
        twelve_data_config: Optional["TwelveDataConfig"] = None,
        cache_dir: Optional[str] = None,
        auto_refresh: bool = False,
        refresh: bool = False,
        persist_cache: bool = False,
    ) -> RegimeResult:
        """
        Fetch OHLCV via DataProvider (Twelve Data + cache) and detect the regime.

        By default this does not persist the cache (persist_cache=False).
        """
        frame = fetch_ohlcv_data_provider(
            symbol=symbol,
            interval=interval,
            max_points=max_points,
            end_time=end_time,
            provider=provider,
            twelve_data_config=twelve_data_config,
            cache_dir=cache_dir,
            auto_refresh=auto_refresh,
            refresh=refresh,
            persist_cache=persist_cache,
        )
        return self.detect(frame)


def detect_regime(df: pd.DataFrame, config: Optional[RegimeConfig] = None) -> RegimeResult:
    """Convenience wrapper for single-call detection."""

    return DeterministicRegimeDetector(config=config).detect(df)


def detect_regime_from_twelve_data(
    symbol: str,
    interval: str,
    *,
    output_size: Optional[int] = None,
    end_time: Optional[pd.Timestamp] = None,
    detector_config: Optional[RegimeConfig] = None,
    twelve_data_config: Optional["TwelveDataConfig"] = None,
    client: Optional["TwelveDataClient"] = None,
) -> RegimeResult:
    """Convenience wrapper that fetches Twelve Data OHLCV then detects regime."""

    detector = DeterministicRegimeDetector(config=detector_config)
    return detector.detect_from_twelve_data(
        symbol=symbol,
        interval=interval,
        output_size=output_size,
        end_time=end_time,
        twelve_data_config=twelve_data_config,
        client=client,
    )


def detect_regime_from_data_provider(
    symbol: str,
    interval: str,
    *,
    max_points: Optional[int] = None,
    end_time: Optional[pd.Timestamp] = None,
    detector_config: Optional[RegimeConfig] = None,
    provider: Optional["DataProvider"] = None,
    twelve_data_config: Optional["TwelveDataConfig"] = None,
    cache_dir: Optional[str] = None,
    auto_refresh: bool = False,
    refresh: bool = False,
    persist_cache: bool = False,
) -> RegimeResult:
    """Convenience wrapper using DataProvider (Twelve Data + cache) then detecting regime."""

    detector = DeterministicRegimeDetector(config=detector_config)
    return detector.detect_from_data_provider(
        symbol=symbol,
        interval=interval,
        max_points=max_points,
        end_time=end_time,
        provider=provider,
        twelve_data_config=twelve_data_config,
        cache_dir=cache_dir,
        auto_refresh=auto_refresh,
        refresh=refresh,
        persist_cache=persist_cache,
    )


def fetch_ohlcv_data_provider(
    *,
    symbol: str,
    interval: str,
    max_points: Optional[int] = None,
    end_time: Optional[pd.Timestamp] = None,
    provider: Optional["DataProvider"] = None,
    twelve_data_config: Optional["TwelveDataConfig"] = None,
    cache_dir: Optional[str] = None,
    auto_refresh: bool = False,
    refresh: bool = False,
    persist_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data using the local DataProvider (Twelve Data + cache).

    Parameters
    ----------
    symbol, interval:
        Instrument and timeframe for Twelve Data.
    max_points:
        Optional number of data points to request.
    end_time:
        Optional timestamp limiting the latest sample (UTC).
    provider:
        Optional DataProvider instance.
    twelve_data_config:
        Optional TwelveDataConfig override. Used only when provider is not provided.
    cache_dir:
        Optional cache directory for DataProvider.
    auto_refresh:
        Enable DataProvider auto-refresh when serving cached history.
    refresh:
        Force fetching latest data from Twelve Data.
    persist_cache:
        When True, persist fetched data into the DataProvider cache.
        Note: providing end_time forces a live fetch via load_latest().
    """

    if provider is None:
        from alphalens_forecast.config import TwelveDataConfig as _TwelveDataConfig
        from alphalens_forecast.data.provider import DataProvider as _DataProvider

        config = twelve_data_config or _TwelveDataConfig()
        provider = _DataProvider(config=config, cache_dir=cache_dir, auto_refresh=auto_refresh)

    end_ts = pd.to_datetime(end_time, utc=True) if end_time is not None else None
    if end_ts is not None or refresh or not persist_cache:
        frame = provider.load_latest(
            symbol=symbol,
            timeframe=interval,
            persist=persist_cache,
            max_points=max_points,
            end_time=end_ts,
        )
    else:
        frame = provider.load_data(
            symbol=symbol,
            timeframe=interval,
            refresh=False,
            max_points=max_points,
        )

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"DataProvider response missing required columns: {missing}")

    cols = required + (["volume"] if "volume" in frame.columns else [])
    return frame[cols].copy()


def fetch_ohlcv_twelve_data(
    *,
    symbol: str,
    interval: str,
    output_size: Optional[int] = None,
    end_time: Optional[pd.Timestamp] = None,
    twelve_data_config: Optional["TwelveDataConfig"] = None,
    client: Optional["TwelveDataClient"] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Twelve Data using the local TwelveDataClient.

    Parameters
    ----------
    symbol, interval:
        Instrument and timeframe for Twelve Data.
    output_size:
        Optional number of data points to request.
    end_time:
        Optional timestamp limiting the latest sample (UTC).
    twelve_data_config:
        Optional TwelveDataConfig override. Used only when client is not provided.
    client:
        Optional TwelveDataClient instance (useful for testing or shared sessions).
    """

    if client is None:
        from alphalens_forecast.config import TwelveDataConfig as _TwelveDataConfig
        from alphalens_forecast.utils.twelve_data_client import TwelveDataClient as _TwelveDataClient

        config = twelve_data_config or _TwelveDataConfig()
        if not getattr(config, "api_key", ""):
            raise ValueError("TWELVE_DATA_API_KEY is not set; cannot fetch Twelve Data OHLCV.")
        client = _TwelveDataClient(config=config)

    end_ts = pd.to_datetime(end_time, utc=True) if end_time is not None else None
    frame = client.fetch_ohlcv(
        symbol=symbol,
        interval=interval,
        output_size=output_size,
        end_time=end_ts,
    )

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Twelve Data response missing required columns: {missing}")

    cols = required + (["volume"] if "volume" in frame.columns else [])
    return frame[cols].copy()


def _prepare_df(df: pd.DataFrame, config: RegimeConfig, tail_only: bool) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    cols = list(required)
    if "volume" in df.columns:
        cols.append("volume")
    cleaned = df[cols].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    before_rows = len(cleaned)
    cleaned = cleaned.dropna(subset=required, how="any")
    if before_rows != len(cleaned) and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Regime detection dropped %d/%d rows due to NaNs in %s.",
            before_rows - len(cleaned),
            before_rows,
            required,
        )

    if cleaned.empty:
        return cleaned

    if tail_only:
        max_window = _max_window(config)
        if len(cleaned) > max_window:
            cleaned = cleaned.iloc[-max_window:]

    return cleaned


def _max_window(config: RegimeConfig) -> int:
    windows = [
        config.lookback,
        config.atr_quantile_window,
        config.chop_window,
        config.trend_window,
        config.donchian_window,
        config.stress_window,
        config.flip_window,
    ]
    if config.enable_walk_forward_calib:
        windows.append(config.calib_window)
    if config.enable_vol_mom_confirm:
        windows.append(config.volume_window)
        windows.append(config.momentum_window)
    return max(windows)


def _compute_features(df: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None

    close_safe = close.replace(0.0, np.nan)
    log_close = np.log(close_safe)
    returns = log_close.diff()

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr = tr.rolling(window=config.atr_window, min_periods=1).mean()
    atr_pct = atr / close_safe

    atr_pct_median = atr_pct.rolling(window=config.atr_quantile_window, min_periods=1).median()
    atr_pct_low = atr_pct.rolling(window=config.atr_quantile_window, min_periods=1).quantile(
        config.atr_low_quantile
    )
    atr_pct_high = atr_pct.rolling(window=config.atr_quantile_window, min_periods=1).quantile(
        config.atr_high_quantile
    )
    atr_pct_extreme = atr_pct.rolling(window=config.atr_quantile_window, min_periods=1).quantile(
        config.atr_extreme_quantile
    )

    slope = _rolling_slope(log_close, config.trend_window)
    slope_norm = (slope / (atr_pct + EPS)).clip(-config.slope_norm_clip, config.slope_norm_clip)

    donchian_high = high.rolling(window=config.donchian_window, min_periods=1).max().shift(1)
    donchian_low = low.rolling(window=config.donchian_window, min_periods=1).min().shift(1)

    breakout_up = close > donchian_high * (1.0 + config.breakout_buffer)
    breakout_down = close < donchian_low * (1.0 - config.breakout_buffer)
    breakout_flag = (breakout_up | breakout_down).astype(float)

    breakout_strength = pd.Series(0.0, index=df.index)
    breakout_strength = breakout_strength.where(~breakout_up, (close - donchian_high) / (atr + EPS))
    breakout_strength = breakout_strength.where(~breakout_down, (donchian_low - close) / (atr + EPS))

    breakout_direction = pd.Series(0.0, index=df.index)
    breakout_direction = breakout_direction.where(~breakout_up, 1.0)
    breakout_direction = breakout_direction.where(~breakout_down, -1.0)

    vol_expansion = atr_pct / (atr_pct_median + EPS)

    sum_abs = returns.abs().rolling(window=config.chop_window, min_periods=2).sum()
    sum_ret = returns.rolling(window=config.chop_window, min_periods=2).sum().abs()
    chop_ratio = (sum_abs / (sum_ret + EPS)).clip(0.0, config.chop_clip)

    sign = np.sign(returns)
    flips = (sign * sign.shift(1) < 0).astype(float)
    sign_flip_rate = flips.rolling(window=config.flip_window, min_periods=2).mean()

    tail_return = returns.rolling(window=config.stress_window, min_periods=2).quantile(config.tail_quantile)
    tail_score = (-tail_return / ((atr_pct + EPS) * config.tail_atr_mult)).clip(0.0, 1.0)

    candle_range_pct = (high - low).abs() / close_safe
    candle_range_q = candle_range_pct.rolling(window=config.stress_window, min_periods=2).quantile(
        config.candle_quantile
    )
    candle_score = ((candle_range_pct / (candle_range_q + EPS)) - 1.0).clip(0.0, 1.0)

    momentum = returns.rolling(window=config.momentum_window, min_periods=2).mean()
    momentum_strength = (momentum.abs() / (atr_pct + EPS)).clip(0.0, config.momentum_clip)

    volume_z = None
    volume_ratio = None
    if volume is not None:
        volume_mean = volume.rolling(window=config.volume_window, min_periods=2).mean()
        volume_std = volume.rolling(window=config.volume_window, min_periods=2).std(ddof=0)
        volume_z = (volume - volume_mean) / (volume_std + EPS)
        volume_ratio = volume / (volume_mean + EPS)

    features = pd.DataFrame(
        {
            "returns": returns,
            "atr_pct": atr_pct,
            "atr_pct_median": atr_pct_median,
            "atr_pct_low": atr_pct_low,
            "atr_pct_high": atr_pct_high,
            "atr_pct_extreme": atr_pct_extreme,
            "slope_norm": slope_norm,
            "breakout_strength": breakout_strength,
            "breakout_flag": breakout_flag,
            "breakout_direction": breakout_direction,
            "vol_expansion": vol_expansion,
            "chop_ratio": chop_ratio,
            "sign_flip_rate": sign_flip_rate,
            "tail_return": tail_return,
            "tail_score": tail_score,
            "candle_range_pct": candle_range_pct,
            "candle_score": candle_score,
            "momentum_strength": momentum_strength,
        },
        index=df.index,
    )

    if volume_z is not None:
        features["volume_z"] = volume_z
    if volume_ratio is not None:
        features["volume_ratio"] = volume_ratio

    return features.replace([np.inf, -np.inf], np.nan)


def _compute_scores(features: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    slope_norm = features["slope_norm"]
    chop_ratio = features["chop_ratio"]
    atr_pct = features["atr_pct"]
    atr_pct_low = features["atr_pct_low"]
    atr_pct_high = features["atr_pct_high"]
    atr_pct_extreme = features["atr_pct_extreme"]
    breakout_strength = features["breakout_strength"]
    breakout_flag = features["breakout_flag"]
    vol_expansion = features["vol_expansion"]
    sign_flip_rate = features["sign_flip_rate"]
    tail_score = features["tail_score"]
    candle_score = features["candle_score"]
    momentum_strength = features["momentum_strength"] if "momentum_strength" in features.columns else None
    volume_ratio = features["volume_ratio"] if "volume_ratio" in features.columns else None
    volume_z = features["volume_z"] if "volume_z" in features.columns else None

    trend_threshold = config.trend_threshold
    range_slope_max = config.range_slope_max
    chop_high_threshold = config.chop_high_threshold
    vol_expansion_ratio = config.vol_expansion_ratio
    stress_drift_max = config.stress_drift_max
    trend_override_threshold = config.trend_override_threshold

    if config.enable_walk_forward_calib:
        window = max(1, min(config.calib_window, len(features)))
        min_history = max(2, min(config.calib_min_history, window))
        slope_abs = slope_norm.abs()
        trend_threshold = slope_abs.rolling(window=window, min_periods=min_history).quantile(
            config.calib_trend_quantile
        )
        range_slope_max = slope_abs.rolling(window=window, min_periods=min_history).quantile(
            config.calib_range_quantile
        )
        chop_high_threshold = chop_ratio.rolling(window=window, min_periods=min_history).quantile(
            config.calib_chop_quantile
        )
        vol_expansion_ratio = vol_expansion.rolling(window=window, min_periods=min_history).quantile(
            config.calib_vol_expansion_quantile
        )
        trend_threshold = trend_threshold.fillna(config.trend_threshold)
        range_slope_max = range_slope_max.fillna(config.range_slope_max)
        chop_high_threshold = chop_high_threshold.fillna(config.chop_high_threshold)
        vol_expansion_ratio = vol_expansion_ratio.fillna(config.vol_expansion_ratio)
        stress_drift_max = range_slope_max
        trend_override_threshold = trend_threshold

    trend_up = ((slope_norm - trend_threshold) / config.trend_scale).clip(0.0, 1.0)
    trend_down = ((-slope_norm - trend_threshold) / config.trend_scale).clip(0.0, 1.0)
    chop_penalty = (chop_ratio / chop_high_threshold).clip(0.0, 1.0) * 0.5
    trend_up = (trend_up * (1.0 - chop_penalty)).clip(0.0, 1.0)
    trend_down = (trend_down * (1.0 - chop_penalty)).clip(0.0, 1.0)

    breakout_score = (breakout_strength / config.breakout_distance_scale).clip(0.0, 1.0)
    vol_exp_score = ((vol_expansion - vol_expansion_ratio) / vol_expansion_ratio).clip(0.0, 1.0)
    breakout = (breakout_flag * (0.7 * breakout_score + 0.3 * vol_exp_score)).clip(0.0, 1.0)

    slope_abs = slope_norm.abs()
    slope_score = (1.0 - (slope_abs / range_slope_max).clip(0.0, 1.0)).clip(0.0, 1.0)
    vol_spread = (atr_pct_high - atr_pct_low).abs()
    vol_score = (1.0 - ((atr_pct - atr_pct_low) / (vol_spread + EPS)).clip(0.0, 1.0)).clip(0.0, 1.0)
    breakout_absent = (1.0 - breakout_score).clip(0.0, 1.0)
    chop_score = (1.0 - ((chop_ratio - config.chop_target).abs() / config.chop_target).clip(0.0, 1.0)).clip(
        0.0, 1.0
    )
    range_score = ((slope_score + vol_score + breakout_absent + chop_score) / 4.0).clip(0.0, 1.0)

    high_vol_score = ((atr_pct / (atr_pct_extreme + EPS)) - 1.0).clip(0.0, 1.0)
    chop_high_score = ((chop_ratio - chop_high_threshold) / chop_high_threshold).clip(0.0, 1.0)
    flip_score = (sign_flip_rate / config.flip_threshold).clip(0.0, 1.0)
    stress = (
        0.2 * high_vol_score
        + 0.25 * chop_high_score
        + 0.2 * flip_score
        + 0.2 * tail_score
        + 0.15 * candle_score
    ).clip(0.0, 1.0)

    stress_vol_ok = atr_pct >= atr_pct_high
    stress_chop_ok = (chop_high_score > 0.0) | (flip_score > 0.0)
    stress_drift_ok = slope_abs <= stress_drift_max
    stress_eligible = stress_vol_ok & stress_chop_ok & stress_drift_ok

    stress_override = slope_abs >= trend_override_threshold
    if config.breakout_overrides_stress:
        breakout_confirmed = (breakout_flag > 0.0) & (vol_expansion >= vol_expansion_ratio)
        stress_override = stress_override | breakout_confirmed

    stress = stress.where(stress_eligible & ~stress_override, 0.0)

    if config.enable_vol_mom_confirm:
        confirm_mask = pd.Series(True, index=features.index)
        if volume_ratio is not None:
            confirm_mask &= volume_ratio >= config.volume_ratio_threshold
        if volume_z is not None:
            confirm_mask &= volume_z >= config.volume_z_threshold
        if momentum_strength is not None:
            confirm_mask &= momentum_strength >= config.momentum_threshold
        breakout = breakout.where(confirm_mask, breakout * config.breakout_confirm_penalty)
        stress = stress.where(confirm_mask, stress * config.stress_confirm_penalty)

    return pd.DataFrame(
        {
            REGIME_TREND_UP: trend_up,
            REGIME_TREND_DOWN: trend_down,
            REGIME_RANGE: range_score,
            REGIME_BREAKOUT: breakout,
            REGIME_STRESS_CHOP: stress,
        },
        index=features.index,
    )


def _compute_scores_with_config(features: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    mode = (config.regime_mode or "heuristic").strip().lower()
    if mode == "hmm":
        scores = _compute_scores_hmm(features, config)
    else:
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
    mode_used = None
    try:
        mode_used = scores.attrs.get("regime_mode_used")
    except Exception:
        mode_used = None
    if config.enable_score_smoothing:
        scores = _smooth_scores(scores, config.score_smoothing_alpha)
        if mode_used:
            scores.attrs["regime_mode_used"] = mode_used
    return scores


def _smooth_scores(scores: pd.DataFrame, alpha: float) -> pd.DataFrame:
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError):
        return scores
    if not np.isfinite(alpha_value) or alpha_value <= 0.0 or alpha_value > 1.0:
        return scores
    smoothed = scores.ewm(alpha=alpha_value, adjust=False).mean()
    return smoothed.clip(0.0, 1.0)


def _features_for_output(features: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in _FEATURE_COLUMNS_BASE if col in features.columns]
    public = features[columns].copy()
    if REGIME_STRESS_CHOP in scores.columns:
        public["stress_score"] = scores[REGIME_STRESS_CHOP]
    return public


def _logsumexp(values: np.ndarray, axis: Optional[int] = None) -> Union[np.ndarray, float]:
    max_val = np.max(values, axis=axis, keepdims=True)
    max_val = np.where(np.isfinite(max_val), max_val, 0.0)
    stable = np.exp(values - max_val)
    summed = np.sum(stable, axis=axis, keepdims=True)
    out = np.log(summed + EPS) + max_val
    if axis is None:
        return float(out.ravel()[0])
    return np.squeeze(out, axis=axis)


def _hmm_filter_posterior(model: Any, obs_scaled: np.ndarray) -> np.ndarray:
    if obs_scaled.size == 0:
        return np.empty((0, 0), dtype=float)
    log_likelihood = model._compute_log_likelihood(obs_scaled)
    log_startprob = np.log(model.startprob_ + EPS)
    log_transmat = np.log(model.transmat_ + EPS)
    n_samples, n_states = log_likelihood.shape
    log_alpha = np.zeros((n_samples, n_states), dtype=float)
    log_alpha[0] = log_startprob + log_likelihood[0]
    log_alpha[0] -= _logsumexp(log_alpha[0])
    for t in range(1, n_samples):
        log_alpha[t] = log_likelihood[t] + _logsumexp(log_alpha[t - 1][:, None] + log_transmat, axis=0)
        log_alpha[t] -= _logsumexp(log_alpha[t])
    return np.exp(log_alpha)


def _map_posterior_to_regime(posterior_row: np.ndarray, regime_map: Dict[int, str]) -> np.ndarray:
    scores = np.zeros(len(REGIME_ORDER), dtype=float)
    if posterior_row is None:
        return scores
    for state_idx, regime in regime_map.items():
        if regime not in REGIME_ORDER:
            continue
        if state_idx >= len(posterior_row):
            continue
        scores[REGIME_ORDER.index(regime)] += float(posterior_row[state_idx])
    return scores


def _compute_scores_hmm_walk_forward(features: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM mode requested but hmmlearn unavailable; falling back to heuristic.")
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        if config.enable_score_smoothing:
            scores = _smooth_scores(scores, config.score_smoothing_alpha)
            scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    missing = [col for col in HMM_OBS_COLS if col not in features.columns]
    if missing:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM mode missing columns %s; falling back to heuristic.", missing)
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        if config.enable_score_smoothing:
            scores = _smooth_scores(scores, config.score_smoothing_alpha)
            scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    obs_all = features[list(HMM_OBS_COLS)].to_numpy(dtype=float)
    obs_all = np.nan_to_num(obs_all, nan=0.0, posinf=0.0, neginf=0.0)
    n_obs = len(obs_all)
    if n_obs < max(20, config.hmm_states * 10):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM mode insufficient data (n=%d); falling back to heuristic.", n_obs)
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        if config.enable_score_smoothing:
            scores = _smooth_scores(scores, config.score_smoothing_alpha)
            scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    heuristic_scores = _compute_scores(features, config).to_numpy(dtype=float)
    scores = np.zeros((n_obs, len(REGIME_ORDER)), dtype=float)
    hmm_used = False

    min_train = max(20, int(config.hmm_states) * 10)
    lookback = max(int(config.lookback), min_train)
    base_config = replace(
        config,
        enable_walk_forward_calib=False,
        enable_score_smoothing=False,
        enable_vol_mom_confirm=False,
        regime_mode="heuristic",
    )

    for idx in range(n_obs):
        end = idx + 1
        start = max(0, end - lookback)
        window_len = end - start
        if window_len < min_train:
            scores[idx] = heuristic_scores[idx]
            continue
        train_obs = obs_all[start:end]
        mean = train_obs.mean(axis=0)
        std = train_obs.std(axis=0)
        std[~np.isfinite(std) | (std <= 0)] = 1.0
        obs_scaled = (train_obs - mean) / std

        model = GaussianHMM(
            n_components=max(2, int(config.hmm_states)),
            covariance_type=config.hmm_covariance,
            n_iter=max(10, int(config.hmm_n_iter)),
            random_state=int(config.hmm_random_state),
        )
        try:
            model.fit(obs_scaled)
            posterior = _hmm_filter_posterior(model, obs_scaled)
            state_path = model.predict(obs_scaled)
        except Exception:
            scores[idx] = heuristic_scores[idx]
            continue

        train_features = features.iloc[start:end]
        state_means = _hmm_state_means(train_features, state_path, posterior)
        regime_map = _label_hmm_states(state_means, base_config)
        scores[idx] = _map_posterior_to_regime(posterior[-1], regime_map)
        hmm_used = True

    scores_df = pd.DataFrame(scores, index=features.index, columns=REGIME_ORDER).clip(0.0, 1.0)
    scores_df.attrs["regime_mode_used"] = "hmm" if hmm_used else "heuristic"
    if config.enable_score_smoothing:
        scores_df = _smooth_scores(scores_df, config.score_smoothing_alpha)
        scores_df.attrs["regime_mode_used"] = "hmm" if hmm_used else "heuristic"
    return scores_df


def _compute_scores_hmm(features: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM mode requested but hmmlearn unavailable; falling back to heuristic.")
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    missing = [col for col in HMM_OBS_COLS if col not in features.columns]
    if missing:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM mode missing columns %s; falling back to heuristic.", missing)
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    obs = features[list(HMM_OBS_COLS)].to_numpy(dtype=float)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if len(obs) < max(20, config.hmm_states * 10):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM mode insufficient data (n=%d); falling back to heuristic.", len(obs))
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    model = GaussianHMM(
        n_components=max(2, int(config.hmm_states)),
        covariance_type=config.hmm_covariance,
        n_iter=max(10, int(config.hmm_n_iter)),
        random_state=int(config.hmm_random_state),
    )
    try:
        lookback = min(len(obs), max(config.lookback, config.hmm_states * 10))
        train_obs = obs[-lookback:]
        mean = train_obs.mean(axis=0)
        std = train_obs.std(axis=0)
        std[~np.isfinite(std) | (std <= 0)] = 1.0
        obs_scaled = (obs - mean) / std
        train_scaled = obs_scaled[-lookback:]
        model.fit(train_scaled)
        posterior = _hmm_filter_posterior(model, obs_scaled)
        state_path = model.predict(train_scaled)
    except Exception:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HMM fit failed; falling back to heuristic.")
        scores = _compute_scores(features, config)
        scores.attrs["regime_mode_used"] = "heuristic"
        return scores

    base_config = replace(
        config,
        enable_walk_forward_calib=False,
        enable_score_smoothing=False,
        enable_vol_mom_confirm=False,
        regime_mode="heuristic",
    )

    train_features = features.iloc[-lookback:]
    state_means = _hmm_state_means(train_features, state_path, posterior[-lookback:])
    regime_map = _label_hmm_states(state_means, base_config)

    regime_scores = np.zeros((len(features), len(REGIME_ORDER)), dtype=float)
    for state_idx, regime in regime_map.items():
        if regime not in REGIME_ORDER:
            continue
        col_idx = REGIME_ORDER.index(regime)
        regime_scores[:, col_idx] += posterior[:, state_idx]

    scores = pd.DataFrame(regime_scores, index=features.index, columns=REGIME_ORDER).clip(0.0, 1.0)
    scores.attrs["regime_mode_used"] = "hmm"
    return scores


def _build_hmm_postprocess_features(features: pd.DataFrame, config: RegimeConfig) -> Dict[str, np.ndarray]:
    """Prepare auxiliary features for HMM post-processing."""
    feat: Dict[str, np.ndarray] = {}
    if "breakout_flag" in features.columns and "vol_expansion" in features.columns:
        breakout_confirmed = (features["breakout_flag"].to_numpy(dtype=float) > 0.0) & (
            features["vol_expansion"].to_numpy(dtype=float) >= float(config.vol_expansion_ratio)
        )
        feat["breakout_confirmed"] = breakout_confirmed
    if "slope_norm" in features.columns:
        feat["slope_norm"] = features["slope_norm"].to_numpy(dtype=float)
    if {"atr_pct", "atr_pct_high", "chop_ratio", "sign_flip_rate"}.issubset(features.columns):
        atr_pct = features["atr_pct"].to_numpy(dtype=float)
        atr_pct_high = features["atr_pct_high"].to_numpy(dtype=float)
        chop_ratio = features["chop_ratio"].to_numpy(dtype=float)
        sign_flip_rate = features["sign_flip_rate"].to_numpy(dtype=float)
        chop_high_score = ((chop_ratio - config.chop_high_threshold) / config.chop_high_threshold).clip(0.0, 1.0)
        flip_score = (sign_flip_rate / config.flip_threshold).clip(0.0, 1.0)
        stress_vol_ok = atr_pct >= atr_pct_high
        stress_chop_ok = (chop_high_score > 0.0) | (flip_score > 0.0)
        slope_abs = np.abs(features["slope_norm"].to_numpy(dtype=float))
        stress_drift_ok = slope_abs <= float(config.stress_drift_max)
        feat["stress_eligible"] = stress_vol_ok & stress_chop_ok & stress_drift_ok
    return feat


def _apply_hmm_postprocess(
    labels: np.ndarray,
    scores: pd.DataFrame,
    features: pd.DataFrame,
    config: RegimeConfig,
) -> np.ndarray:
    """Apply post-processing to HMM labels. Fail-safe on errors."""
    try:
        from alphalens_forecast.regime_detection.hmm_postprocess import SmoothConfig, smooth_regimes
    except Exception:
        return labels

    try:
        feat = _build_hmm_postprocess_features(features, config)
        smoothed = smooth_regimes(labels, scores.to_numpy(), SmoothConfig(), feat)
        return smoothed
    except Exception:
        return labels


def fit_hmm_regime_model(
    df: pd.DataFrame,
    config: Optional[RegimeConfig] = None,
    *,
    lookback: Optional[int] = None,
) -> HMMRegimeModel:
    """
    Fit an unsupervised HMM on the provided OHLCV data and return a reusable model.

    This does not alter production behavior; it is an optional helper for analysis/playgrounds.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("hmmlearn is required to fit HMM regime models.") from exc

    cfg = config or RegimeConfig()
    cleaned = _prepare_df(df, cfg, tail_only=False)
    if cleaned.empty or len(cleaned) < max(20, cfg.hmm_states * 10):
        raise ValueError("Insufficient data to fit HMM regime model.")

    features = _compute_features(cleaned, cfg)
    missing = [col for col in HMM_OBS_COLS if col not in features.columns]
    if missing:
        raise ValueError(f"Missing HMM feature columns: {missing}")

    obs = features[list(HMM_OBS_COLS)].to_numpy(dtype=float)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if len(obs) < max(20, cfg.hmm_states * 10):
        raise ValueError("Insufficient data to fit HMM regime model.")

    lookback_window = min(len(obs), max(lookback or cfg.lookback, cfg.hmm_states * 10))
    train_obs = obs[-lookback_window:]
    mean = train_obs.mean(axis=0)
    std = train_obs.std(axis=0)
    std[~np.isfinite(std) | (std <= 0)] = 1.0

    obs_scaled = (obs - mean) / std
    train_scaled = obs_scaled[-lookback_window:]

    model = GaussianHMM(
        n_components=max(2, int(cfg.hmm_states)),
        covariance_type=cfg.hmm_covariance,
        n_iter=max(10, int(cfg.hmm_n_iter)),
        random_state=int(cfg.hmm_random_state),
    )
    model.fit(train_scaled)

    posterior = _hmm_filter_posterior(model, train_scaled)
    state_path = model.predict(train_scaled)

    base_config = replace(
        cfg,
        enable_walk_forward_calib=False,
        enable_score_smoothing=False,
        enable_vol_mom_confirm=False,
        regime_mode="heuristic",
    )
    train_features = features.iloc[-lookback_window:]
    state_means = _hmm_state_means(train_features, state_path, posterior)
    regime_map = _label_hmm_states(state_means, base_config)

    return HMMRegimeModel(
        model=model,
        mean=mean,
        std=std,
        regime_map=regime_map,
        obs_cols=HMM_OBS_COLS,
        config=cfg,
        lookback=lookback_window,
    )


def apply_hmm_regime_model(df: pd.DataFrame, hmm_model: HMMRegimeModel) -> pd.DataFrame:
    """
    Apply a fitted HMM regime model to new OHLCV data.

    Returns a DataFrame with regime, confidence, and score columns (score_*).
    """
    cfg = hmm_model.config
    cleaned = _prepare_df(df, cfg, tail_only=False)
    if cleaned.empty or len(cleaned) < 2:
        return pd.DataFrame(columns=["regime", "confidence"], index=cleaned.index)

    features = _compute_features(cleaned, cfg)
    obs = features[list(hmm_model.obs_cols)].to_numpy(dtype=float)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    obs_scaled = (obs - hmm_model.mean) / hmm_model.std

    posterior = _hmm_filter_posterior(hmm_model.model, obs_scaled)
    regime_scores = np.zeros((len(features), len(REGIME_ORDER)), dtype=float)
    for state_idx, regime in hmm_model.regime_map.items():
        if regime not in REGIME_ORDER:
            continue
        col_idx = REGIME_ORDER.index(regime)
        regime_scores[:, col_idx] += posterior[:, state_idx]

    scores = pd.DataFrame(regime_scores, index=features.index, columns=REGIME_ORDER).clip(0.0, 1.0)
    if cfg.enable_score_smoothing:
        scores = _smooth_scores(scores, cfg.score_smoothing_alpha)

    scores_values = scores.to_numpy()
    best_idx = np.argmax(scores_values, axis=1)
    best_scores = scores_values[np.arange(scores_values.shape[0]), best_idx]
    second_best = _second_best(scores_values)
    margins = best_scores - second_best
    confidence = _sigmoid(margins / cfg.confidence_temperature) * np.clip(best_scores, 0.0, 1.0)
    history_ratio = _history_ratio_series(len(scores), cfg.lookback)
    confidence = np.clip(confidence * history_ratio, 0.0, 1.0)

    regimes = [scores.columns[i] for i in best_idx]
    out = pd.DataFrame({"regime": regimes, "confidence": confidence}, index=scores.index)
    for col in scores.columns:
        out[f"score_{col}"] = scores[col].values
    return out


def _hmm_state_means(
    features: pd.DataFrame,
    state_path: np.ndarray,
    posterior: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    n_states = posterior.shape[1]
    means: Dict[int, Dict[str, float]] = {}
    for state_idx in range(n_states):
        mask = state_path == state_idx
        if mask.any():
            subset = features.iloc[mask]
            state_values = subset.mean(numeric_only=True)
        else:
            weights = posterior[:, state_idx]
            weights = np.where(np.isfinite(weights), weights, 0.0)
            if weights.sum() <= 0:
                weights = np.ones(len(features), dtype=float)
            weighted = {}
            for col in _FEATURE_COLUMNS_BASE:
                if col not in features.columns:
                    continue
                values = features[col].to_numpy(dtype=float)
                weighted[col] = float(np.average(values, weights=weights))
            state_values = pd.Series(weighted)
        means[state_idx] = {col: float(state_values.get(col, 0.0)) for col in _FEATURE_COLUMNS_BASE}
    return means


def _label_hmm_states(
    state_means: Dict[int, Dict[str, float]],
    base_config: RegimeConfig,
) -> Dict[int, str]:
    if not state_means:
        return {}
    state_df = pd.DataFrame.from_dict(state_means, orient="index")
    state_scores = _compute_scores(state_df, base_config).fillna(0.0)
    mapping: Dict[int, str] = {}
    available_states = set(state_df.index.tolist())

    for regime in REGIME_ORDER:
        if not available_states:
            break
        best_state = None
        best_score = -np.inf
        for state_idx in available_states:
            score = float(state_scores.loc[state_idx, regime])
            if score > best_score:
                best_state = state_idx
                best_score = score
        if best_state is None:
            continue
        mapping[int(best_state)] = regime
        available_states.remove(best_state)

    for state_idx in sorted(available_states):
        row = state_scores.loc[state_idx]
        mapping[int(state_idx)] = str(row.idxmax())

    return mapping


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def _slope(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return 0.0
        x = np.arange(arr.size, dtype=float)
        x = x - x.mean()
        y = arr - arr.mean()
        denom = np.sum(x**2)
        if denom <= 0.0:
            return 0.0
        return float(np.dot(x, y) / denom)

    return series.rolling(window=window, min_periods=2).apply(_slope, raw=True)


def _second_best(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        if values.size < 2:
            return 0.0
        return float(np.partition(values, -2)[-2])
    return np.partition(values, -2, axis=1)[:, -2]


def _regime_persisted(scores: pd.DataFrame, candidate: str, bars: int) -> bool:
    if bars <= 1:
        return True
    if len(scores) < bars:
        return False
    recent_best = scores.tail(bars).idxmax(axis=1)
    return bool((recent_best == candidate).all())


def _history_ratio(df: pd.DataFrame, lookback: int) -> float:
    if lookback <= 0:
        return 1.0
    return min(1.0, len(df) / float(lookback))


def _history_ratio_series(length: int, lookback: int) -> np.ndarray:
    if lookback <= 0:
        return np.ones(length, dtype=float)
    return np.minimum(1.0, (np.arange(1, length + 1, dtype=float) / float(lookback)))


def _sigmoid(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _compute_confidence(
    selected_regime: str,
    best_regime: str,
    best_score: float,
    second_best_score: float,
    selected_score: float,
    temperature: float,
    history_ratio: float,
) -> float:
    if temperature <= 0:
        temperature = 1.0
    if selected_regime == best_regime:
        margin = best_score - second_best_score
    else:
        margin = selected_score - best_score
    confidence = _sigmoid(margin / temperature) * np.clip(selected_score, 0.0, 1.0)
    confidence *= history_ratio
    return float(np.clip(confidence, 0.0, 1.0))


def _sanitize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_meta(config: RegimeConfig, history_ratio: float, rows: int) -> Dict[str, Any]:
    return {
        "lookback": config.lookback,
        "atr_window": config.atr_window,
        "trend_window": config.trend_window,
        "donchian_window": config.donchian_window,
        "chop_window": config.chop_window,
        "stress_window": config.stress_window,
        "atr_quantile_window": config.atr_quantile_window,
        "hysteresis_margin": config.hysteresis_margin,
        "hysteresis_persist_bars": config.hysteresis_persist_bars,
        "confidence_temperature": config.confidence_temperature,
        "history_ratio": history_ratio,
        "rows_used": rows,
    }


def _insufficient_data_result(config: RegimeConfig, rows: int) -> RegimeResult:
    meta = _build_meta(config, history_ratio=0.0, rows=rows)
    meta["insufficient_data"] = True
    return RegimeResult(
        regime=REGIME_RANGE,
        confidence=0.0,
        features={},
        scores={},
        meta=meta,
    )
