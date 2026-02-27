"""Performance-focused trade overlay (regime-aware gating and risk controls)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerformanceOverlayConfig:
    """Configuration for performance overlay rules."""

    donchian_window: int = 20
    range_bottom_frac: float = 0.25
    breakout_confirm_bars: int = 2
    breakout_vol_ratio: float = 1.2
    anti_ext_ma_window: int = 20
    anti_ext_atr_window: int = 14
    anti_ext_threshold: float = 1.5


class PerformanceOverlay:
    """Apply performance-oriented trade gating using regime context."""

    def __init__(self, config: Optional[PerformanceOverlayConfig] = None) -> None:
        self._config = config or PerformanceOverlayConfig()

    def apply(self, trade: Any, context: Mapping[str, Any]) -> Any:
        if trade is None:
            return None
        if not isinstance(context, Mapping):
            return trade
        if not bool(context.get("performance_patches_enabled", False)):
            return trade
        if not bool(context.get("regime_enabled", False)):
            return trade
        if isinstance(trade, dict) and "horizons" in trade and isinstance(trade["horizons"], list):
            return self._apply_payload(trade, context)
        if isinstance(trade, dict):
            return self._apply_trade(trade, context)
        return trade

    def _apply_payload(self, payload: Dict[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
        updated = dict(payload)
        horizons = payload.get("horizons") or []
        if not isinstance(horizons, list):
            return updated
        updated_horizons = []
        blocked_total = 0
        for horizon in horizons:
            if isinstance(horizon, dict):
                trade, blocked = self._apply_trade_with_reason(dict(horizon), context)
                if blocked:
                    blocked_total += 1
                updated_horizons.append(trade)
            else:
                updated_horizons.append(horizon)
        updated["horizons"] = updated_horizons
        if blocked_total and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Performance overlay blocked %d trades.", blocked_total)
        return updated

    def _apply_trade(self, trade: Dict[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
        updated, _ = self._apply_trade_with_reason(trade, context)
        return updated

    def _apply_trade_with_reason(
        self, trade: Dict[str, Any], context: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Optional[str]]:
        regime_label = context.get("regime_label")
        if regime_label is None:
            return trade, None
        regime_label = str(regime_label)
        direction = str(trade.get("direction") or "").lower()
        position_size = _coerce_float(trade.get("position_size"))
        if position_size is None:
            return trade, None
        if position_size <= 0:
            return trade, None

        # Rule 1: BREAKOUT hardening
        if regime_label == "BREAKOUT_VOL_EXPANSION":
            if not self._breakout_confirmed(context, trade):
                trade["position_size"] = 0.0
                return trade, "BREAKOUT_SKIP"

        # Rule 2: RANGE long filter
        if regime_label == "RANGE" and direction == "long":
            if not self._range_long_allowed(context, trade):
                trade["position_size"] = 0.0
                return trade, "RANGE_LONG_SKIP"

        # Rule 3: Anti-extension filter
        if regime_label in {"TREND_UP", "TREND_DOWN", "BREAKOUT_VOL_EXPANSION"}:
            if self._is_overextended(context, trade):
                trade["position_size"] = 0.0
                return trade, "EXTENSION_SKIP"

        return trade, None

    def _breakout_confirmed(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> bool:
        # Prefer explicit hints if present
        route = str(context.get("regime_route") or "").lower()
        if "breakout_confirm" in route:
            return True
        if route == "breakout_confirmed":
            return True

        frame = _get_price_frame(context)
        if frame is None:
            return False

        df = frame.tail(max(self._config.donchian_window + 5, 10))
        if len(df) < 3:
            return False
        donchian_high, donchian_low = _donchian_bounds(df, self._config.donchian_window)
        if donchian_high is None or donchian_low is None:
            return False

        close = df["close"].to_numpy(dtype=float)
        direction = str(trade.get("direction") or "").lower()
        if direction == "long":
            breakout = close > donchian_high
        else:
            breakout = close < donchian_low
        if len(breakout) < self._config.breakout_confirm_bars:
            return False
        if not breakout[-self._config.breakout_confirm_bars :].all():
            return False

        atr_pct = _atr_pct(df, self._config.anti_ext_atr_window)
        if atr_pct is None:
            return False
        median = np.nanmedian(atr_pct[-max(5, self._config.donchian_window) :])
        if not np.isfinite(median) or median <= 0:
            return False
        vol_ratio = atr_pct[-1] / median
        return bool(vol_ratio >= self._config.breakout_vol_ratio)

    def _range_long_allowed(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> bool:
        frame = _get_price_frame(context)
        entry_price = _coerce_float(trade.get("entry_price"))
        if frame is None or entry_price is None:
            return True
        df = frame.tail(max(self._config.donchian_window + 2, 5))
        donchian_high, donchian_low = _donchian_bounds(df, self._config.donchian_window)
        if donchian_high is None or donchian_low is None:
            return True
        range_size = float(donchian_high - donchian_low)
        if range_size <= 0 or not np.isfinite(range_size):
            return True
        threshold = float(donchian_low + self._config.range_bottom_frac * range_size)
        return bool(entry_price <= threshold)

    def _is_overextended(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> bool:
        frame = _get_price_frame(context)
        entry_price = _coerce_float(trade.get("entry_price"))
        if frame is None or entry_price is None:
            return False
        df = frame.tail(max(self._config.anti_ext_ma_window + 2, self._config.anti_ext_atr_window + 2))
        if df.empty or len(df) < 3:
            return False
        close = df["close"].to_numpy(dtype=float)
        ma = np.nanmean(close[-self._config.anti_ext_ma_window :])
        atr_pct = _atr_pct(df, self._config.anti_ext_atr_window)
        if atr_pct is None or not np.isfinite(ma):
            return False
        atr = atr_pct[-1] * close[-1]
        if not np.isfinite(atr) or atr <= 0:
            return False
        extension = abs(entry_price - ma) / atr
        return bool(extension > self._config.anti_ext_threshold)


def _get_price_frame(context: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    frame = context.get("price_frame")
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        required = {"open", "high", "low", "close"}
        if required.issubset(frame.columns):
            return frame
    return None


def _donchian_bounds(df: pd.DataFrame, window: int) -> tuple[Optional[float], Optional[float]]:
    if df.empty:
        return None, None
    high = df["high"].rolling(window=window, min_periods=1).max().shift(1)
    low = df["low"].rolling(window=window, min_periods=1).min().shift(1)
    if high.isna().all() or low.isna().all():
        return None, None
    return float(high.iloc[-1]), float(low.iloc[-1])


def _atr_pct(df: pd.DataFrame, window: int) -> Optional[np.ndarray]:
    if df.empty:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    close_safe = close.replace(0.0, np.nan)
    atr_pct = (atr / close_safe).to_numpy(dtype=float)
    return atr_pct


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


__all__ = ["PerformanceOverlay", "PerformanceOverlayConfig"]
