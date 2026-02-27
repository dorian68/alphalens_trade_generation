"""Contextual insights overlay for trade payloads (non-intrusive enrichment)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ContextInsightsConfig:
    """Configuration for contextual market insights."""

    bb_window: int = 20
    bb_std: float = 2.0
    atr_window: int = 14
    ema_window: int = 20
    vol_window: int = 20
    donchian_window: int = 20
    range_atr_mult: float = 1.0
    breakout_atr_mult: float = 1.0
    volume_window: int = 20
    min_history: int = 5
    max_insights: int = 4
    context_key: str = "context_insights"
    low_vol_threshold: float = 0.012
    high_vol_threshold: float = 0.05


class ContextInsightsOverlay:
    """Append contextual insights to trade payloads without altering signals."""

    def __init__(self, config: Optional[ContextInsightsConfig] = None) -> None:
        self._config = config or ContextInsightsConfig()

    def apply(self, trade: Any, context: Mapping[str, Any]) -> Any:
        if trade is None:
            return None
        if not isinstance(context, Mapping):
            return trade
        try:
            if isinstance(trade, dict) and "horizons" in trade and isinstance(trade["horizons"], list):
                return self._apply_payload(trade, context)
            if isinstance(trade, dict):
                return self._apply_trade(trade, context)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Context insights overlay failed; returning original trade. (%s)", exc)
            return trade
        return trade

    def _apply_payload(self, payload: Dict[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
        updated = dict(payload)
        metrics = _compute_market_metrics(context, self._config)
        summary_insights = _build_summary_insights(metrics, context, self._config)
        if summary_insights:
            updated[self._config.context_key] = summary_insights

        horizons = payload.get("horizons") or []
        if not isinstance(horizons, list):
            return updated
        updated_horizons = []
        for horizon in horizons:
            if isinstance(horizon, dict):
                updated_horizons.append(self._apply_trade(horizon, context, metrics))
            else:
                updated_horizons.append(horizon)
        updated["horizons"] = updated_horizons
        return updated

    def _apply_trade(
        self,
        trade: Dict[str, Any],
        context: Mapping[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        updated = dict(trade)
        metrics = metrics or _compute_market_metrics(context, self._config)
        insights = _build_trade_insights(metrics, context, trade, self._config)
        if insights:
            updated[self._config.context_key] = insights
        return updated


def enrich_model_output(
    signal: Dict[str, Any],
    data: pd.DataFrame,
    regime_state: Union[Mapping[str, Any], str, None],
    *,
    config: Optional[ContextInsightsConfig] = None,
) -> Dict[str, Any]:
    """
    Append contextual insights to an existing signal without altering the signal itself.

    This is a lightweight helper for direct integration into ad-hoc workflows.
    """
    context: Dict[str, Any] = {"price_frame": data}
    if isinstance(regime_state, Mapping):
        context.update(regime_state)
    elif regime_state is not None:
        context["regime_label"] = str(regime_state)
    return ContextInsightsOverlay(config=config).apply(signal, context)


def _compute_market_metrics(
    context: Mapping[str, Any],
    config: ContextInsightsConfig,
) -> Optional[Dict[str, float]]:
    frame = _get_price_frame(context)
    if frame is None or len(frame) < config.min_history:
        return None

    max_window = max(
        config.bb_window,
        config.atr_window,
        config.ema_window,
        config.vol_window,
        config.donchian_window,
        config.volume_window,
    )
    df = frame.tail(max_window + 5).copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    last_close = float(close.iloc[-1])
    sma = float(close.rolling(window=config.bb_window, min_periods=1).mean().iloc[-1])
    std = float(close.rolling(window=config.bb_window, min_periods=1).std(ddof=0).iloc[-1])
    upper = float(sma + config.bb_std * std)
    lower = float(sma - config.bb_std * std)
    ema = float(close.ewm(span=config.ema_window, adjust=False).mean().iloc[-1])

    # Step 1: compute ATR (TA-Lib True Range average; Chan, 2013).
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(window=config.atr_window, min_periods=1).mean().iloc[-1])
    atr_pct = float(atr / last_close) if last_close > 0 else np.nan

    return_std = float(
        close.pct_change()
        .rolling(window=config.vol_window, min_periods=2)
        .std(ddof=0)
        .iloc[-1]
    )

    donchian_high = float(
        high.rolling(window=config.donchian_window, min_periods=1).max().shift(1).iloc[-1]
    )
    donchian_low = float(
        low.rolling(window=config.donchian_window, min_periods=1).min().shift(1).iloc[-1]
    )
    if not np.isfinite(donchian_high):
        donchian_high = float(high.rolling(window=config.donchian_window, min_periods=1).max().iloc[-1])
    if not np.isfinite(donchian_low):
        donchian_low = float(low.rolling(window=config.donchian_window, min_periods=1).min().iloc[-1])

    vol_ratio = None
    if "volume" in df.columns:
        volume = df["volume"].astype(float)
        vol_mean = float(volume.rolling(window=config.volume_window, min_periods=1).mean().iloc[-1])
        if np.isfinite(vol_mean) and vol_mean > 0:
            vol_ratio = float(volume.iloc[-1] / vol_mean)

    return {
        "last_close": last_close,
        "sma": sma,
        "ema": ema,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_std": std,
        "atr": atr,
        "atr_pct": atr_pct,
        "return_std": return_std,
        "donchian_high": donchian_high,
        "donchian_low": donchian_low,
        "volume_ratio": float(vol_ratio) if vol_ratio is not None else np.nan,
    }


def _build_summary_insights(
    metrics: Optional[Dict[str, float]],
    context: Mapping[str, Any],
    config: ContextInsightsConfig,
) -> List[str]:
    # Keep a concise summary at the payload level.
    if metrics is None:
        return _fallback_insights(context, config)
    regime_label = str(context.get("regime_label") or "").strip().upper()
    regime_conf = _coerce_float(context.get("regime_confidence"))
    insights = []

    if regime_label:
        insights.append(_format_regime_header(regime_label, regime_conf))

    vol_label = _volatility_label(metrics.get("atr_pct"), config)
    if vol_label:
        ret_std = _fmt_pct(metrics.get("return_std"))
        insights.append(
            "Volatility: "
            f"ATR(14) {_fmt_price(metrics['atr'])} ({_fmt_pct(metrics['atr_pct'])}), {vol_label}; "
            f"20-bar return stdev {ret_std}."
        )

    return insights[: config.max_insights]


def _build_trade_insights(
    metrics: Optional[Dict[str, float]],
    context: Mapping[str, Any],
    trade: Mapping[str, Any],
    config: ContextInsightsConfig,
) -> List[str]:
    if metrics is None:
        return _fallback_insights(context, config)

    regime_label = str(context.get("regime_label") or "").strip().upper()
    regime_conf = _coerce_float(context.get("regime_confidence"))
    regime_scores = context.get("regime_scores")
    regime_mode = str(context.get("regime_mode_used") or "").strip().lower()

    insights: List[str] = []
    if regime_label:
        insights.append(_format_regime_header(regime_label, regime_conf))

    if regime_label in {"RANGE", "STRESS_CHOP"}:
        insights.extend(_range_insights(metrics, config))
    elif regime_label == "BREAKOUT_VOL_EXPANSION":
        insights.extend(_breakout_insights(metrics, config, regime_scores, regime_mode))
    else:
        insights.extend(_generic_insights(metrics, config))

    if regime_conf is not None and regime_conf < 0.55:
        insights.append(
            f"Regime confidence is low ({_fmt_pct(regime_conf)}); treat signals as tentative and monitor closely."
        )

    entry_price = _coerce_float(trade.get("entry_price"))
    sma = metrics.get("sma")
    if entry_price is not None and sma is not None and np.isfinite(sma) and sma != 0:
        diff = (entry_price - sma) / sma
        insights.append(f"Entry vs SMA(20): {_fmt_pct(diff)} from the mean.")

    return insights[: config.max_insights]


def _range_insights(metrics: Dict[str, float], config: ContextInsightsConfig) -> List[str]:
    upper = metrics["bb_upper"]
    lower = metrics["bb_lower"]
    last_close = metrics["last_close"]
    band_width = upper - lower
    band_pos = None
    if np.isfinite(band_width) and band_width > 0:
        band_pos = (last_close - lower) / band_width
    pos_label = _band_position_label(band_pos)

    # Step 2: interpret Bollinger Bands for range context (Bollinger, 2001).
    insights = [
        f"Range context: Bollinger(20, 2x std) { _fmt_price(lower) }-{ _fmt_price(upper) }; price is {pos_label}."
    ]

    atr_trigger_high = upper + config.range_atr_mult * metrics["atr"]
    atr_trigger_low = lower - config.range_atr_mult * metrics["atr"]
    insights.append(
        "Monitor for regime shift if price "
        f"> { _fmt_price(atr_trigger_high) } or < { _fmt_price(atr_trigger_low) } (band +/- ATR)."
    )

    return insights


def _breakout_insights(
    metrics: Dict[str, float],
    config: ContextInsightsConfig,
    regime_scores: Any,
    regime_mode: str,
) -> List[str]:
    high = metrics["donchian_high"]
    low = metrics["donchian_low"]
    atr = metrics["atr"]
    last_close = metrics["last_close"]
    trigger_high = high + config.breakout_atr_mult * atr
    trigger_low = low - config.breakout_atr_mult * atr

    insights = [
        f"Key levels: resistance { _fmt_price(high) }, support { _fmt_price(low) } (Donchian 20)."
    ]
    vol_ratio = metrics.get("volume_ratio")
    vol_hint = ""
    if vol_ratio is not None and np.isfinite(vol_ratio):
        vol_hint = f" (volume {vol_ratio:.2f}x avg)"
    insights.append(
        "Breakout watch: confirm above "
        f"{ _fmt_price(trigger_high) } or below { _fmt_price(trigger_low) } with volume expansion{vol_hint}."
    )

    # Step 3: translate regime scores (HMM when available) into shift risk (Hamilton, 1989),
    # or fall back to a deterministic heuristic so the shift-risk message is always present.
    shift_prob, source = _estimate_shift_probability(
        regime_scores,
        regime_mode,
        current_label="BREAKOUT_VOL_EXPANSION",
        metrics=metrics,
        config=config,
    )
    prefix = "HMM posterior" if source == "hmm" else ("Regime score" if source == "scores" else "Heuristic")
    insights.append(f"{prefix} implies ~{_fmt_pct(shift_prob)} shift risk.")

    vol_label = _volatility_label(metrics.get("atr_pct"), config)
    if vol_label:
        insights.append(f"Volatility context: {vol_label}; last close { _fmt_price(last_close) }.")

    return insights


def _generic_insights(metrics: Dict[str, float], config: ContextInsightsConfig) -> List[str]:
    vol_label = _volatility_label(metrics.get("atr_pct"), config)
    insight = (
        f"Context: price { _fmt_price(metrics['last_close']) } vs SMA(20) { _fmt_price(metrics['sma']) } "
        f"and EMA(20) { _fmt_price(metrics['ema']) }."
    )
    if vol_label:
        insight += f" Volatility is {vol_label}."
    return [insight]


def _estimate_shift_probability(
    regime_scores: Any,
    regime_mode: str,
    current_label: str,
    *,
    metrics: Dict[str, float],
    config: ContextInsightsConfig,
) -> tuple[float, str]:
    if isinstance(regime_scores, Mapping):
        current_score = _coerce_float(regime_scores.get(current_label))
        if current_score is not None:
            return float(np.clip(1.0 - current_score, 0.05, 0.95)), (
                "hmm" if regime_mode == "hmm" else "scores"
            )
        trend_score = 0.0
        for key in ("TREND_UP", "TREND_DOWN"):
            val = _coerce_float(regime_scores.get(key))
            if val is not None:
                trend_score += val
        if trend_score > 0:
            return float(np.clip(trend_score, 0.05, 0.95)), (
                "hmm" if regime_mode == "hmm" else "scores"
            )
    return _heuristic_shift_probability(metrics, config), "heuristic"


def _heuristic_shift_probability(metrics: Dict[str, float], config: ContextInsightsConfig) -> float:
    atr = metrics.get("atr")
    last_close = metrics.get("last_close")
    high = metrics.get("donchian_high")
    low = metrics.get("donchian_low")
    atr_pct = metrics.get("atr_pct")
    if atr is None or last_close is None or high is None or low is None:
        return 0.5
    if not (np.isfinite(atr) and np.isfinite(last_close) and np.isfinite(high) and np.isfinite(low)):
        return 0.5
    if atr <= 0:
        return 0.5

    trigger_high = high + config.breakout_atr_mult * atr
    trigger_low = low - config.breakout_atr_mult * atr
    dist = min(abs(trigger_high - last_close), abs(last_close - trigger_low))
    dist_atr = dist / atr if atr > 0 else 3.0
    proximity = 1.0 - float(np.clip(dist_atr / 3.0, 0.0, 1.0))
    if last_close >= trigger_high or last_close <= trigger_low:
        proximity = max(proximity, 0.7)

    vol_ratio = metrics.get("volume_ratio")
    vol_boost = 0.0
    if vol_ratio is not None and np.isfinite(vol_ratio):
        vol_boost = float(np.clip((vol_ratio - 1.0) / 1.0, 0.0, 1.0)) * 0.2

    vol_boost += 0.1 if (atr_pct is not None and np.isfinite(atr_pct) and atr_pct >= config.high_vol_threshold) else 0.0
    prob = 0.15 + 0.6 * proximity + vol_boost
    return float(np.clip(prob, 0.05, 0.95))


def _fallback_insights(context: Mapping[str, Any], config: ContextInsightsConfig) -> List[str]:
    label = str(context.get("regime_label") or "").strip().upper()
    conf = _coerce_float(context.get("regime_confidence"))
    if label or conf is not None:
        return [_format_regime_header(label or "UNKNOWN", conf)]
    return ["Regime unclear; monitor closely."]


def _format_regime_header(regime_label: str, regime_conf: Optional[float]) -> str:
    if regime_conf is None:
        return f"Regime: {regime_label}."
    return f"Regime: {regime_label} (confidence { _fmt_pct(regime_conf) })."


def _band_position_label(band_pos: Optional[float]) -> str:
    if band_pos is None or not np.isfinite(band_pos):
        return "within the band"
    if band_pos <= 0.2:
        return "near the lower band (mean-reversion favorable)"
    if band_pos >= 0.8:
        return "near the upper band (mean-reversion favorable)"
    return "mid-range"


def _volatility_label(atr_pct: Optional[float], config: ContextInsightsConfig) -> Optional[str]:
    if atr_pct is None or not np.isfinite(atr_pct):
        return None
    if atr_pct >= config.high_vol_threshold:
        return "elevated"
    if atr_pct <= config.low_vol_threshold:
        return "low"
    return "moderate"


def _fmt_price(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    abs_val = abs(float(value))
    if abs_val >= 1000:
        return f"{value:.2f}"
    if abs_val >= 100:
        return f"{value:.2f}"
    if abs_val >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _get_price_frame(context: Mapping[str, Any]) -> Optional[pd.DataFrame]:
    frame = context.get("price_frame")
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        required = {"open", "high", "low", "close"}
        if required.issubset(frame.columns):
            return frame
    return None


__all__ = [
    "ContextInsightsOverlay",
    "ContextInsightsConfig",
    "enrich_model_output",
]
