"""Regime-aware risk and eligibility overlay (post-trade generation)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional


logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for the regime risk overlay."""

    stress_chop_size_scale: float = 0.25
    range_size_scale: float = 0.6
    min_vol_scale: float = 0.5
    max_vol_scale: float = 1.2
    trend_confidence_min: float = 0.70
    range_confidence_min: float = 0.85
    stress_confidence_min: float = 0.85
    breakout_confidence_min: float = 0.70
    direction_confidence_min: float = 0.60
    regime_confidence_min: float = 0.55
    low_regime_confidence_scale: float = 0.5
    guidance_key: str = "guidance"


class RegimeRiskOverlay:
    """Apply regime-aware eligibility and risk scaling to a trade payload."""

    def __init__(self, config: Optional[OverlayConfig] = None) -> None:
        self._config = config or OverlayConfig()

    def apply(self, trade: Any, context: Mapping[str, Any]) -> Any:
        """Apply the overlay to a trade payload or single trade dict."""
        if trade is None:
            return None
        if not isinstance(context, Mapping):
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
        if context.get("regime_label") is None and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Overlay: regime_label missing for %d/%d trades.", len(horizons), len(horizons))
        updated_horizons = []
        blocked_total = 0
        blocked_by_reason: Dict[str, int] = {}
        for horizon in horizons:
            if isinstance(horizon, dict):
                trade, reason = self._apply_trade_with_reason(dict(horizon), context)
                if reason:
                    blocked_total += 1
                    blocked_by_reason[reason] = blocked_by_reason.get(reason, 0) + 1
                updated_horizons.append(trade)
            else:
                updated_horizons.append(horizon)
        updated["horizons"] = updated_horizons
        if blocked_total and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Overlay blocked %d trades | reasons=%s",
                blocked_total,
                blocked_by_reason,
            )
        return updated

    def _apply_trade(self, trade: Dict[str, Any], context: Mapping[str, Any]) -> Dict[str, Any]:
        updated, _ = self._apply_trade_with_reason(trade, context)
        return updated

    def _apply_trade_with_reason(
        self,
        trade: Dict[str, Any],
        context: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        regime_label_raw = context.get("regime_label")
        regime_label = str(regime_label_raw) if regime_label_raw is not None else None
        direction = str(trade.get("direction") or "").lower()
        position_size = _coerce_float(trade.get("position_size"))
        if position_size is None:
            return trade, None

        performance_enabled = bool(context.get("performance_patches_enabled", False))
        breakout_allowed = self._breakout_allowed(context, trade)

        if regime_label is None:
            if performance_enabled and direction == "long":
                trade["position_size"] = 0.0
                logger.debug("Overlay: UNKNOWN regime blocked long trade.")
                return trade, "UNKNOWN_LONG"
            return trade, None

        if performance_enabled:
            rejection = self._performance_rejection_reason(context, trade, regime_label, breakout_allowed)
            if rejection:
                trade["position_size"] = 0.0
                logger.debug("Overlay: blocked trade due to %s.", rejection)
                return trade, rejection
        else:
            regime_conf = _coerce_float(context.get("regime_confidence"))
            if regime_conf is not None and regime_conf < self._config.regime_confidence_min:
                position_size *= self._config.low_regime_confidence_scale

        if regime_label == "STRESS_CHOP":
            if not breakout_allowed and not self._allow_stress_without_breakout(context, trade):
                trade["position_size"] = 0.0
                logger.debug("Overlay: STRESS_CHOP blocked trade.")
                return trade, "STRESS_CHOP"
            position_size *= self._config.stress_chop_size_scale

        if regime_label == "TREND_DOWN" and direction == "long":
            if not self._is_mean_reversion(trade):
                trade["position_size"] = 0.0
                logger.debug("Overlay: TREND_DOWN blocked long trade.")
                return trade, "TREND_DOWN_LONG"

        if regime_label == "TREND_UP" and direction == "short":
            if not self._is_mean_reversion(trade):
                trade["position_size"] = 0.0
                logger.debug("Overlay: TREND_UP blocked short trade.")
                return trade, "TREND_UP_SHORT"

        if regime_label == "RANGE":
            if not self._range_confidence_ok(context, trade):
                position_size *= self._config.range_size_scale

        vol_scale = self._volatility_scale(context, trade)
        if vol_scale is not None:
            position_size *= vol_scale

        trade["position_size"] = max(0.0, float(position_size))
        return trade, None

    def _performance_rejection_reason(
        self,
        context: Mapping[str, Any],
        trade: Mapping[str, Any],
        regime_label: str,
        breakout_allowed: bool,
    ) -> Optional[str]:
        trade_conf = _coerce_float(trade.get("confidence"))
        regime_conf = _coerce_float(context.get("regime_confidence"))

        trade_min = self._confidence_threshold_for_regime(regime_label, breakout_allowed)
        trade_min = self._adjust_threshold_for_volatility(trade_min, context, trade)

        if trade_conf is not None and trade_conf < trade_min:
            return "trade_confidence"
        if regime_conf is not None and regime_conf < self._config.regime_confidence_min:
            return "regime_confidence"
        return None

    def _confidence_threshold_for_regime(self, regime_label: str, breakout_allowed: bool) -> float:
        if regime_label == "STRESS_CHOP":
            return self._config.breakout_confidence_min if breakout_allowed else self._config.stress_confidence_min
        if regime_label == "RANGE":
            return self._config.range_confidence_min
        if regime_label == "BREAKOUT_VOL_EXPANSION":
            return self._config.breakout_confidence_min
        if regime_label in {"TREND_UP", "TREND_DOWN"}:
            return self._config.trend_confidence_min
        return self._config.trend_confidence_min

    def _adjust_threshold_for_volatility(
        self,
        threshold: float,
        context: Mapping[str, Any],
        trade: Mapping[str, Any],
    ) -> float:
        vol_ratio = self._vol_ratio(context, trade)
        if vol_ratio is None:
            return threshold
        if vol_ratio >= 1.5:
            return min(0.95, threshold + 0.05)
        return threshold

    def _breakout_allowed(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> bool:
        route = str(context.get("regime_route") or "").lower()
        if route == "breakout":
            return True
        if str(context.get("regime_label")) == "BREAKOUT_VOL_EXPANSION":
            return True
        strategy = str(trade.get("strategy") or trade.get("route") or "").lower()
        return "breakout" in strategy

    def _allow_stress_without_breakout(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> bool:
        regime_conf = _coerce_float(context.get("regime_confidence"))
        trade_conf = _coerce_float(trade.get("confidence"))
        if regime_conf is None or trade_conf is None:
            return False
        return regime_conf >= self._config.stress_confidence_min and trade_conf >= self._config.direction_confidence_min

    def _range_confidence_ok(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> bool:
        regime_conf = _coerce_float(context.get("regime_confidence"))
        trade_conf = _coerce_float(trade.get("confidence"))
        if regime_conf is None or trade_conf is None:
            return False
        return regime_conf >= self._config.range_confidence_min and trade_conf >= self._config.direction_confidence_min

    def _is_mean_reversion(self, trade: Mapping[str, Any]) -> bool:
        marker = str(trade.get("strategy") or trade.get("route") or "").lower()
        if not marker:
            model_info = trade.get("model")
            if isinstance(model_info, Mapping):
                marker = str(model_info.get("mean") or "").lower()
        if "mean" in marker and "reversion" in marker:
            return True
        if "ou" in marker:
            return True
        return False

    def _volatility_scale(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> Optional[float]:
        entry_vol = self._resolve_entry_vol(context, trade)
        vol_ref = _coerce_float(context.get("vol_ref"))
        if entry_vol is None or vol_ref is None or entry_vol <= 0 or vol_ref <= 0:
            return None
        raw = vol_ref / entry_vol
        return float(_clamp(raw, self._config.min_vol_scale, self._config.max_vol_scale))

    def _resolve_entry_vol(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> Optional[float]:
        entry_vol = _coerce_float(context.get("entry_model_vol"))
        if entry_vol is not None:
            return entry_vol
        vol_map = context.get("entry_model_vol_by_horizon")
        if isinstance(vol_map, Mapping):
            label = str(trade.get("h") or trade.get("horizon") or "").strip()
            if label:
                return _coerce_float(vol_map.get(label))
        return None

    def _vol_ratio(self, context: Mapping[str, Any], trade: Mapping[str, Any]) -> Optional[float]:
        entry_vol = self._resolve_entry_vol(context, trade)
        vol_ref = _coerce_float(context.get("vol_ref"))
        if entry_vol is None or vol_ref is None or entry_vol <= 0 or vol_ref <= 0:
            return None
        return float(entry_vol / vol_ref)


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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
