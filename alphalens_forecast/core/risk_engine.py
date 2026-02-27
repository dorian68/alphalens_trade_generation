"""Risk aggregation layer for AlphaLens forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import t

from alphalens_forecast.config import AppConfig, RiskConfig


@dataclass
class HorizonForecast:
    """Mean/volatility forecast summary for a specific horizon."""

    horizon_label: str
    median: float
    p20: float
    p80: float
    sigma: float
    dof: float
    drift: float
    model_name: str
    vol_model_name: str
    calibrated: bool
    probability_hit_tp_before_sl: Optional[float]
    last_price: float
    execution_price: Optional[float] = None
    quantiles_anchor: Optional[float] = None


class RiskEngine:
    """Combine mean and volatility forecasts into actionable trade setups."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or AppConfig()
        self._risk: RiskConfig = self._config.risk

    def _confidence(self, dof: float) -> float:
        """Probability that |error| < 1*sigma under a Student-t distribution."""
        if dof <= 2:
            return 0.0
        cdf_val = t.cdf(1.0, df=dof)
        return float(max(0.0, min(1.0, 2 * cdf_val - 1)))

    def _position_size(self, sigma: float, horizon_hours: Optional[float]) -> float:
        """Volatility-targeted position sizing capped by configuration."""
        if sigma <= 0:
            return 0.0
        annual_target = float(self._risk.target_volatility)
        if horizon_hours is not None and horizon_hours > 0:
            hours_per_year = 24.0 * 365.0
            horizon_target = annual_target * np.sqrt(horizon_hours / hours_per_year)
        else:
            horizon_target = annual_target
        size = horizon_target / sigma
        return float(min(size, self._risk.max_position))

    def build(
        self,
        symbol: str,
        as_of: str,
        timeframe: str,
        horizons: List[HorizonForecast],
        use_montecarlo: bool,
        trade_mode: str = "spot",
    ) -> Dict[str, object]:
        """Return the final structured JSON payload expected by the client."""
        trade_mode_normalized = str(trade_mode or "spot").strip().lower()
        if trade_mode_normalized not in {"spot", "forward"}:
            raise ValueError("trade_mode must be one of: spot, forward")
        payload = {
            "symbol": symbol,
            "asOf": as_of,
            "timeframe": timeframe,
            "use_montecarlo": use_montecarlo,
            "horizons": [],
        }
        root_entry_price: Optional[float] = None

        for horizon in horizons:
            execution_price = horizon.execution_price
            quantiles_anchor = horizon.quantiles_anchor or horizon.last_price
            if quantiles_anchor <= 0:
                quantiles_anchor = horizon.last_price
            direction = "long" if horizon.median >= horizon.last_price else "short"
            horizon_hours = _parse_horizon_hours(horizon.horizon_label)
            if trade_mode_normalized == "forward":
                entry_price = horizon.median
                entry_type = "conditional_forecast"
                entry_method = "median"
            else:
                if execution_price is not None and execution_price > 0:
                    entry_price = execution_price
                    entry_method = "execution_price"
                else:
                    entry_price = horizon.last_price
                    entry_method = "last_price"
                entry_type = "market"
            if root_entry_price is None:
                # Root entry_price mirrors the first horizon entry_price to stay trade_mode-aligned.
                root_entry_price = entry_price
            if direction == "long":
                tp = horizon.p80
                sl = horizon.p20
            else:
                tp = horizon.p20
                sl = horizon.p80
            if trade_mode_normalized == "spot" and execution_price is not None and execution_price > 0:
                scale = entry_price / quantiles_anchor if quantiles_anchor > 0 else 1.0
                tp *= scale
                sl *= scale
            if direction == "long":
                denominator = max(abs(entry_price - sl), 1e-9)
                risk_reward = abs(tp - entry_price) / denominator
            else:
                denominator = max(abs(sl - entry_price), 1e-9)
                risk_reward = abs(entry_price - tp) / denominator

            probability = horizon.probability_hit_tp_before_sl
            if probability is None:
                span = max(abs(tp - sl), 1e-9)
                reference = entry_price
                if direction == "long":
                    probability = max(0.0, min(1.0, (tp - reference) / span))
                else:
                    probability = max(0.0, min(1.0, (reference - tp) / span))

            confidence = self._confidence(horizon.dof)
            position_size = self._position_size(horizon.sigma, horizon_hours)

            payload["horizons"].append(
                {
                    "h": horizon.horizon_label,
                    "direction": direction,
                    "trade_mode": trade_mode_normalized,
                    "entry_price": entry_price,
                    "entry_type": entry_type,
                    "entry_method": entry_method,
                    "forecast": {
                        "medianPrice": horizon.median,
                        "p20": horizon.p20,
                        "p80": horizon.p80,
                    },
                    "tp": tp,
                    "sl": sl,
                    "riskReward": round(risk_reward, 4),
                    "prob_hit_tp_before_sl": round(probability, 4),
                    "confidence": round(confidence, 4),
                    "position_size": round(position_size, 4),
                    "model": {
                        "mean": horizon.model_name,
                        "vol": horizon.vol_model_name,
                        "calibrated": horizon.calibrated,
                    },
                }
            )

        if root_entry_price is not None:
            payload["entry_price"] = float(root_entry_price)

        return payload


def _parse_horizon_hours(label: str) -> Optional[float]:
    if not label:
        return None
    text = str(label).strip().lower()
    if text.endswith("h"):
        text = text[:-1]
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None
