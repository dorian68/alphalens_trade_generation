"""Risk aggregation layer for AlphaLens forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

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

    def _position_size(self, sigma: float) -> float:
        """Volatility-targeted position sizing capped by configuration."""
        if sigma <= 0:
            return 0.0
        per_period_target = self._risk.target_volatility
        size = per_period_target / sigma
        return float(min(size, self._risk.max_position))

    def build(
        self,
        symbol: str,
        as_of: str,
        timeframe: str,
        horizons: List[HorizonForecast],
        use_montecarlo: bool,
    ) -> Dict[str, object]:
        """Return the final structured JSON payload expected by the client."""
        payload = {
            "symbol": symbol,
            "asOf": as_of,
            "timeframe": timeframe,
            "use_montecarlo": use_montecarlo,
            "horizons": [],
        }

        for horizon in horizons:
            direction = "long" if horizon.median >= horizon.last_price else "short"
            if direction == "long":
                tp = horizon.p80
                sl = horizon.p20
                denominator = max(horizon.median - sl, 1e-9)
                risk_reward = (tp - horizon.median) / denominator
            else:
                tp = horizon.p20
                sl = horizon.p80
                denominator = max(sl - horizon.median, 1e-9)
                risk_reward = (horizon.median - tp) / denominator

            probability = horizon.probability_hit_tp_before_sl
            if probability is None:
                span = max(horizon.p80 - horizon.p20, 1e-9)
                if direction == "long":
                    probability = max(
                        0.0, min(1.0, (horizon.p80 - horizon.median) / span)
                    )
                else:
                    probability = max(
                        0.0, min(1.0, (horizon.median - horizon.p20) / span)
                    )

            confidence = self._confidence(horizon.dof)
            position_size = self._position_size(horizon.sigma)

            payload["horizons"].append(
                {
                    "h": horizon.horizon_label,
                    "direction": direction,
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

        return payload
