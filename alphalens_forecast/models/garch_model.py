"""GARCH volatility modeling utilities."""
from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)

VOL_FLOOR = 1e-6
VOL_CEILING = 5.0
SCALE_FACTOR = 1000.0


@dataclass
class EGARCHForecast:
    """Container for EGARCH variance forecasts."""

    sigma: pd.Series
    variance: pd.Series
    dof: float
    skew: float
    method: str = "analytic"


class EGARCHVolModel:
    """Wrap the volatility engine to preserve backwards-compatible API."""

    def __init__(self) -> None:
        self._engine = _VolatilityEngine()

    def fit(self, residuals: pd.Series) -> None:
        """Fit residuals via the volatility engine."""
        self._engine.fit(residuals)

    def forecast(
        self,
        steps: int,
    ) -> EGARCHForecast:
        """Forecast conditional variance for the requested number of steps."""
        return self._engine.forecast(steps)

    def state_dict(self) -> Dict[str, Any]:
        return self._engine.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._engine.load_state_dict(state)


class _VolatilityEngine:
    """Encapsulate GARCH(1,1) + realized-vol fallback."""

    _CANDIDATES: Tuple[Dict[str, Any], ...] = (
        {"label": "GARCH", "vol": "GARCH", "p": 1, "o": 0, "q": 1},
    )

    def __init__(self) -> None:
        self._result = None
        self._model_name: str = "UNFIT"
        self._residual_std: float = 0.0  # stored in scaled units
        self._last_sigma: float = VOL_FLOOR
        self._vol_ceiling: float = VOL_CEILING
        self._realized_series: Optional[pd.Series] = None
        self._dof: float = 6.0
        self._skew: float = 0.0
        self._scale = SCALE_FACTOR

    def fit(self, residuals: pd.Series) -> None:
        clean = residuals.dropna()
        if clean.empty:
            raise ValueError("Residual series is empty; cannot fit volatility model.")
        std = float((clean * self._scale).std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1e-3
        self._residual_std = std
        self._vol_ceiling = max(VOL_CEILING, (std / self._scale) * 50.0)
        self._result = None
        self._realized_series = None
        self._model_name = "UNFIT"
        self._last_sigma = max(std / self._scale, VOL_FLOOR)

        for candidate in self._CANDIDATES:
            result = self._try_candidate(clean, candidate)
            if result is None:
                continue
            self._result = result
            self._model_name = candidate["label"]
            cond_vol = getattr(result, "conditional_volatility", None)
            if cond_vol is not None and not cond_vol.empty:
                self._last_sigma = float(cond_vol.iloc[-1]) / self._scale
            self._record_distribution_stats()
            self._log_model_selection(result, candidate["label"])
            return

        self._fallback_realized(clean)

    def forecast(self, steps: int) -> EGARCHForecast:
        if steps <= 0:
            raise ValueError("Forecast steps must be positive.")
        if self._model_name == "REALIZED":
            sigma_last = float(self._realized_series.iloc[-1])
            sigma_path = pd.Series(
                np.full(steps, max(sigma_last, VOL_FLOOR)),
                index=pd.RangeIndex(1, steps + 1),
                dtype=float,
            ).clip(upper=self._vol_ceiling)
            variance = (sigma_path**2).clip(lower=VOL_FLOOR**2, upper=self._vol_ceiling**2)
            return EGARCHForecast(
                sigma=sigma_path,
                variance=variance,
                dof=self._dof,
                skew=self._skew,
                method="realized_vol",
            )
        if self._result is None:
            raise RuntimeError("Volatility engine must be fitted before forecasting.")

        last_sigma = self._last_sigma if np.isfinite(self._last_sigma) else self._residual_std
        last_sigma = max(last_sigma, VOL_FLOOR)
        attempts: Iterable[Dict[str, Optional[str]]] = (
            {"name": "simulation", "method": "simulation"},
        )

        for attempt in attempts:
            try:
                forecast = self._dispatch_forecast(steps, attempt["method"])
                variance_raw = self._extract_variance(forecast, attempt["method"])
                variance = self._sanitize_variance(variance_raw, last_sigma)
                variance = variance / (self._scale**2)
                with np.errstate(over="raise", invalid="raise"):
                    sigma_raw = np.sqrt(variance)
                sigma = self._sanitize_sigma(sigma_raw, last_sigma)
                logger.debug(
                    "Volatility forecast using %s succeeded | range=(%.6f, %.6f)",
                    attempt["name"],
                    float(sigma.min()),
                    float(sigma.max()),
                )
                return EGARCHForecast(
                    sigma=sigma,
                    variance=variance,
                    dof=self._dof,
                    skew=self._skew,
                    method=attempt["name"],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Volatility forecast attempt using %s failed: %s",
                    attempt["name"],
                    exc,
                )
        logger.warning(
            "All volatility forecast attempts failed; using constant sigma %.6f",
            last_sigma,
        )
        sigma_fallback = pd.Series(
            np.full(steps, max(last_sigma, VOL_FLOOR)),
            index=pd.RangeIndex(start=1, stop=steps + 1),
            dtype=float,
        )
        variance_fallback = sigma_fallback**2
        return EGARCHForecast(
            sigma=sigma_fallback,
            variance=variance_fallback,
            dof=self._dof,
            skew=self._skew,
            method="fallback_constant",
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "residual_std": self._residual_std,
            "vol_ceiling": self._vol_ceiling,
            "last_sigma": self._last_sigma,
            "dof": self._dof,
            "skew": self._skew,
            "result": pickle.dumps(self._result) if self._result is not None else None,
            "realized_series": pickle.dumps(self._realized_series) if self._realized_series is not None else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._model_name = state.get("model_name", "UNFIT")
        self._residual_std = float(state.get("residual_std", 0.0))
        self._vol_ceiling = float(state.get("vol_ceiling", VOL_CEILING))
        self._last_sigma = float(state.get("last_sigma", VOL_FLOOR))
        self._dof = float(state.get("dof", 6.0))
        self._skew = float(state.get("skew", 0.0))
        result_blob = state.get("result")
        self._result = pickle.loads(result_blob) if result_blob is not None else None
        realized_blob = state.get("realized_series")
        self._realized_series = pickle.loads(realized_blob) if realized_blob is not None else None

    def _try_candidate(self, residuals: pd.Series, candidate: Dict[str, Any]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = arch_model(
                residuals * self._scale,
                dist="skewt",
                mean="AR",
                rescale=False,
                vol=candidate["vol"],
                p=candidate["p"],
                o=candidate["o"],
                q=candidate["q"],
            )
            result = model.fit(disp="off")
        if not self._is_valid_result(result, caught):
            return None
        return result

    def _is_valid_result(self, result, caught) -> bool:
        if getattr(result, "convergence_flag", 1) != 0:
            return False
        for warning_item in caught:
            if issubclass(warning_item.category, ConvergenceWarning):
                return False
        params = getattr(result, "params", pd.Series(dtype=float))
        if params.isna().any():
            return False
        alpha_val = None
        beta_val = None
        alpha_mask = params[[name for name in params.index if "alpha" in name.lower()]]
        beta_mask = params[[name for name in params.index if "beta" in name.lower()]]
        if not alpha_mask.empty:
            alpha_val = float(alpha_mask.iloc[0])
        if not beta_mask.empty:
            beta_val = float(beta_mask.iloc[0])
        if alpha_val is not None and beta_val is not None:
            if alpha_val + beta_val >= 0.9999:
                return False
        cond_vol = getattr(result, "conditional_volatility", None)
        if cond_vol is None or cond_vol.empty:
            return False
        cond_vol = cond_vol / self._scale
        if not np.isfinite(cond_vol).all():
            return False
        sigma_min = float(cond_vol.min())
        sigma_max = float(cond_vol.max())
        if sigma_min <= 0 or not np.isfinite(sigma_min):
            return False
        if sigma_max - sigma_min < VOL_FLOOR:
            return False
        if sigma_max > self._vol_ceiling * 10.0:
            return False
        return True

    def _fallback_realized(self, residuals: pd.Series) -> None:
        window = self._infer_realized_window(residuals.index)
        realized = (
            residuals.rolling(window, min_periods=max(5, window // 2))
            .std(ddof=0)
            .replace([np.inf, -np.inf], np.nan)
            .bfill()
            .ffill()
        )
        if realized.empty or not np.isfinite(realized.iloc[-1]) or realized.iloc[-1] <= 0:
            realized = pd.Series(np.full(len(residuals), max(self._residual_std, VOL_FLOOR)), index=residuals.index)
        realized = realized.clip(lower=VOL_FLOOR, upper=self._vol_ceiling)
        self._realized_series = realized
        self._model_name = "REALIZED"
        self._last_sigma = float(realized.iloc[-1])
        self._dof = 6.0
        self._skew = 0.0
        logger.warning(
            "Volatility engine falling back to realized volatility window=%s | sigma_last=%.6f",
            window,
            self._last_sigma,
        )

    def _record_distribution_stats(self) -> None:
        distribution = getattr(self._result, "distribution", None)
        self._dof = float(getattr(distribution, "nu", 6.0)) if distribution is not None else 6.0
        self._skew = float(getattr(distribution, "skew", 0.0)) if distribution is not None else 0.0

    def _log_model_selection(self, result, label: str) -> None:
        params = getattr(result, "params", pd.Series(dtype=float))
        cond_vol = getattr(result, "conditional_volatility", None)
        if cond_vol is not None:
            cond_vol = cond_vol / self._scale
        sigma_min = float(cond_vol.min())
        sigma_max = float(cond_vol.max())
        sigma_mean = float(cond_vol.mean())
        logger.info(
            "Volatility engine selected %s | convergence=%s | omega=%.6f | alpha=%.6f | gamma=%.6f | beta=%.6f "
            "| sigma_stats(min=%.6f, max=%.6f, mean=%.6f)",
            label,
            int(getattr(result, "convergence_flag", 1) == 0),
            float(params.get("omega", np.nan)),
            float(params.filter(like="alpha").iloc[0]) if params.filter(like="alpha").size else np.nan,
            float(params.filter(like="gamma").iloc[0]) if params.filter(like="gamma").size else np.nan,
            float(params.filter(like="beta").iloc[0]) if params.filter(like="beta").size else np.nan,
            sigma_min,
            sigma_max,
            sigma_mean,
        )
        if sigma_min <= VOL_FLOOR:
            logger.warning("Volatility floor breached (%.6f <= %.6f).", sigma_min, VOL_FLOOR)
        if sigma_max >= self._vol_ceiling:
            logger.warning("Volatility ceiling breached (%.6f >= %.6f).", sigma_max, self._vol_ceiling)
        summary = None
        try:
            summary = result.summary()
        except Exception:  # noqa: BLE001
            summary = None
        if summary is not None:
            logger.info("Volatility model summary:\n%s", summary.as_text())

    def _dispatch_forecast(self, steps: int, method: Optional[str]):
        kwargs = {"horizon": steps, "reindex": False}
        if method is not None:
            kwargs["method"] = method
            if method in {"simulation", "bootstrap"}:
                kwargs["simulations"] = max(1000, steps * 50)
                kwargs["random_state"] = 42
        return self._result.forecast(**kwargs)

    def _extract_variance(self, forecast, method: Optional[str]) -> pd.Series:
        if method == "simulation":
            return forecast.variance.mean(axis=0)
        return forecast.variance.iloc[-1]

    def _sanitize_variance(self, variance: pd.Series, fallback_sigma: float) -> pd.Series:
        clean = variance.astype(float).replace([np.inf, -np.inf], np.nan)
        if clean.isna().all():
            raise RuntimeError("Variance series contains only invalid values.")
        clean = clean.ffill().bfill()
        nominal_std = (self._residual_std / self._scale) if self._residual_std > 0 else fallback_sigma
        max_variance = max((nominal_std * 10.0) ** 2, fallback_sigma**2, VOL_FLOOR**2, self._vol_ceiling**2)
        clean = clean.clip(lower=VOL_FLOOR**2, upper=max_variance)
        return clean

    def _sanitize_sigma(self, sigma: pd.Series, fallback_sigma: float) -> pd.Series:
        clean = sigma.astype(float).replace([np.inf, -np.inf], np.nan)
        clean = clean.ffill().bfill()
        if clean.isna().any():
            raise RuntimeError("Sigma path contains non-finite values after sanitisation.")
        clean = clean.clip(lower=VOL_FLOOR, upper=self._vol_ceiling)
        return clean

    def _infer_realized_window(self, index: pd.Index) -> int:
        if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
            freq = pd.infer_freq(index)
            if freq:
                try:
                    offset = to_offset(freq)
                    minutes = offset.delta.total_seconds() / 60.0 if offset.delta else offset.nanos / 60e9
                    if minutes > 0:
                        return max(8, int(round((24 * 60) / minutes)))
                except ValueError:
                    pass
        return 48
