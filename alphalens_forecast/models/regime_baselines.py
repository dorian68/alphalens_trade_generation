"""Deterministic baseline forecasters used for regime-based routing."""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphalens_forecast.models.base import BaseForecaster

EPS = 1e-12


def _suppress_statsmodels_warnings() -> None:
    """Silence noisy statsmodels fit warnings for lightweight baselines."""
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except Exception:
        # statsmodels not installed or warning class moved.
        pass
    warnings.filterwarnings(
        "ignore",
        message="Non-stationary starting autoregressive parameters found.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="Non-invertible starting MA parameters found.*",
    )


def _build_future_index(last_index: pd.Timestamp, steps: int, freq: str) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(freq)
    start = last_index + offset
    return pd.date_range(start=start, periods=steps, freq=freq)


@dataclass
class MeanReversionParams:
    window: int = 50
    half_life: float = 10.0


@dataclass
class MomentumParams:
    window: int = 30
    max_abs_drift: float = 0.02


@dataclass
class ARIMAParams:
    p_values: Tuple[int, ...] = (0, 1, 2)
    d_values: Tuple[int, ...] = (0, 1)
    q_values: Tuple[int, ...] = (0, 1, 2)
    max_order: int = 5
    use_log: bool = True


@dataclass
class ETSParams:
    trend: Optional[str] = "add"
    damped_trend: bool = True
    use_log: bool = True


@dataclass
class OUParams:
    window: int = 60
    half_life: Optional[float] = None
    use_log: bool = True


@dataclass
class KalmanParams:
    use_log: bool = True


class FlatForecaster(BaseForecaster):
    """Flat (last-price) baseline for chaotic/high-vol regimes."""

    def __init__(self, name: str = "regime_flat_baseline") -> None:
        super().__init__(name=name)
        self._last_index: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        self._last_index = pd.to_datetime(target.index[-1])
        self._last_value = float(target.iloc[-1])

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_value is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        values = np.full(steps, self._last_value, dtype=float)
        return pd.DataFrame({"yhat": values}, index=index)


class MeanReversionForecaster(BaseForecaster):
    """Mean-reversion baseline for range-bound regimes."""

    def __init__(self, params: Optional[MeanReversionParams] = None) -> None:
        super().__init__(name="regime_mean_reversion")
        self._params = params or MeanReversionParams()
        self._last_index: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None
        self._mean_level: Optional[float] = None

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        target = target.astype(float)
        self._last_index = pd.to_datetime(target.index[-1])
        self._last_value = float(target.iloc[-1])
        window = max(2, int(self._params.window))
        recent = target.iloc[-window:] if len(target) >= window else target
        mean_level = float(recent.mean())
        if not np.isfinite(mean_level) or mean_level <= 0:
            mean_level = self._last_value
        self._mean_level = mean_level

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_value is None or self._mean_level is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        half_life = max(self._params.half_life, EPS)
        decay_rate = np.log(2.0) / half_life
        t = np.arange(1, steps + 1, dtype=float)
        decay = np.exp(-decay_rate * t)
        values = self._mean_level + (self._last_value - self._mean_level) * decay
        return pd.DataFrame({"yhat": values}, index=index)


class MomentumForecaster(BaseForecaster):
    """Momentum baseline for breakout regimes (damped linear trend in log space)."""

    def __init__(self, params: Optional[MomentumParams] = None) -> None:
        super().__init__(name="regime_momentum")
        self._params = params or MomentumParams()
        self._last_index: Optional[pd.Timestamp] = None
        self._last_log: Optional[float] = None
        self._drift: float = 0.0

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        target = target.astype(float)
        self._last_index = pd.to_datetime(target.index[-1])
        last_value = float(target.iloc[-1])
        self._last_log = float(np.log(max(last_value, EPS)))
        window = max(3, int(self._params.window))
        recent = target.iloc[-window:] if len(target) >= window else target
        if len(recent) < 2:
            self._drift = 0.0
            return
        log_vals = np.log(np.maximum(recent.values.astype(float), EPS))
        x = np.arange(len(log_vals), dtype=float)
        x = x - x.mean()
        y = log_vals - log_vals.mean()
        denom = np.sum(x**2)
        if denom <= 0:
            slope = 0.0
        else:
            slope = float(np.dot(x, y) / denom)
        max_drift = abs(float(self._params.max_abs_drift))
        self._drift = float(np.clip(slope, -max_drift, max_drift))

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_log is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        t = np.arange(1, steps + 1, dtype=float)
        log_path = self._last_log + self._drift * t
        values = np.exp(log_path)
        return pd.DataFrame({"yhat": values}, index=index)


class ARIMAForecaster(BaseForecaster):
    """ARIMA baseline with lightweight AIC grid selection."""

    def __init__(self, params: Optional[ARIMAParams] = None) -> None:
        super().__init__(name="regime_arima")
        self._params = params or ARIMAParams()
        self._last_index: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None
        self._model = None
        self._use_log = bool(self._params.use_log)

    def _prepare_series(self, target: pd.Series) -> np.ndarray:
        series = target.astype(float).to_numpy()
        if self._use_log:
            series = np.log(np.maximum(series, EPS))
        return series

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        self._last_index = pd.to_datetime(target.index[-1])
        self._last_value = float(target.iloc[-1])
        series = self._prepare_series(target)
        best_aic = np.inf
        best_res = None
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._model = None
            raise ImportError("statsmodels is required for ARIMA baseline.") from exc
        with warnings.catch_warnings():
            _suppress_statsmodels_warnings()
            for p in self._params.p_values:
                for d in self._params.d_values:
                    for q in self._params.q_values:
                        if p + d + q > self._params.max_order:
                            continue
                        try:
                            res = ARIMA(
                                series,
                                order=(int(p), int(d), int(q)),
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit()
                        except Exception:
                            continue
                        mle_retvals = getattr(res, "mle_retvals", None) or {}
                        if isinstance(mle_retvals, dict):
                            if mle_retvals.get("converged") is False:
                                continue
                            if int(mle_retvals.get("warnflag", 0)) != 0:
                                continue
                        aic = float(getattr(res, "aic", np.inf))
                        if np.isfinite(aic) and aic < best_aic:
                            best_aic = aic
                            best_res = res
        self._model = best_res

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_value is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        if self._model is None:
            values = np.full(steps, self._last_value, dtype=float)
            return pd.DataFrame({"yhat": values}, index=index)
        try:
            forecast_values = self._model.forecast(steps=steps)
        except Exception:
            values = np.full(steps, self._last_value, dtype=float)
            return pd.DataFrame({"yhat": values}, index=index)
        values = np.asarray(forecast_values, dtype=float)
        if self._use_log:
            values = np.exp(values)
        return pd.DataFrame({"yhat": values}, index=index)


class ETSForecaster(BaseForecaster):
    """Exponential smoothing baseline for range regimes."""

    def __init__(self, params: Optional[ETSParams] = None) -> None:
        super().__init__(name="regime_ets")
        self._params = params or ETSParams()
        self._last_index: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None
        self._model = None
        self._use_log = bool(self._params.use_log)

    def _prepare_series(self, target: pd.Series) -> np.ndarray:
        series = target.astype(float).to_numpy()
        if self._use_log:
            series = np.log(np.maximum(series, EPS))
        return series

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        self._last_index = pd.to_datetime(target.index[-1])
        self._last_value = float(target.iloc[-1])
        series = self._prepare_series(target)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._model = None
            raise ImportError("statsmodels is required for ETS baseline.") from exc
        try:
            with warnings.catch_warnings():
                _suppress_statsmodels_warnings()
                model = ExponentialSmoothing(
                    series,
                    trend=self._params.trend,
                    damped_trend=self._params.damped_trend,
                    seasonal=None,
                    initialization_method="estimated",
                )
                self._model = model.fit(optimized=True)
        except Exception:
            self._model = None

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_value is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        if self._model is None:
            values = np.full(steps, self._last_value, dtype=float)
            return pd.DataFrame({"yhat": values}, index=index)
        try:
            forecast_values = self._model.forecast(steps)
        except Exception:
            values = np.full(steps, self._last_value, dtype=float)
            return pd.DataFrame({"yhat": values}, index=index)
        values = np.asarray(forecast_values, dtype=float)
        if self._use_log:
            values = np.exp(values)
        return pd.DataFrame({"yhat": values}, index=index)


class OUForecaster(BaseForecaster):
    """Ornstein-Uhlenbeck style mean-reversion baseline."""

    def __init__(self, params: Optional[OUParams] = None) -> None:
        super().__init__(name="regime_ou")
        self._params = params or OUParams()
        self._last_index: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None
        self._mean_level: Optional[float] = None
        self._phi: float = 0.0
        self._use_log = bool(self._params.use_log)

    def _prepare_series(self, target: pd.Series) -> np.ndarray:
        series = target.astype(float).to_numpy()
        if self._use_log:
            series = np.log(np.maximum(series, EPS))
        return series

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        self._last_index = pd.to_datetime(target.index[-1])
        self._last_value = float(target.iloc[-1])
        series = self._prepare_series(target)
        window = max(3, int(self._params.window))
        recent = series[-window:] if series.size >= window else series
        mean_level = float(np.mean(recent)) if recent.size else float(series[-1])
        self._mean_level = mean_level
        if recent.size < 2:
            self._phi = 0.0
            return
        x = recent[:-1]
        y = recent[1:]
        x_centered = x - x.mean()
        denom = float(np.sum(x_centered**2))
        if denom <= 0:
            self._phi = 0.0
            return
        phi = float(np.dot(x_centered, y - y.mean()) / denom)
        if not np.isfinite(phi):
            phi = 0.0
        self._phi = float(np.clip(phi, -0.99, 0.99))

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_value is None or self._mean_level is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        last_value = float(np.log(max(self._last_value, EPS))) if self._use_log else float(self._last_value)
        t = np.arange(1, steps + 1, dtype=float)
        values = self._mean_level + (last_value - self._mean_level) * (self._phi**t)
        if self._use_log:
            values = np.exp(values)
        return pd.DataFrame({"yhat": values}, index=index)


class KalmanForecaster(BaseForecaster):
    """Local-level Kalman filter baseline for noisy regimes."""

    def __init__(self, params: Optional[KalmanParams] = None) -> None:
        super().__init__(name="regime_kalman")
        self._params = params or KalmanParams()
        self._last_index: Optional[pd.Timestamp] = None
        self._last_value: Optional[float] = None
        self._model = None
        self._use_log = bool(self._params.use_log)

    def _prepare_series(self, target: pd.Series) -> np.ndarray:
        series = target.astype(float).to_numpy()
        if self._use_log:
            series = np.log(np.maximum(series, EPS))
        return series

    def fit(self, target: pd.Series, regressors: Optional[pd.DataFrame] = None) -> None:
        del regressors
        if target.empty:
            raise ValueError("Target series is empty.")
        self._last_index = pd.to_datetime(target.index[-1])
        self._last_value = float(target.iloc[-1])
        series = self._prepare_series(target)
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self._model = None
            raise ImportError("statsmodels is required for Kalman baseline.") from exc
        try:
            with warnings.catch_warnings():
                _suppress_statsmodels_warnings()
                model = UnobservedComponents(series, level="llevel")
                self._model = model.fit(disp=False)
        except Exception:
            self._model = None

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        del future_regressors
        if self._last_index is None or self._last_value is None:
            raise RuntimeError("Model must be fit before forecasting.")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        index = _build_future_index(self._last_index, steps, freq)
        if self._model is None:
            values = np.full(steps, self._last_value, dtype=float)
            return pd.DataFrame({"yhat": values}, index=index)
        try:
            forecast_values = self._model.get_forecast(steps).predicted_mean
        except Exception:
            values = np.full(steps, self._last_value, dtype=float)
            return pd.DataFrame({"yhat": values}, index=index)
        values = np.asarray(forecast_values, dtype=float)
        if self._use_log:
            values = np.exp(values)
        return pd.DataFrame({"yhat": values}, index=index)


__all__ = [
    "FlatForecaster",
    "MeanReversionForecaster",
    "MomentumForecaster",
    "ARIMAForecaster",
    "ETSForecaster",
    "OUForecaster",
    "KalmanForecaster",
    "MeanReversionParams",
    "MomentumParams",
    "ARIMAParams",
    "ETSParams",
    "OUParams",
    "KalmanParams",
]
