import unittest

import numpy as np
import pandas as pd

from alphalens_forecast.config import AppConfig
from alphalens_forecast.forecasting import _select_regime_baseline
from alphalens_forecast.models.regime_baselines import (
    ARIMAForecaster,
    ARIMAParams,
    ETSForecaster,
    ETSParams,
    KalmanForecaster,
    KalmanParams,
)
from alphalens_forecast.regime_detection.deterministic import (
    REGIME_BREAKOUT,
    REGIME_RANGE,
    REGIME_STRESS_CHOP,
)


class TestRegimeBaselines(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range("2025-01-01", periods=80, freq="1h")
        values = np.linspace(100.0, 105.0, num=len(index))
        self.series = pd.Series(values, index=index)

    def test_arima_baseline_forecast_shape(self) -> None:
        model = ARIMAForecaster(params=ARIMAParams(p_values=(0, 1), d_values=(0, 1), q_values=(0,), max_order=2))
        model.fit(self.series)
        forecast = model.forecast(steps=6, freq="1h")
        self.assertEqual(len(forecast), 6)
        self.assertIn("yhat", forecast.columns)

    def test_ets_baseline_forecast_shape(self) -> None:
        model = ETSForecaster(params=ETSParams())
        model.fit(self.series)
        forecast = model.forecast(steps=5, freq="1h")
        self.assertEqual(len(forecast), 5)
        self.assertIn("yhat", forecast.columns)

    def test_kalman_baseline_forecast_shape(self) -> None:
        model = KalmanForecaster(params=KalmanParams())
        model.fit(self.series)
        forecast = model.forecast(steps=4, freq="1h")
        self.assertEqual(len(forecast), 4)
        self.assertIn("yhat", forecast.columns)

    def test_regime_selection_defaults(self) -> None:
        config = AppConfig()
        model = _select_regime_baseline(REGIME_RANGE, config=config)
        self.assertIsNotNone(model)
        self.assertEqual(getattr(model, "name", ""), "regime_mean_reversion")

    def test_regime_selection_arima(self) -> None:
        config = AppConfig()
        config.regime_range_model = "arima"
        model = _select_regime_baseline(REGIME_RANGE, config=config)
        self.assertIsInstance(model, ARIMAForecaster)

    def test_regime_selection_kalman(self) -> None:
        config = AppConfig()
        config.regime_stress_model = "kalman"
        model = _select_regime_baseline(REGIME_STRESS_CHOP, config=config)
        self.assertIsInstance(model, KalmanForecaster)

    def test_regime_selection_breakout_ets(self) -> None:
        config = AppConfig()
        config.regime_breakout_model = "ets"
        model = _select_regime_baseline(REGIME_BREAKOUT, config=config)
        self.assertIsInstance(model, ETSForecaster)


if __name__ == "__main__":
    unittest.main()
