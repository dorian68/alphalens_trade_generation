import os
import tempfile
import unittest
from unittest.mock import patch

from alphalens_forecast.config import AppConfig
from alphalens_forecast.forecasting import _regime_model_type, _select_regime_baseline
from alphalens_forecast.models.regime_baselines import FlatForecaster
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.regime_detection.deterministic import REGIME_RANGE


class TestRegimeBaselineFallback(unittest.TestCase):
    def _save_model(self, router: ModelRouter, model_type: str, symbol: str, timeframe: str, name: str) -> None:
        model = FlatForecaster()
        model.name = name
        router.save_model(model_type, symbol, timeframe, model, metadata={"name": name})

    def test_fallback_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "ALPHALENS_S3_ONLY": "0",
                    "ALPHALENS_REQUIRE_S3": "0",
                    "ALPHALENS_MODEL_BUCKET": "",
                },
                clear=False,
            ):
                router = ModelRouter(base_dir=temp_dir)
                model_type = _regime_model_type(REGIME_RANGE, "mean_reversion")
                self._save_model(router, model_type, "global", "global", "global_fallback")
                self._save_model(router, model_type, "global", "1h", "timeframe_fallback")
                self._save_model(router, model_type, "EUR/USD", "1h", "specific")

                config = AppConfig()
                config.regime_baseline_cache = True
                config.regime_per_instrument_models = True

                model = _select_regime_baseline(
                    REGIME_RANGE,
                    config=config,
                    model_router=router,
                    symbol="EUR/USD",
                    timeframe="1h",
                )
                self.assertIsNotNone(model)
                self.assertEqual(getattr(model, "name", ""), "specific")

                # Remove specific, expect timeframe fallback
                model = _select_regime_baseline(
                    REGIME_RANGE,
                    config=config,
                    model_router=router,
                    symbol="GBP/USD",
                    timeframe="1h",
                )
                self.assertIsNotNone(model)
                self.assertEqual(getattr(model, "name", ""), "timeframe_fallback")

                # Remove timeframe, expect global fallback
                model = _select_regime_baseline(
                    REGIME_RANGE,
                    config=config,
                    model_router=router,
                    symbol="GBP/USD",
                    timeframe="4h",
                )
                self.assertIsNotNone(model)
                self.assertEqual(getattr(model, "name", ""), "global_fallback")


if __name__ == "__main__":
    unittest.main()
