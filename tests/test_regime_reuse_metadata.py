import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from alphalens_forecast.config import AppConfig
from alphalens_forecast.forecasting import ForecastEngine, compute_dataframe_hash
from alphalens_forecast.models import ModelRouter
from alphalens_forecast.regime_detection.deterministic import RegimeResult, REGIME_TREND_UP
from alphalens_forecast.utils.model_store import ModelStore
from alphalens_forecast.utils.text import slugify


class _DummyProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        del symbol, timeframe
        return self._frame.copy()


class TestRegimeReuseMetadata(unittest.TestCase):
    def _make_frame(self) -> pd.DataFrame:
        idx = pd.date_range("2025-01-01", periods=300, freq="1h", tz="UTC")
        close = 100.0 + np.arange(len(idx)) * 0.5
        data = {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.ones_like(close),
        }
        return pd.DataFrame(data, index=idx)

    def test_reuse_updates_regime_metadata(self) -> None:
        frame = self._make_frame()
        data_hash = compute_dataframe_hash(frame)
        payload = {
            "symbol": "EUR/USD",
            "asOf": frame.index[-1].isoformat(),
            "timeframe": "1h",
            "use_montecarlo": False,
            "horizons": [
                {
                    "h": "24h",
                    "direction": "long",
                    "entry_price": float(frame["close"].iloc[-1]),
                    "tp": float(frame["close"].iloc[-1] * 1.01),
                    "sl": float(frame["close"].iloc[-1] * 0.99),
                    "position_size": 1.0,
                }
            ],
        }
        metadata = {"data_hash": data_hash, "timestamp_slug": "test"}

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            symbol_slug = slugify("EUR/USD")
            timeframe_slug = slugify("1h")
            manifest_path = base_dir / f"{symbol_slug}_{timeframe_slug}_test.json"
            manifest_path.write_text(json.dumps({"metadata": metadata, "payload": payload}), encoding="utf-8")

            logger = logging.getLogger("test")
            model_store = ModelStore(base_dir, logger)
            provider = _DummyProvider(frame)
            router = ModelRouter(base_dir)
            engine = ForecastEngine(AppConfig(), provider, router)

            forced_regime = RegimeResult(
                regime=REGIME_TREND_UP,
                confidence=0.9,
                features={},
                scores={},
                meta={},
            )

            with mock.patch(
                "alphalens_forecast.forecasting.DeterministicRegimeDetector.detect",
                return_value=forced_regime,
            ):
                result = engine.forecast(
                    symbol="EUR/USD",
                    timeframe="1h",
                    horizons=[24],
                    paths=1,
                    use_montecarlo=False,
                    reuse_cached=True,
                    model_store=model_store,
                    show_progress=False,
                    enable_regime_switching=True,
                )

            self.assertTrue(result.used_cached_artifacts)
            self.assertIn("regime", result.metadata)
            self.assertEqual(result.metadata["regime"].get("label"), REGIME_TREND_UP)


if __name__ == "__main__":
    unittest.main()
