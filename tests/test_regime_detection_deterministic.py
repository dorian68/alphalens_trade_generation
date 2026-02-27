import unittest
from typing import Optional

import numpy as np
import pandas as pd

from alphalens_forecast.regime_detection.deterministic import (
    DeterministicRegimeDetector,
    REGIME_BREAKOUT,
    REGIME_RANGE,
    REGIME_STRESS_CHOP,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
)


class TestDeterministicRegimeDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = DeterministicRegimeDetector()

    def _make_ohlc(
        self,
        close: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None,
        open_: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        close = np.asarray(close, dtype=float)
        if open_ is None:
            open_ = np.concatenate(([close[0]], close[:-1]))
        if high is None:
            high = np.maximum(open_, close) + 0.5
        if low is None:
            low = np.minimum(open_, close) - 0.5
        return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    def _assert_basic(self, result) -> None:
        self.assertTrue(0.0 <= result.confidence <= 1.0)
        for key in ["atr_pct", "slope_norm", "breakout_strength", "chop_ratio", "stress_score"]:
            self.assertIn(key, result.features)
        for key in [REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_BREAKOUT, REGIME_STRESS_CHOP]:
            self.assertIn(key, result.scores)

    def test_trend_up(self) -> None:
        n = 260
        close = 100.0 + np.linspace(0, 20, n) + 0.2 * np.sin(np.linspace(0, 8 * np.pi, n))
        df = self._make_ohlc(close)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_TREND_UP)

    def test_volatile_trend_up_not_stress_chop(self) -> None:
        n = 260
        drift = 2.0
        noise = np.where(np.arange(n) % 2 == 0, 3.0, -3.0)
        close = 100.0 + np.cumsum(drift + noise)
        high = close * 1.03
        low = close * 0.97
        df = self._make_ohlc(close, high=high, low=low)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_TREND_UP)

    def test_trend_down(self) -> None:
        n = 260
        close = 120.0 - np.linspace(0, 20, n) + 0.2 * np.sin(np.linspace(0, 8 * np.pi, n))
        df = self._make_ohlc(close)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_TREND_DOWN)

    def test_range(self) -> None:
        n = 260
        close = 100.0 + 1.2 * np.sin(np.linspace(0, 12 * np.pi, n))
        df = self._make_ohlc(close)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_RANGE)

    def test_breakout_vol_expansion(self) -> None:
        n = 260
        base = 100.0 + 0.2 * np.sin(np.linspace(0, 6 * np.pi, n))
        breakout = np.concatenate([np.zeros(n - 20), np.linspace(0, 15, 20)])
        close = base + breakout
        high = close + np.concatenate([np.full(n - 20, 0.6), np.full(20, 2.5)])
        low = close - np.concatenate([np.full(n - 20, 0.6), np.full(20, 2.5)])
        df = self._make_ohlc(close, high=high, low=low)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_BREAKOUT)

    def test_breakout_overrides_stress_chop(self) -> None:
        n = 260
        returns = np.where(np.arange(n) % 2 == 0, 1.5, -1.5)
        close = 100.0 + np.cumsum(returns)
        ramp = np.concatenate([np.zeros(n - 12), np.linspace(0, 20, 12)])
        close = close + ramp
        ranges = np.concatenate([np.full(n - 12, 1.0), np.full(12, 4.0)])
        high = close + ranges
        low = close - ranges
        df = self._make_ohlc(close, high=high, low=low)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_BREAKOUT)

    def test_stress_chop(self) -> None:
        n = 260
        returns = np.where(np.arange(n) % 2 == 0, 2.5, -2.5)
        close = 100.0 + np.cumsum(returns)
        high = close + 3.0
        low = close - 3.0
        df = self._make_ohlc(close, high=high, low=low)
        result = self.detector.detect(df)
        self._assert_basic(result)
        self.assertEqual(result.regime, REGIME_STRESS_CHOP)

    def test_detect_from_twelve_data_uses_client(self) -> None:
        n = 260
        close = 100.0 + np.linspace(0, 10, n)
        df = self._make_ohlc(close)

        class DummyClient:
            def __init__(self, frame: pd.DataFrame) -> None:
                self.frame = frame
                self.calls = []

            def fetch_ohlcv(self, symbol=None, interval=None, output_size=None, end_time=None):
                self.calls.append((symbol, interval, output_size, end_time))
                return self.frame

        dummy = DummyClient(df)
        result = self.detector.detect_from_twelve_data(
            symbol="BTC/USD",
            interval="1h",
            output_size=200,
            client=dummy,
        )
        self._assert_basic(result)
        self.assertEqual(dummy.calls[0][0], "BTC/USD")
        self.assertEqual(dummy.calls[0][1], "1h")

    def test_detect_from_data_provider_uses_provider(self) -> None:
        n = 260
        close = 100.0 + np.linspace(0, 8, n)
        df = self._make_ohlc(close)

        class DummyProvider:
            def __init__(self, frame: pd.DataFrame) -> None:
                self.frame = frame
                self.calls = []

            def load_latest(self, symbol, timeframe, persist, max_points=None, end_time=None):
                self.calls.append((symbol, timeframe, persist, max_points, end_time))
                return self.frame

        dummy = DummyProvider(df)
        result = self.detector.detect_from_data_provider(
            symbol="ETH/USD",
            interval="15min",
            max_points=200,
            provider=dummy,
            persist_cache=False,
        )
        self._assert_basic(result)
        self.assertEqual(dummy.calls[0][0], "ETH/USD")
        self.assertEqual(dummy.calls[0][1], "15min")
        self.assertFalse(dummy.calls[0][2])


if __name__ == "__main__":
    unittest.main()
