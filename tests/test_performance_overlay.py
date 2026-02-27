import unittest

import numpy as np
import pandas as pd

from alphalens_forecast.trading.overlays.performance_overlay import PerformanceOverlay


def _make_price_frame(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )
    return frame


class TestPerformanceOverlay(unittest.TestCase):
    def setUp(self) -> None:
        self.overlay = PerformanceOverlay()

    def _context(self, regime_label: str, price_frame: pd.DataFrame) -> dict:
        return {
            "performance_patches_enabled": True,
            "regime_enabled": True,
            "regime_label": regime_label,
            "regime_confidence": 0.8,
            "regime_route": "alternate",
            "price_frame": price_frame,
        }

    def test_breakout_skipped_without_confirmation(self) -> None:
        frame = _make_price_frame([100] * 30)
        trade = {"direction": "long", "entry_price": 100.0, "position_size": 1.0}
        context = self._context("BREAKOUT_VOL_EXPANSION", frame)
        updated = self.overlay.apply(trade, context)
        self.assertEqual(updated["position_size"], 0.0)

    def test_breakout_allowed_with_confirmation(self) -> None:
        closes = [100] * 25 + [110, 111, 112]
        frame = _make_price_frame(closes)
        # increase volatility on last bars
        frame.loc[frame.index[-3:], "high"] = frame["close"].iloc[-3:] + 5.0
        frame.loc[frame.index[-3:], "low"] = frame["close"].iloc[-3:] - 5.0
        trade = {"direction": "long", "entry_price": 112.0, "position_size": 1.0}
        context = self._context("BREAKOUT_VOL_EXPANSION", frame)
        updated = self.overlay.apply(trade, context)
        self.assertGreater(updated["position_size"], 0.0)

    def test_range_long_skips_mid_range(self) -> None:
        closes = list(np.linspace(100, 110, 25))
        frame = _make_price_frame(closes)
        trade = {"direction": "long", "entry_price": 108.0, "position_size": 1.0}
        context = self._context("RANGE", frame)
        updated = self.overlay.apply(trade, context)
        self.assertEqual(updated["position_size"], 0.0)

    def test_range_long_allows_near_bottom(self) -> None:
        closes = list(np.linspace(100, 110, 25))
        frame = _make_price_frame(closes)
        trade = {"direction": "long", "entry_price": 101.0, "position_size": 1.0}
        context = self._context("RANGE", frame)
        updated = self.overlay.apply(trade, context)
        self.assertGreater(updated["position_size"], 0.0)

    def test_anti_extension_skips_trend(self) -> None:
        closes = [100] * 30
        frame = _make_price_frame(closes)
        trade = {"direction": "long", "entry_price": 105.0, "position_size": 1.0}
        context = self._context("TREND_UP", frame)
        updated = self.overlay.apply(trade, context)
        self.assertEqual(updated["position_size"], 0.0)

    def test_noop_when_disabled(self) -> None:
        frame = _make_price_frame([100] * 30)
        trade = {"direction": "long", "entry_price": 105.0, "position_size": 1.0}
        context = {
            "performance_patches_enabled": False,
            "regime_enabled": True,
            "regime_label": "TREND_UP",
            "price_frame": frame,
        }
        updated = self.overlay.apply(trade, context)
        self.assertEqual(updated["position_size"], 1.0)


if __name__ == "__main__":
    unittest.main()
