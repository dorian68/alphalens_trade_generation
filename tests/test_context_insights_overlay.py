import unittest

import numpy as np
import pandas as pd

from alphalens_forecast.trading.overlays.context_insights_overlay import ContextInsightsOverlay


def _make_price_frame(closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(100, 120, len(close)),
        }
    )
    return frame


class TestContextInsightsOverlay(unittest.TestCase):
    def setUp(self) -> None:
        self.overlay = ContextInsightsOverlay()

    def _context(self, regime_label: str, price_frame: pd.DataFrame) -> dict:
        return {
            "regime_enabled": True,
            "regime_label": regime_label,
            "regime_confidence": 0.8,
            "regime_mode_used": "hmm",
            "price_frame": price_frame,
        }

    def test_range_trade_enrichment(self) -> None:
        frame = _make_price_frame(list(np.linspace(100, 102, 30)))
        trade = {"direction": "long", "entry_price": 101.0, "position_size": 1.0}
        context = self._context("RANGE", frame)
        updated = self.overlay.apply(trade, context)
        self.assertEqual(updated["direction"], "long")
        self.assertEqual(updated["entry_price"], 101.0)
        self.assertEqual(updated["position_size"], 1.0)
        self.assertIn("context_insights", updated)
        self.assertIsInstance(updated["context_insights"], list)

    def test_payload_enrichment(self) -> None:
        frame = _make_price_frame(list(np.linspace(100, 105, 30)))
        payload = {
            "symbol": "BTC/USD",
            "asOf": "2025-01-01T00:00:00Z",
            "timeframe": "1h",
            "use_montecarlo": False,
            "horizons": [
                {
                    "h": "1h",
                    "direction": "long",
                    "entry_price": 103.0,
                    "tp": 108.0,
                    "sl": 98.0,
                    "confidence": 0.7,
                    "position_size": 1.0,
                }
            ],
        }
        context = self._context("BREAKOUT_VOL_EXPANSION", frame)
        updated = self.overlay.apply(payload, context)
        self.assertIn("context_insights", updated)
        self.assertIn("context_insights", updated["horizons"][0])


if __name__ == "__main__":
    unittest.main()
