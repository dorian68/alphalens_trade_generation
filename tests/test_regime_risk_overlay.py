import unittest

from alphalens_forecast.trading.overlays.regime_risk_overlay import OverlayConfig, RegimeRiskOverlay


class TestRegimeRiskOverlay(unittest.TestCase):
    def setUp(self) -> None:
        self.overlay = RegimeRiskOverlay(config=OverlayConfig())

    def _payload(self, direction: str = "long", position_size: float = 1.0, confidence: float = 0.7):
        return {
            "symbol": "BTC/USD",
            "asOf": "2025-01-01T00:00:00Z",
            "timeframe": "1h",
            "use_montecarlo": False,
            "horizons": [
                {
                    "h": "1h",
                    "direction": direction,
                    "entry_price": 100.0,
                    "tp": 110.0,
                    "sl": 90.0,
                    "confidence": confidence,
                    "position_size": position_size,
                }
            ],
        }

    def test_noop_when_regime_disabled(self) -> None:
        payload = self._payload()
        context = {"regime_enabled": False, "regime_label": "STRESS_CHOP"}
        result = self.overlay.apply(payload, context)
        self.assertEqual(result["horizons"][0]["position_size"], 1.0)

    def test_stress_chop_blocks_trade(self) -> None:
        payload = self._payload()
        context = {
            "regime_enabled": True,
            "regime_label": "STRESS_CHOP",
            "regime_confidence": 0.4,
        }
        result = self.overlay.apply(payload, context)
        self.assertEqual(result["horizons"][0]["position_size"], 0.0)

    def test_stress_chop_allows_breakout(self) -> None:
        payload = self._payload()
        context = {
            "regime_enabled": True,
            "regime_label": "STRESS_CHOP",
            "regime_route": "breakout",
            "regime_confidence": 0.4,
        }
        result = self.overlay.apply(payload, context)
        self.assertAlmostEqual(result["horizons"][0]["position_size"], 0.25)

    def test_trend_down_blocks_long(self) -> None:
        payload = self._payload(direction="long")
        context = {"regime_enabled": True, "regime_label": "TREND_DOWN"}
        result = self.overlay.apply(payload, context)
        self.assertEqual(result["horizons"][0]["position_size"], 0.0)

    def test_trend_up_blocks_short(self) -> None:
        payload = self._payload(direction="short")
        context = {"regime_enabled": True, "regime_label": "TREND_UP"}
        result = self.overlay.apply(payload, context)
        self.assertEqual(result["horizons"][0]["position_size"], 0.0)

    def test_range_scales_down(self) -> None:
        payload = self._payload()
        context = {
            "regime_enabled": True,
            "regime_label": "RANGE",
            "regime_confidence": 0.3,
            "entry_model_vol_by_horizon": {"1h": 2.0},
            "vol_ref": 1.0,
        }
        result = self.overlay.apply(payload, context)
        self.assertAlmostEqual(result["horizons"][0]["position_size"], 0.3)

    def test_performance_patches_block_low_confidence(self) -> None:
        payload = self._payload(confidence=0.5)
        context = {
            "regime_enabled": True,
            "regime_label": "RANGE",
            "regime_confidence": 0.9,
            "performance_patches_enabled": True,
        }
        result = self.overlay.apply(payload, context)
        self.assertEqual(result["horizons"][0]["position_size"], 0.0)

    def test_unknown_long_blocked_with_performance_patch(self) -> None:
        payload = self._payload(direction="long")
        context = {
            "regime_enabled": True,
            "regime_label": None,
            "performance_patches_enabled": True,
        }
        result = self.overlay.apply(payload, context)
        self.assertEqual(result["horizons"][0]["position_size"], 0.0)

    def test_schema_unchanged(self) -> None:
        payload = self._payload()
        context = {"regime_enabled": True, "regime_label": "TREND_DOWN"}
        result = self.overlay.apply(payload, context)
        self.assertEqual(set(payload.keys()), set(result.keys()))
        self.assertEqual(set(payload["horizons"][0].keys()), set(result["horizons"][0].keys()))


if __name__ == "__main__":
    unittest.main()
