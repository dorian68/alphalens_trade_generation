import unittest

import numpy as np

from alphalens_forecast.regime_detection.hmm_postprocess import (
    REGIME_BREAKOUT,
    REGIME_RANGE,
    REGIME_STRESS_CHOP,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    SmoothConfig,
    smooth_regimes,
)


def _count_switches(labels: np.ndarray) -> int:
    return int(np.sum(labels[1:] != labels[:-1]))


class TestHMMPostprocess(unittest.TestCase):
    def test_micro_switching_reduction(self) -> None:
        labels = np.array(
            [REGIME_TREND_UP if i % 2 == 0 else REGIME_TREND_DOWN for i in range(20)],
            dtype=object,
        )
        smoothed = smooth_regimes(labels, None, SmoothConfig())
        self.assertLess(_count_switches(smoothed), _count_switches(labels))

    def test_breakout_persistence(self) -> None:
        labels = np.array([REGIME_RANGE] * 10, dtype=object)
        features = {"breakout_confirmed": np.array([False, False, True, False, False, False, False, False, False, False])}
        config = SmoothConfig(breakout_hold_bars=3)
        smoothed = smooth_regimes(labels, None, config, features=features)
        self.assertEqual(smoothed[2], REGIME_BREAKOUT)
        self.assertEqual(smoothed[3], REGIME_BREAKOUT)
        self.assertEqual(smoothed[4], REGIME_BREAKOUT)

    def test_stress_chop_gating(self) -> None:
        labels = np.array([REGIME_STRESS_CHOP] * 5, dtype=object)
        features = {
            "slope_norm": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            "stress_eligible": np.array([False, False, False, False, False]),
        }
        smoothed = smooth_regimes(labels, None, SmoothConfig(), features=features)
        self.assertTrue(all(label == REGIME_TREND_UP for label in smoothed))

        features_ok = {
            "slope_norm": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "stress_eligible": np.array([True, True, True, True, True]),
        }
        smoothed_ok = smooth_regimes(labels, None, SmoothConfig(), features=features_ok)
        self.assertTrue(all(label == REGIME_STRESS_CHOP for label in smoothed_ok))

    def test_determinism(self) -> None:
        labels = np.array([REGIME_RANGE, REGIME_TREND_UP, REGIME_RANGE, REGIME_TREND_UP], dtype=object)
        config = SmoothConfig()
        first = smooth_regimes(labels, None, config)
        second = smooth_regimes(labels, None, config)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
