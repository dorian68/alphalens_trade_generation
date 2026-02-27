import unittest

import numpy as np
import pandas as pd

from alphalens_forecast.metrics.direction import (
    brier_score_direction,
    coverage_from_mc,
    direction_accuracy_v1_step,
    direction_accuracy_v2_anchor,
    direction_accuracy_v3_deadzone,
)


class TestDirectionMetrics(unittest.TestCase):
    def test_direction_accuracy_v1_step(self) -> None:
        actual = np.array([100.0, 101.0, 100.0, 101.0], dtype=float)
        pred = np.array([100.0, 101.0, 102.0, 101.0], dtype=float)
        expected = 1 / 3
        result = direction_accuracy_v1_step(
            actual=pd.Series(actual),
            pred=pd.Series(pred),
        )
        self.assertAlmostEqual(result, expected)

    def test_direction_accuracy_v2_anchor(self) -> None:
        result = direction_accuracy_v2_anchor(100.0, 102.0, 101.0)
        self.assertEqual(result, 1.0)

    def test_direction_accuracy_v3_deadzone(self) -> None:
        below = direction_accuracy_v3_deadzone(
            last_observed=100.0,
            actual_end=102.0,
            pred_end=103.0,
            deadzone_abs=5.0,
        )
        self.assertTrue(np.isnan(below))
        above = direction_accuracy_v3_deadzone(
            last_observed=100.0,
            actual_end=106.0,
            pred_end=105.0,
            deadzone_abs=5.0,
        )
        self.assertEqual(above, 1.0)

    def test_coverage_from_mc(self) -> None:
        samples = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        inside = coverage_from_mc(2.0, samples, alpha=0.5)
        outside = coverage_from_mc(4.0, samples, alpha=0.5)
        self.assertEqual(inside, 1.0)
        self.assertEqual(outside, 0.0)

    def test_brier_score_direction(self) -> None:
        self.assertAlmostEqual(brier_score_direction(0.7, True), 0.09)


if __name__ == "__main__":
    unittest.main()
