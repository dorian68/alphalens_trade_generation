import os
import unittest
from contextlib import contextmanager

import numpy as np
import pandas as pd

from alphalens_forecast.backtesting import ForecastTrajectory, evaluate_trajectory
from alphalens_forecast.metrics.direction import (
    direction_accuracy_v1_step,
)


@contextmanager
def _set_env(key: str, value: str | None):
    original = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original


class TestDirectionAccuracy(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {
            "DIRECTION_ACCURACY_V2": os.environ.pop("DIRECTION_ACCURACY_V2", None),
            "DIRECTION_ACCURACY_MODE": os.environ.pop("DIRECTION_ACCURACY_MODE", None),
            "DIRECTION_DEADZONE_ABS": os.environ.pop("DIRECTION_DEADZONE_ABS", None),
            "DIRECTION_DEADZONE_ATR_K": os.environ.pop("DIRECTION_DEADZONE_ATR_K", None),
            "REPORTING_EXTENDED_METRICS": os.environ.pop("REPORTING_EXTENDED_METRICS", None),
        }

    def tearDown(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _make_series(self, values: list[float], start: str = "2024-01-01", freq: str = "1H") -> pd.Series:
        idx = pd.date_range(start=start, periods=len(values), freq=freq)
        return pd.Series(values, index=idx, dtype=float)

    def _make_trajectory(self, series: pd.Series, label: str = "1h") -> ForecastTrajectory:
        return ForecastTrajectory(
            horizon_label=label,
            timestamps=list(series.index),
            predictions=series.to_numpy(dtype=float).tolist(),
        )

    def test_direction_accuracy_step_counts(self) -> None:
        actual = self._make_series([100.0, 101.0, 100.0, 101.0, 100.0])
        predicted = self._make_series([100.0, 101.0, 102.0, 101.0, 100.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)

        expected = direction_accuracy_v1_step(actual, predicted)
        self.assertAlmostEqual(metrics["direction_accuracy"], expected)

    def test_direction_accuracy_uptrend(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0, 103.0])
        predicted = self._make_series([100.0, 101.0, 102.0, 103.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)
        self.assertEqual(metrics["direction_accuracy"], 1.0)

    def test_direction_accuracy_downtrend(self) -> None:
        actual = self._make_series([103.0, 102.0, 101.0, 100.0])
        predicted = self._make_series([103.0, 102.0, 101.0, 100.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)
        self.assertEqual(metrics["direction_accuracy"], 1.0)

    def test_direction_accuracy_flat(self) -> None:
        actual = self._make_series([100.0, 100.0, 100.0])
        predicted = self._make_series([100.0, 100.0, 100.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)
        self.assertEqual(metrics["direction_accuracy"], 1.0)

    def test_direction_accuracy_flat_vs_move(self) -> None:
        actual = self._make_series([100.0, 101.0, 100.0])
        predicted = self._make_series([100.0, 100.0, 100.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)
        self.assertEqual(metrics["direction_accuracy"], 0.0)

    def test_direction_accuracy_nan(self) -> None:
        actual = self._make_series([100.0, float("nan"), 102.0])
        predicted = self._make_series([100.0, 101.0, 102.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)
        self.assertTrue(np.isnan(metrics["direction_accuracy"]))

    def test_direction_accuracy_short_horizon(self) -> None:
        actual = self._make_series([100.0])
        predicted = self._make_series([100.0])
        traj = self._make_trajectory(predicted)
        metrics = evaluate_trajectory(actual, traj)
        self.assertTrue(np.isnan(metrics["direction_accuracy"]))

    def test_direction_accuracy_v2_anchor(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0])
        predicted = self._make_series([100.0, 99.0, 102.0])
        traj = self._make_trajectory(predicted)

        metrics_default = evaluate_trajectory(actual, traj, last_observed=100.0)
        self.assertEqual(metrics_default["direction_accuracy"], 0.5)

        with _set_env("DIRECTION_ACCURACY_MODE", "v2"):
            metrics_v2 = evaluate_trajectory(actual, traj, last_observed=100.0)
        self.assertEqual(metrics_v2["direction_accuracy"], 1.0)

    def test_direction_accuracy_v2_mismatch(self) -> None:
        actual = self._make_series([100.0, 99.0, 98.0])
        predicted = self._make_series([100.0, 101.0, 102.0])
        traj = self._make_trajectory(predicted)
        with _set_env("DIRECTION_ACCURACY_MODE", "v2"):
            metrics = evaluate_trajectory(actual, traj, last_observed=100.0)
        self.assertEqual(metrics["direction_accuracy"], 0.0)

    def test_direction_accuracy_v3_deadzone(self) -> None:
        actual = self._make_series([100.0, 101.0, 101.5])
        predicted = self._make_series([100.0, 101.0, 102.0])
        traj = self._make_trajectory(predicted)
        with _set_env("DIRECTION_ACCURACY_MODE", "v3"), _set_env("DIRECTION_DEADZONE_ABS", "5"):
            metrics = evaluate_trajectory(actual, traj, last_observed=100.0)
        self.assertTrue(np.isnan(metrics["direction_accuracy"]))

    def test_direction_accuracy_legacy_v2_flag(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0])
        predicted = self._make_series([100.0, 99.0, 102.0])
        traj = self._make_trajectory(predicted)
        with _set_env("DIRECTION_ACCURACY_V2", "1"):
            metrics = evaluate_trajectory(actual, traj, last_observed=100.0)
        self.assertEqual(metrics["direction_accuracy"], 1.0)

    def test_extended_metrics_present(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0])
        predicted = self._make_series([100.0, 101.0, 102.0])
        traj = self._make_trajectory(predicted)
        with _set_env("REPORTING_EXTENDED_METRICS", "1"):
            metrics = evaluate_trajectory(actual, traj, last_observed=100.0)
        for key in (
            "direction_accuracy_v1",
            "direction_accuracy_v2",
            "direction_accuracy_v3",
            "conditional_direction_accuracy",
        ):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
