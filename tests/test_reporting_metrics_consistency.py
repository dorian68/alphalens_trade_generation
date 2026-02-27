import os
import unittest
from contextlib import contextmanager

import pandas as pd

from alphalens_forecast.backtesting import ForecastTrajectory, evaluate_trajectory
from alphalens_forecast.metrics.direction import direction_accuracy_v1_step
from alphalens_forecast.reporting import generate_performance_report


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


class TestReportingMetricsConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self._env = {
            "DIRECTION_ACCURACY_MODE": os.environ.pop("DIRECTION_ACCURACY_MODE", None),
            "REPORTING_USE_SAME_METRICS": os.environ.pop("REPORTING_USE_SAME_METRICS", None),
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

    def test_reporting_matches_backtest_v2(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0])
        predicted = self._make_series([100.0, 99.0, 102.0])
        traj = ForecastTrajectory(
            horizon_label="1h",
            timestamps=list(predicted.index),
            predictions=predicted.to_numpy(dtype=float).tolist(),
        )
        with _set_env("DIRECTION_ACCURACY_MODE", "v2"), _set_env("REPORTING_USE_SAME_METRICS", "1"):
            backtest_metrics = evaluate_trajectory(actual, traj, last_observed=100.0)
            report = generate_performance_report(
                actual=actual,
                predicted=predicted,
                last_observed=100.0,
            )
        self.assertEqual(report.metrics["direction_accuracy"], backtest_metrics["direction_accuracy"])

    def test_reporting_default_v1_even_when_mode_v2(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0])
        predicted = self._make_series([100.0, 99.0, 102.0])
        with _set_env("DIRECTION_ACCURACY_MODE", "v2"):
            report = generate_performance_report(
                actual=actual,
                predicted=predicted,
                last_observed=100.0,
            )
        expected = direction_accuracy_v1_step(actual, predicted)
        self.assertEqual(report.metrics["direction_accuracy"], expected)

    def test_reporting_extended_metrics_default_on(self) -> None:
        actual = self._make_series([100.0, 101.0, 102.0])
        predicted = self._make_series([100.0, 101.0, 102.0])
        report = generate_performance_report(
            actual=actual,
            predicted=predicted,
            last_observed=100.0,
        )
        for key in (
            "direction_accuracy_v1",
            "direction_accuracy_v2",
            "direction_accuracy_v3",
        ):
            self.assertIn(key, report.metrics)


if __name__ == "__main__":
    unittest.main()
