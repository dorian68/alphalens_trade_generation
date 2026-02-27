import unittest

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from alphalens_forecast.regime_detection import deterministic as det
from alphalens_forecast.regime_detection.hmm_store import HMMRegimeStore


class TestRegimeDetectorOptions(unittest.TestCase):
    def test_score_smoothing_changes_values(self) -> None:
        scores = pd.DataFrame(
            {
                "TREND_UP": [0.0, 1.0, 0.0, 1.0],
                "TREND_DOWN": [1.0, 0.0, 1.0, 0.0],
            },
            index=pd.RangeIndex(4),
        )
        smoothed = det._smooth_scores(scores, 0.2)
        self.assertFalse(np.allclose(scores["TREND_UP"].values, smoothed["TREND_UP"].values))

    def test_vol_mom_confirm_penalizes_breakout(self) -> None:
        features = pd.DataFrame(
            {
                "slope_norm": [0.0],
                "chop_ratio": [2.0],
                "atr_pct": [0.02],
                "atr_pct_low": [0.01],
                "atr_pct_high": [0.02],
                "atr_pct_extreme": [0.03],
                "breakout_strength": [2.0],
                "breakout_flag": [1.0],
                "vol_expansion": [2.0],
                "sign_flip_rate": [0.1],
                "tail_score": [0.2],
                "candle_score": [0.2],
                "momentum_strength": [0.0],
                "volume_ratio": [0.5],
                "volume_z": [0.0],
            },
            index=pd.RangeIndex(1),
        )
        config_off = det.RegimeConfig(enable_vol_mom_confirm=False)
        config_on = det.RegimeConfig(enable_vol_mom_confirm=True)
        score_off = det._compute_scores(features, config_off)[det.REGIME_BREAKOUT].iloc[0]
        score_on = det._compute_scores(features, config_on)[det.REGIME_BREAKOUT].iloc[0]
        self.assertLess(score_on, score_off)

    def test_hmm_mode_falls_back_or_runs(self) -> None:
        idx = pd.date_range("2025-01-01", periods=80, freq="1h", tz="UTC")
        close = 100.0 + np.sin(np.linspace(0, 6.28, len(idx))) * 2.0 + np.arange(len(idx)) * 0.05
        frame = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": np.linspace(1000.0, 1100.0, len(idx)),
            },
            index=idx,
        )
        detector = det.DeterministicRegimeDetector(det.RegimeConfig(regime_mode="hmm"))
        result = detector.detect(frame)
        self.assertIn(result.regime, det.REGIME_ORDER)

        series = detector.detect_series(frame)
        self.assertIn("regime", series.columns)
        self.assertIn("confidence", series.columns)
        for col in series.columns:
            if col in {"regime", "confidence"}:
                continue
            self.assertTrue(col.startswith("score_") or col.startswith("feat_"))

    def test_fit_and_apply_hmm_model(self) -> None:
        try:
            import hmmlearn  # noqa: F401
        except Exception:
            self.skipTest("hmmlearn not available")

        idx = pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC")
        close = 100.0 + np.sin(np.linspace(0, 12.56, len(idx))) * 1.5 + np.arange(len(idx)) * 0.02
        frame = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": np.linspace(1000.0, 1100.0, len(idx)),
            },
            index=idx,
        )
        model = det.fit_hmm_regime_model(frame, config=det.RegimeConfig())
        applied = det.apply_hmm_regime_model(frame, model)
        self.assertFalse(applied.empty)
        self.assertIn("regime", applied.columns)
        self.assertIn("confidence", applied.columns)
        for regime in det.REGIME_ORDER:
            self.assertIn(f"score_{regime}", applied.columns)
        self.assertTrue((applied["confidence"] >= 0.0).all())
        self.assertTrue((applied["confidence"] <= 1.0).all())

    def test_hmm_store_roundtrip(self) -> None:
        try:
            import hmmlearn  # noqa: F401
        except Exception:
            self.skipTest("hmmlearn not available")

        idx = pd.date_range("2025-01-01", periods=120, freq="1h", tz="UTC")
        close = 100.0 + np.sin(np.linspace(0, 12.56, len(idx))) * 1.5 + np.arange(len(idx)) * 0.02
        frame = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": np.linspace(1000.0, 1100.0, len(idx)),
            },
            index=idx,
        )
        model = det.fit_hmm_regime_model(frame, config=det.RegimeConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = HMMRegimeStore(base_dir=Path(tmp_dir))
            store.save("BTC/USD", "1h", model)
            loaded = store.load("BTC/USD", "1h")
            self.assertIsNotNone(loaded)
            applied = det.apply_hmm_regime_model(frame, loaded)
            self.assertFalse(applied.empty)


if __name__ == "__main__":
    unittest.main()
