"""NeuralProphet forecaster wrapper."""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas._libs.tslibs.timestamps import _unpickle_timestamp
from pandas._libs.tslibs.timedeltas import _timedelta_unpickle
from numpy import dtype as np_dtype
from numpy.core.multiarray import scalar as np_scalar
import torch
from neuralprophet import NeuralProphet, save as save_neuralprophet
from neuralprophet.configure import Normalization
from neuralprophet.df_utils import ShiftScale
from pandas.tseries.frequencies import to_offset

from alphalens_forecast.core.feature_engineering import to_neural_prophet_frame
from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.dataloader_audit import log_dataloader_audit


_SAFE_GLOBALS_REGISTERED = False
logger = logging.getLogger(__name__)


def _ensure_neuralprophet_safe_globals() -> None:
    """Allowlist NeuralProphet classes required for torch.load."""
    global _SAFE_GLOBALS_REGISTERED
    if _SAFE_GLOBALS_REGISTERED:
        return
    try:
        torch.serialization.add_safe_globals([
            NeuralProphet,
            Normalization,
            ShiftScale,
            _unpickle_timestamp,
            _timedelta_unpickle,
            np_scalar,
            np_dtype,
        ])
        _SAFE_GLOBALS_REGISTERED = True
    except AttributeError:
        pass


class NeuralProphetForecaster(BaseForecaster):
    """Wrapper around NeuralProphet configured for intraday forecasting."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(name="NeuralProphet", device=device)
        self._model: Optional[NeuralProphet] = None
        self._train_frame: Optional[pd.DataFrame] = None
        self._freq: Optional[str] = None
        self._progress: str | None = None
        self._trainer_config = None
        self._batch_size = 16
        if self.device.lower().startswith("cuda"):
            # Centralized device handling: let Lightning manage GPU placement.
            self._trainer_config = {"accelerator": "gpu", "devices": 1}

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Restore pickled forecasters while maintaining backwards compatibility.

        Older checkpoints (before _progress existed) won't include the attribute,
        so we provision the default used by current versions during load.
        """
        self.__dict__.update(state)
        if "device" not in state:
            self.device = "cpu"
        if "_progress" not in state:
            self._progress = None
        if "_batch_size" not in state:
            self._batch_size = 16
        if "_trainer_config" not in state:
            self._trainer_config = None
            if getattr(self, "device", "cpu").lower().startswith("cuda"):
                # Centralized device handling for restored checkpoints.
                self._trainer_config = {"accelerator": "gpu", "devices": 1}

    def _progress_display(self) -> str | None:
        """
        Return the progress bar setting, repairing older checkpoints on the fly.

        Some persisted models predate ``_progress``; resolving it lazily ensures we
        can still load and run inference without re-saving them immediately.
        """
        value = getattr(self, "_progress", None)
        self._progress = value
        return value

    def fit(
        self,
        target: pd.Series,
        regressors: Optional[pd.DataFrame] = None,
    ) -> None:
        frame = to_neural_prophet_frame(target, regressors)
        freq = pd.infer_freq(pd.DatetimeIndex(frame["ds"]))
        if freq is None:
            deltas = frame["ds"].diff().dropna()
            if deltas.empty:
                raise ValueError("Unable to infer data frequency for NeuralProphet.")
            freq = to_offset(deltas.mode().iloc[0]).freqstr
        self._freq = freq

        kwargs = {}
        if self._trainer_config is not None:
            kwargs["trainer_config"] = self._trainer_config
        model = NeuralProphet(
            n_lags=30,
            n_changepoints=20,
            n_forecasts=96,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            learning_rate=0.001,
            epochs=20,
            batch_size=self._batch_size,
            **kwargs,
        )

        self._model = model
        self._train_frame = frame
        num_workers = 0
        config = getattr(self, "_dataloader_config", None)
        if config is not None:
            num_workers = int(getattr(config, "num_workers", 0))
        if num_workers > 0:
            logger.info("Using DataLoader workers: num_workers=%s", num_workers)
            self._model.fit(
                frame,
                freq=freq,
                progress=self._progress_display(),
                num_workers=num_workers,
            )
        else:
            self._model.fit(frame, freq=freq, progress=self._progress_display())
        log_dataloader_audit(
            model_name=self.name,
            device=self.device,
            batch_size=self._batch_size,
            model=self._model,
            source_hint="NeuralProphet internal",
        )

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self._model is None or self._train_frame is None:
            raise RuntimeError("NeuralProphetForecaster must be fitted first.")
        if future_regressors is not None and not future_regressors.empty:
            raise ValueError("NeuralProphet forecasts no longer accept future regressors.")

        future = self._model.make_future_dataframe(
            self._train_frame,
            periods=steps,
            n_historic_predictions=False,
        )
        self._ensure_trainer_ready(freq or self._freq)

        forecast = self._model.predict(future)
        return forecast[["ds", "yhat1"]].rename(columns={"yhat1": "yhat"})

    def _ensure_trainer_ready(self, freq: Optional[str]) -> None:
        """Ensure NeuralProphet carries a valid Lightning trainer before predict()."""
        if self._model is None or self._train_frame is None:
            raise RuntimeError("NeuralProphetForecaster must be fitted first.")
        trainer = getattr(self._model, "trainer", None)
        if trainer is not None and hasattr(trainer, "_accelerator_connector"):
            return
        original_epochs = getattr(self._model.config_train, "epochs", None)
        was_fitted = getattr(self._model, "fitted", False)
        original_changepoints = getattr(self._model.config_trend, "changepoints", None)
        try:
            if hasattr(self._model.config_train, "epochs"):
                self._model.config_train.epochs = 0
            self._model.trainer = None
            if was_fitted:
                self._model.fitted = False
            if original_changepoints is not None:
                self._model.config_trend.changepoints = None
            self._model.fit(
                self._train_frame,
                freq=freq or self._freq,
                progress=self._progress_display(),
            )
        finally:
            if original_changepoints is not None:
                self._model.config_trend.changepoints = original_changepoints
            if was_fitted:
                self._model.fitted = True
            if original_epochs is not None:
                self._model.config_train.epochs = original_epochs

    def save(self, path: Path) -> Path:
        """Persist underlying NeuralProphet checkpoint plus training frame."""
        if self._model is None or self._train_frame is None or self._freq is None:
            raise RuntimeError("Cannot save NeuralProphetForecaster before fitting.")
        base = Path(path)
        prefix = base if not base.suffix else base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        model_file = prefix.with_suffix(".np")
        frame_file = prefix.with_suffix(".train.json")
        meta_file = prefix.with_suffix(".meta.json")
        save_neuralprophet(self._model, str(model_file))
        self._train_frame.to_json(frame_file, orient="split", date_format="iso")
        meta = {
            "freq": self._freq,
            "progress": self._progress,
        }
        meta_file.write_text(json.dumps(meta))
        return model_file

    @classmethod
    def load_native(cls, path: Path) -> "NeuralProphetForecaster":
        base = Path(path)
        prefix = base if not base.suffix else base.with_suffix("")
        model_file = prefix.with_suffix(".np")
        frame_file = prefix.with_suffix(".train.json")
        meta_file = prefix.with_suffix(".meta.json")
        instance = cls()
        if not model_file.exists():
            raise FileNotFoundError(f"NeuralProphet checkpoint missing at {model_file}")
        _ensure_neuralprophet_safe_globals()
        instance._model = _load_checkpoint(str(model_file))
        if frame_file.exists():
            frame = pd.read_json(frame_file, orient="split")
            frame["ds"] = pd.to_datetime(frame["ds"])
            instance._train_frame = frame
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            instance._freq = meta.get("freq")
            instance._progress = meta.get("progress", instance._progress)
        return instance
def _load_checkpoint(path: Path) -> NeuralProphet:
    model = torch.load(path, map_location=None, weights_only=False)
    model.restore_trainer(accelerator=None)
    return model
