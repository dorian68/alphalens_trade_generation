"""NHITS forecaster wrapper using Darts."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models import NHiTSModel
from torch.nn import SmoothL1Loss
from torch.optim.adam import Adam
from torchmetrics.collections import MetricCollection

from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.dataloader_audit import log_dataloader_audit
from alphalens_forecast.utils.scaling import ScalerWrapper
from alphalens_forecast.utils.timeseries import (
    build_timeseries,
    series_to_dataframe,
    timeseries_to_dataframe,
)

logger = logging.getLogger(__name__)
_SAFE_GLOBALS_CONFIGURED = False


def _ensure_torch_safe_globals() -> None:
    """Allowlist optimizer classes so torch.load succeeds on PyTorch >=2.6."""
    global _SAFE_GLOBALS_CONFIGURED
    if _SAFE_GLOBALS_CONFIGURED:
        return
    try:
        torch.serialization.add_safe_globals([Adam, SmoothL1Loss, MetricCollection])
    except AttributeError:
        # Older torch versions do not expose add_safe_globals.
        return
    _SAFE_GLOBALS_CONFIGURED = True


class NHiTSForecaster(BaseForecaster):
    """Wrapper around Darts N-HiTS model for high-frequency data."""

    MODEL_VERSION = 3

    def __init__(
        self,
        input_chunk_length: int = 48,
        output_chunk_length: int = 24,
        device: str = "cpu",
    ) -> None:
        super().__init__(name="NHITS", device=device)
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self._n_epochs = 100
        self._batch_size = 10000
        self._dropout = 0.1
        self._random_state = 42
        self._learning_rate = 1e-3
        self._model = self._build_backend()
        self._series: Optional[TimeSeries] = None
        self._scaled_series: Optional[TimeSeries] = None
        self._scaler: ScalerWrapper = ScalerWrapper()
        self._checkpoint_path: Optional[Path] = None
        self._scaler_path: Optional[Path] = None
        self._schema_version: int = self.MODEL_VERSION

    def _build_backend(self) -> NHiTSModel:
        """Instantiate the underlying Darts model with stored hyperparameters."""
        pl_trainer_kwargs = None
        if self.device.lower().startswith("cuda"):
            # Centralized device handling: let Lightning manage GPU placement.
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": 1}
        kwargs = {}
        if pl_trainer_kwargs is not None:
            kwargs["pl_trainer_kwargs"] = pl_trainer_kwargs
        return NHiTSModel(
            input_chunk_length=self._input_chunk_length,
            output_chunk_length=self._output_chunk_length,
            n_epochs=self._n_epochs,
            batch_size=self._batch_size,
            dropout=self._dropout,
            random_state=self._random_state,
            loss_fn=SmoothL1Loss(),
            optimizer_kwargs={"lr": self._learning_rate},
            **kwargs,
        )

    def set_device(self, device: str) -> None:
        super().set_device(device)
        if self._model is None:
            return
        if self.device.lower().startswith("cuda"):
            return
        try:
            self._model.to_cpu()
        except Exception as exc:  # noqa: BLE001
            logger.warning("NHITS failed to move model to CPU: %s", exc)

    def _build_target_series(self, target: pd.Series) -> TimeSeries:
        frame = series_to_dataframe(target)
        return build_timeseries(frame)

    def _dataloader_kwargs(self) -> Optional[dict]:
        config = getattr(self, "_dataloader_config", None)
        if config is None or getattr(config, "num_workers", 0) <= 0:
            return None
        kwargs = {"num_workers": int(config.num_workers)}
        if getattr(config, "pin_memory", False):
            kwargs["pin_memory"] = True
        if getattr(config, "persistent_workers", False):
            kwargs["persistent_workers"] = True
        logger.info(
            "Using DataLoader workers: num_workers=%s, pin_memory=%s, persistent_workers=%s",
            kwargs.get("num_workers"),
            kwargs.get("pin_memory", False),
            kwargs.get("persistent_workers", False),
        )
        return kwargs

    def fit(
        self,
        target: pd.Series,
        regressors: Optional[pd.DataFrame] = None,
    ) -> None:
        if regressors is not None and not regressors.empty:
            raise ValueError("NHITS is now univariate and does not accept regressors.")
        series = self._build_target_series(target)
        raw_values = series.univariate_values()
        logger.info(
            "NHITS training | n_obs=%d | close_range=(%.6f, %.6f)",
            len(raw_values),
            float(np.min(raw_values)),
            float(np.max(raw_values)),
        )
        self._scaler.fit_from_series(series)
        logger.info("NHITS scaler fitted | mean=%.6f std=%.6f", self._scaler.mean_, self._scaler.std_)
        scaled_series = self._scaler.transform(series).astype(np.float32)
        dataloader_kwargs = self._dataloader_kwargs()
        if dataloader_kwargs:
            try:
                self._model.fit(scaled_series, dataloader_kwargs=dataloader_kwargs)
            except TypeError:
                logger.warning("NHITS dataloader_kwargs unsupported; falling back to defaults.")
                self._model.fit(scaled_series)
        else:
            self._model.fit(scaled_series)
        self._series = series.astype(np.float32)
        self._scaled_series = scaled_series
        self._schema_version = self.MODEL_VERSION
        log_dataloader_audit(
            model_name=self.name,
            device=self.device,
            batch_size=self._batch_size,
            model=self._model,
            source_hint="Darts internal",
        )

    def requires_retraining(self) -> bool:
        """Return True if this checkpoint predates the univariate upgrade."""
        schema_version = getattr(self, "_schema_version", 1)
        if schema_version < self.MODEL_VERSION:
            return True
        covariates = getattr(self, "_covariates", None)
        return covariates not in (None, False)

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self._series is None:
            raise RuntimeError("NHiTSForecaster must be fitted before forecasting.")
        if future_regressors is not None and not future_regressors.empty:
            raise ValueError("NHITS forecasts no longer accept future regressors.")

        scaled_series = self._scaled_series
        if scaled_series is None:
            scaled_series = self._scaler.transform(self._series)
            self._scaled_series = scaled_series

        scaled_forecast = self._model.predict(
            n=steps,
            series=scaled_series,
        )
        forecast = self._scaler.inverse_transform(scaled_forecast)
        result = timeseries_to_dataframe(forecast, value_column="yhat")
        if result.empty:
            raise RuntimeError("NHITS forecast produced an empty dataframe.")
        yhat = result["yhat"].to_numpy(dtype=float)
        if not np.isfinite(yhat).all():
            logger.error("Divergent NHITS prediction detected (non-finite values).")
            raise RuntimeError("NHITS prediction diverged (non-finite values).")
        result = result.rename(columns={"datetime": "ds"})
        return result

    # ------------------------------------------------------------------ #
    # Persistence helpers so ModelRouter can reload NHITS checkpoints.

    def save_artifacts(self, model_dir: Path) -> None:
        """Persist underlying Torch checkpoint using Darts-native API."""
        if self._model is None:
            raise RuntimeError("Cannot save NHITS artifacts before fitting the model.")
        checkpoint_path = Path(model_dir) / "nhits_model.pt"
        checkpoint_ckpt = checkpoint_path.with_suffix(checkpoint_path.suffix + ".ckpt")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if checkpoint_ckpt.exists():
            checkpoint_ckpt.unlink()
        self._model.save(str(checkpoint_path))
        self._checkpoint_path = checkpoint_path
        self._save_scaler(model_dir)

    def load_artifacts(self, model_dir: Path) -> None:
        """Reload Torch checkpoint after the wrapper is unpickled."""
        checkpoint_path = Path(model_dir) / "nhits_model.pt"
        if checkpoint_path.exists():
            try:
                _ensure_torch_safe_globals()
                load_kwargs = {}
                if not self.device.lower().startswith("cuda"):
                    load_kwargs["pl_trainer_kwargs"] = {"accelerator": "cpu"}
                    load_kwargs["map_location"] = "cpu"
                self._model = NHiTSModel.load(str(checkpoint_path), **load_kwargs)
                self._checkpoint_path = checkpoint_path
                if not self.device.lower().startswith("cuda"):
                    try:
                        self._model.to_cpu()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("NHITS failed to move checkpoint to CPU: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NHITS checkpoint load failed at %s; fallback to embedded state: %s",
                    checkpoint_path,
                    exc,
                )
                if self._model is None:
                    self._model = self._build_backend()
        else:
            logger.warning(
                "NHITS checkpoint missing at %s; falling back to pickled model state.",
                checkpoint_path,
            )
        scaler_path = self._get_scaler_path(model_dir)
        if scaler_path.exists():
            self._scaler = ScalerWrapper.load(scaler_path)
            self._scaler_path = scaler_path
            logger.info(
                "Loaded NHITS scaler | mean=%.6f std=%.6f",
                self._scaler.mean_,
                self._scaler.std_,
            )
            self._refresh_scaled_series()
        else:
            logger.warning("NHITS scaler missing at %s; predictions may be inconsistent.", scaler_path)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "device" not in self.__dict__:
            self.device = "cpu"
        defaults = {
            "_input_chunk_length": 48,
            "_output_chunk_length": 24,
            "_n_epochs": 300,
            "_batch_size": 32,
            "_dropout": 0.1,
            "_random_state": 42,
            "_learning_rate": 1e-3,
        }
        for attr, value in defaults.items():
            if attr not in self.__dict__:
                setattr(self, attr, value)
        if "_model" not in self.__dict__ or self._model is None:
            self._model = self._build_backend()
        if "_checkpoint_path" not in self.__dict__:
            self._checkpoint_path = None
        if "_schema_version" not in self.__dict__:
            self._schema_version = 1
        if "_scaler" not in self.__dict__ or self._scaler is None:
            self._scaler = ScalerWrapper()
        if "_scaled_series" not in self.__dict__:
            self._scaled_series = None

    def _save_scaler(self, model_dir: Path) -> None:
        if self._scaler and self._scaler.is_fitted:
            scaler_path = self._get_scaler_path(model_dir)
            self._scaler.save(scaler_path)
            self._scaler_path = scaler_path
            logger.info(
                "Persisted NHITS scaler to %s | mean=%.6f std=%.6f",
                scaler_path,
                self._scaler.mean_,
                self._scaler.std_,
            )

    def _get_scaler_path(self, model_dir: Path) -> Path:
        return Path(model_dir) / "nhits_scaler.json"

    def _refresh_scaled_series(self) -> None:
        if self._series is not None and self._scaler.is_fitted:
            self._scaled_series = self._scaler.transform(self._series)
