"""Temporal Fusion Transformer forecaster using Darts."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel

from alphalens_forecast.models.base import BaseForecaster
from alphalens_forecast.models.dataloader_audit import log_dataloader_audit


class TFTForecaster(BaseForecaster):
    """Wrapper around Darts Temporal Fusion Transformer for medium horizons."""

    def __init__(
        self,
        input_chunk_length: int = 64,
        output_chunk_length: int = 16,
        device: str = "cpu",
    ) -> None:
        super().__init__(name="TFT", device=device)
        self._batch_size = 64
        pl_trainer_kwargs = None
        if self.device.lower().startswith("cuda"):
            # Centralized device handling: let Lightning manage GPU placement.
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": 1}
        kwargs = {}
        if pl_trainer_kwargs is not None:
            kwargs["pl_trainer_kwargs"] = pl_trainer_kwargs
        self._model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            random_state=42,
            hidden_size=32,
            lstm_layers=1,
            dropout=0.1,
            batch_size=self._batch_size,
            n_epochs=200,
            **kwargs,
        )
        self._series: Optional[TimeSeries] = None
        self._past_covariates: Optional[TimeSeries] = None
        self._future_covariates: Optional[TimeSeries] = None

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
        series = TimeSeries.from_series(target)
        covariates = (
            TimeSeries.from_dataframe(regressors, value_cols=list(regressors.columns))
            if regressors is not None
            else None
        )
        dataloader_kwargs = self._dataloader_kwargs()
        if dataloader_kwargs:
            try:
                self._model.fit(
                    series,
                    past_covariates=covariates,
                    future_covariates=covariates,
                    dataloader_kwargs=dataloader_kwargs,
                )
            except TypeError:
                logger.warning("TFT dataloader_kwargs unsupported; falling back to defaults.")
                self._model.fit(
                    series,
                    past_covariates=covariates,
                    future_covariates=covariates,
                )
        else:
            self._model.fit(
                series,
                past_covariates=covariates,
                future_covariates=covariates,
            )
        self._series = series
        self._past_covariates = covariates
        self._future_covariates = covariates
        log_dataloader_audit(
            model_name=self.name,
            device=self.device,
            batch_size=self._batch_size,
            model=self._model,
            source_hint="Darts internal",
        )

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self._series is None:
            raise RuntimeError("TFTForecaster must be fitted before forecasting.")

        future_covariates = self._future_covariates
        if self._future_covariates is not None:
            if future_regressors is None:
                raise ValueError("Future covariates required for TFT forecast.")
            future_covariates = TimeSeries.from_dataframe(
                future_regressors,
                value_cols=list(future_regressors.columns),
            )

        forecast = self._model.predict(
            n=steps,
            past_covariates=self._past_covariates,
            future_covariates=future_covariates,
        )
        result = forecast.pd_dataframe()
        result.columns = ["yhat"]
        result = result.reset_index().rename(columns={"index": "ds"})
        return result
