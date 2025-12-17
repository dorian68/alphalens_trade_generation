"""Temporal Fusion Transformer forecaster using Darts."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel

from alphalens_forecast.models.base import BaseForecaster


class TFTForecaster(BaseForecaster):
    """Wrapper around Darts Temporal Fusion Transformer for medium horizons."""

    def __init__(
        self,
        input_chunk_length: int = 64,
        output_chunk_length: int = 16,
    ) -> None:
        super().__init__(name="TFT")
        self._model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            random_state=42,
            hidden_size=32,
            lstm_layers=1,
            dropout=0.1,
            batch_size=64,
            n_epochs=200,
        )
        self._series: Optional[TimeSeries] = None
        self._past_covariates: Optional[TimeSeries] = None
        self._future_covariates: Optional[TimeSeries] = None

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
        self._model.fit(
            series,
            past_covariates=covariates,
            future_covariates=covariates,
        )
        self._series = series
        self._past_covariates = covariates
        self._future_covariates = covariates

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
