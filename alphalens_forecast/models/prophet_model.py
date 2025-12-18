"""Prophet forecaster wrapper."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from prophet import Prophet

from alphalens_forecast.core.feature_engineering import to_prophet_frame
from alphalens_forecast.models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Wrapper around Facebook Prophet with sensible defaults."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(name="Prophet", device=device)
        self._model: Optional[Prophet] = None
        self._regressor_columns: list[str] = []

    def fit(
        self,
        target: pd.Series,
        regressors: Optional[pd.DataFrame] = None,
    ) -> None:
        frame = to_prophet_frame(target)
        model = Prophet(
            weekly_seasonality=True,
            daily_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        if regressors is not None:
            self._regressor_columns = list(regressors.columns)
            for column in self._regressor_columns:
                model.add_regressor(column)
            frame = frame.merge(
                regressors.reset_index().rename(columns={"index": "ds"}),
                on="ds",
                how="left",
            )
        self._model = model.fit(frame)

    def forecast(
        self,
        steps: int,
        freq: str,
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("ProphetForecaster must be fitted before forecasting.")

        future = self._model.make_future_dataframe(
            periods=steps,
            freq=freq,
            include_history=False,
        )

        if self._regressor_columns:
            if future_regressors is None:
                raise ValueError("Future regressors required for Prophet forecast.")
            missing = set(self._regressor_columns) - set(future_regressors.columns)
            if missing:
                raise ValueError(f"Missing regressors for forecast: {missing}")
            future = future.merge(
                future_regressors.reset_index().rename(columns={"index": "ds"}),
                on="ds",
                how="left",
            )

        forecast = self._model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
