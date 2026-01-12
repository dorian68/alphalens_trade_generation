# %%
from pathlib import Path
import pandas as pd
path_file = "/home/ubuntu/.cache/kagglehub/datasets/mczielinski/bitcoin-historical-data/versions/434"
# # Download latest version of the BTC dataset and load it.
# path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data")
# print("Path to dataset files:", path)

path = path_file
BTC_DATASET_DIR = Path(path)
df_btc = pd.read_csv(BTC_DATASET_DIR / "btcusd_1-min_data.csv", parse_dates=["Timestamp"])
df_btc["Timestamp"] = pd.to_datetime(df_btc["Timestamp"], unit="s", utc=True)
df_btc = df_btc.set_index("Timestamp").sort_index()

# %%
# %%
# ************************************************************************
# * ALPHALENS FORECAST TESTING SCRIPT                                    *
# *   Copyright (c) 2024 AlphaLens Labs Inc.                             *
# ************************************************************************


import copy
import sys
from pathlib import Path
import importlib

# importlib.reload(ts_utils)
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

# import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
# import alphalens_forecast.utils.timeseries as ts_utils

from alphalens_forecast.backtesting import TrajectoryRecorder, evaluate_trajectory
from alphalens_forecast.config import get_config
from alphalens_forecast.core import horizon_to_steps, prepare_features, prepare_residuals
from alphalens_forecast.forecasting import ForecastEngine, forecast_from_series
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.utils.timeseries import align_series_to_timeframe, series_to_price_frame

df_eurusd = pd.read_csv(
    "/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST/alphalens_forecast/data/cache/EUR_USD/15min.csv",
    parse_dates=["datetime"],
).set_index("datetime").sort_index()



# %%
import copy
import sys
from pathlib import Path

importlib.reload(ts_utils)
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

# import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import alphalens_forecast.utils.timeseries as ts_utils
import importlib


from alphalens_forecast.backtesting import TrajectoryRecorder, evaluate_trajectory
from alphalens_forecast.config import get_config
from alphalens_forecast.core import horizon_to_steps, prepare_features, prepare_residuals
from alphalens_forecast.forecasting import ForecastEngine, forecast_from_series
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.utils.timeseries import align_series_to_timeframe, series_to_price_frame


# Load pre-trained NHITS/EGARCH checkpoints.
PROJECT_ROOT = Path("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")
router = ModelRouter(PROJECT_ROOT / "models")
model = router.load_model("nhits", "BTC/USD", "15min")
vol = router.load_egarch("BTC/USD", "15min")
if model is None or vol is None:
    raise RuntimeError("Pas de checkpoint trouvé")

# Align the close series to the 15min cadence expected by the model.
raw_series = df_btc["Close"].astype(float)
series = align_series_to_timeframe(raw_series, "15min")

# Run the forecast using stored mean/volatility models.
result = forecast_from_series(
    series,
    model=model,
    timeframe="15min",
    horizons=[3, 6, 12, 24],
    symbol="BTC/USD",
    use_montecarlo=True,
    fit_model=False,
    vol_model=vol,
)
print(result.payload)

class _MemoryDataProvider:
    """Serve a pre-loaded price frame to ForecastEngine."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:  # noqa: D401
        del symbol, timeframe
        return self._frame.copy()


def _plot_trajectories_vs_actual(
    actual_future: pd.Series,
    history_context: pd.Series,
    recorder: TrajectoryRecorder,
    title: str,
    *,
    min_context_points: int = 1,
    window_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    """Plot model trajectories against realised prices per horizon."""
    actual_future = actual_future.sort_index()
    context = history_context.sort_index()
    if len(context) < min_context_points and not actual_future.empty:
        context = pd.concat([actual_future.iloc[:1], context])
    for trajectory in recorder.trajectories:
        timestamps = pd.to_datetime(trajectory.timestamps, utc=True)
        predicted = pd.Series(trajectory.predictions, index=timestamps)
        actual_aligned = actual_future.reindex(timestamps)
        overlap = actual_aligned.dropna()
        if overlap.empty:
            # No overlap: prepend last context point to reveal starting anchor.
            actual_aligned = pd.concat([context.tail(1), actual_future]).drop_duplicates()
        plt.figure(figsize=(10, 4))
        plt.plot(context.index, context.values, label="Actual (history)", color="gray", alpha=0.6)
        plt.plot(actual_aligned.index, actual_aligned.values, label="Actual (holdout)", linewidth=2)
        plt.plot(predicted.index, predicted.values, label="Predicted", linestyle="--")
        plt.title(f"{title} – {trajectory.horizon_label}")
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend()
        if window_bounds is not None:
            start_bound, end_bound = window_bounds
            x_min = start_bound if context.empty else min(context.index.min(), start_bound)
            plt.xlim(x_min, end_bound)
        plt.tight_layout()
        plt.show()
        if not overlap.empty:
            metrics = evaluate_trajectory(overlap, trajectory)
            print(f"Holdout metrics for {trajectory.horizon_label}: {metrics}")
        else:
            print(f"Warning: no overlapping timestamps for {trajectory.horizon_label}; metrics unavailable.")


def _refit_models_for_window(price_frame: pd.DataFrame):
    """Return fresh copies of the cached models trained on the provided price frame."""
    mean_model_clone = copy.deepcopy(model)
    features = prepare_features(price_frame)
    mean_model_clone.fit(features.target, features.regressors)
    vol_model_clone = copy.deepcopy(vol)
    residuals = prepare_residuals(price_frame["log_return"])
    vol_model_clone.fit(residuals)
    return mean_model_clone, vol_model_clone


def run_holdout_evaluation(
    aligned_series: pd.Series,
    horizons: list[int],
    timeframe: str,
    *,
    window_end: pd.Timestamp | str | None = None,
    window_hours: int | None = None,
    window_start: pd.Timestamp | str | None = None,
    context_hours: int = 24,
) -> None:
    """
    Hold out a custom window, forecast it, and plot actual vs predicted trajectories.

    Parameters
    ----------
    aligned_series:
        Resampled close series matching the timeframe cadence.
    horizons:
        Horizons (in hours) to evaluate.
    timeframe:
        Timeframe string (e.g., 15min).
    window_end, window_start:
        Optional datetime bounds for the evaluation window.
    window_hours:
        Optional duration (hours) for the holdout if window_start is not provided.
    """
    config = get_config()
    required_steps = horizon_to_steps(max(horizons), timeframe)
    window_end_ts = pd.to_datetime(window_end, utc=True) if window_end is not None else aligned_series.index[-1]
    if window_start is not None:
        window_start_ts = pd.to_datetime(window_start, utc=True)
    else:
        hours = window_hours if window_hours is not None else max(horizons)
        window_start_ts = window_end_ts - pd.to_timedelta(hours, unit="H")
    actual_future = aligned_series.loc[window_start_ts:window_end_ts]
    if len(actual_future) < required_steps:
        raise ValueError("Holdout window is shorter than the largest horizon.")
    cutoff = actual_future.index[0]
    history = aligned_series.loc[:cutoff].iloc[:-1]
    if history.empty:
        raise ValueError("Insufficient history before the holdout window.")
    context_steps = horizon_to_steps(context_hours, timeframe)
    history_context = history.iloc[-context_steps:]
    price_frame = series_to_price_frame(history)
    provider = _MemoryDataProvider(price_frame)
    mean_model_eval, vol_model_eval = _refit_models_for_window(price_frame)
    recorder = TrajectoryRecorder()
    engine = ForecastEngine(config, provider, router)
    engine.forecast(
        symbol="BTC/USD",
        timeframe=timeframe,
        horizons=horizons,
        paths=config.monte_carlo.paths,
        use_montecarlo=config.monte_carlo.use_montecarlo,
        reuse_cached=False,
        model_store=None,
        show_progress=False,
        trajectory_recorder=recorder,
        price_frame=price_frame,
        mean_model_override=mean_model_eval,
        vol_model_override=vol_model_eval,
    )
    _plot_trajectories_vs_actual(
        actual_future,
        history_context,
        recorder,
        title="BTC/USD Holdout Evaluation",
        window_bounds=(window_start_ts, window_end_ts + pd.to_timedelta(max(horizons), unit="H")),
    )


run_holdout_evaluation(
    series,
    horizons=[3, 6, 12, 24],
    timeframe="15min",
    window_start="2024-01-01 00:00:00+00:00",
    window_end="2024-01-03 00:00:00+00:00",
)

# %%

import pandas as pd
from pathlib import Path
from alphalens_forecast.training import train_nhits, train_egarch
from alphalens_forecast.models.router import ModelRouter

symbol = "BTC/USD"
timeframe = "15min"
start = "2021-01-01"
end = "2021-03-31"

df = (
    pd.read_csv("data/btc_history_15min.csv", parse_dates=["datetime"])
      .set_index("datetime")
      .sort_index()
      .loc[start:end]
)

router = ModelRouter(Path("models"))
train_nhits(symbol, timeframe, price_frame=df, model_router=router)
train_egarch(symbol, timeframe, price_frame=df, model_router=router)

# %%
import importlib
import sys
from pathlib import Path
import pandas as pd

# importlib.reload(ts_utils)
sys.path.append("C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation")

from dotenv import load_dotenv
load_dotenv()
from alphalens_forecast.training import train_nhits, train_neuralprophet, train_prophet, train_egarch
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.data import DataProvider   
from alphalens_forecast.evaluation import load_model, test_model, time_split, plot_forecast_vs_real

provider = DataProvider()
# frame = provider.load_data("EUR/USD", "15min", refresh=True, max_points=500000)

PROJECT_ROOT = Path("C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation")
# PROJECT_ROOT = Path("C:\\Users\\Labry\\.cache\\kagglehub\\datasets\\mczielinski\\bitcoin-historical-data\\versions\\454")

router = ModelRouter(PROJECT_ROOT / "models")
# frame = pd.read_csv( PROJECT_ROOT / "alphalens_forecast/data/cache/BTC_USD/15min.csv", parse_dates=["datetime"]).set_index("datetime")
# frame = pd.read_csv( PROJECT_ROOT / "btcusd_1-min_data.csv", parse_dates=["Timestamp"])
# frame.rename(columns={"Timestamp": "datetime"}, inplace=True)
# frame.set_index("datetime", inplace=True)
# frame = frame.tail(20000)
# frame = df_btc
DEVICE = "cuda"

# frame.rename(columns={"Close": "close"}, inplace=True)
# close_series = frame["close"].dropna()

# train, valid, test = time_split(close_series)
# print(frame.head())

print("---- Training just started ----")


# symbol = "EUR/USD"
asset_list = ["GBP/USD","AUD/USD","BTC/USD","ETH/USD","XAU/USD","XLM/USD"]
for symbol in asset_list:
    for timeframe in ("15min","30min","1h","4h"):
        train_nhits(symbol, timeframe, model_router=router,device=DEVICE)
        train_neuralprophet(symbol, timeframe, model_router=router,device=DEVICE)
        train_prophet(symbol, timeframe, model_router=router,device=DEVICE)
        train_egarch(symbol, timeframe, model_router=router)

print("---- Training just ended ----")

# %%
import importlib
import sys
from pathlib import Path

# importlib.reload(ts_utils)
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

from alphalens_forecast.evaluation import evaluate_on_test

result = evaluate_on_test("nhits", "EUR/USD", "1h")
print(result["metrics"])


# %%
#**********************************************************************************
#*                                                                                *
#*                        TEST MEAN MODEL NEURALPROPHET                           *
#*                                                                                *
#**********************************************************************************

import importlib
import sys
from pathlib import Path
import pandas as pd

# importlib.reload(ts_utils)
sys.path.append(r"C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation")


from alphalens_forecast.evaluation import load_model, test_model, time_split, plot_forecast_vs_real
from alphalens_forecast.data import DataProvider

PROJECT_ROOT = Path(r"C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation")

frame = pd.read_csv( PROJECT_ROOT / "alphalens_forecast/data/cache/XLM_USD/15min.csv", parse_dates=["datetime"]).set_index("datetime")
close_series = frame["close"].dropna()

# provider = DataProvider()
# price_frame = provider.load_data("EUR/USD", "15min")
# close_series = price_frame["close"].dropna()

roll_steps = 500
train, _, test = time_split(close_series)
model = load_model("neuralprophet", "EUR/USD", "15min")
preds = test_model("neuralprophet", model, test, "15min",train_series=train,rolling_steps=roll_steps)
plot_forecast_vs_real(preds, test[:roll_steps],show_metrics=True,show_confidence=True)

# %%
plot_forecast_vs_real(preds, test[:roll_steps],show_metrics=True,show_confidence=True)


# %%
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))
cut = 100

axes[0,0].hist(preds[cut:], bins=30, color="tab:orange", edgecolor="black")
axes[0,0].set_title("Prédictions")
axes[0,0].set_xlabel("x")
axes[0,0].set_ylabel("sin(x)")

axes[0,1].hist(test[:2000][cut:], bins=30, color="tab:blue", edgecolor="black")
axes[0,1].set_title("données réelles")
axes[0,1].set_xlabel("Valeur")
axes[0,1].set_ylabel("Fréquence")

axes[1,0].plot(preds[cut:], color="tab:orange", linewidth=2)
axes[1,0].set_title("Prédictions")
axes[1,0].set_xlabel("x")
axes[1,0].set_ylabel("sin(x)")

axes[1,1].plot(test[:2000][cut:], color="tab:blue", linewidth=2)
axes[1,1].set_title("données réelles")
axes[1,1].set_xlabel("Valeur")
axes[1,1].set_ylabel("Fréquence")

plt.tight_layout()
plt.show()

# %%
import importlib
import sys
from pathlib import Path

# importlib.reload(ts_utils)
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

from alphalens_forecast.evaluation import evaluate_mean_and_vol_on_test

result = evaluate_mean_and_vol_on_test("neuralprophet", "EUR/USD", "15min")
print(result["metrics"], result["volatility"]["metrics"])

# %%
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")
import alphalens_forecast

from alphalens_forecast.data import DataProvider

provider = DataProvider()  # nécessite TWELVE_DATA_API_KEY dans l'env
frame = provider.load_data("XAU/USD", "15min", refresh=True, max_points=500000)

# %%
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
# path_file = "/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST/alphalens_forecast/data/cache/EUR_USD/15min.csv"
path_file = r"C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation\alphalens_forecast\data\cache\EUR_USD\15min.csv"
df_eurusd = pd.read_csv(path_file, parse_dates=["datetime"])
plt.plot(df_eurusd["datetime"], df_eurusd["close"])

window = 96
recent = df_eurusd["log_return"].tail(window)
sigma_ref = float(recent.ewm(span=window, adjust=False).std().iloc[-1])
print(f"sigma ref is {sigma_ref}")

#%%
#----------------------------------------------------------------------------#
# * ALPHALENS FORECAST TESTING SCRIPT VOL                                  *
#----------------------------------------------------------------------------
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from alphalens_forecast.evaluation import (
    load_model,
    test_model,
    time_split,
    plot_forecast_vs_real,
)
from alphalens_forecast.utils.timeseries import series_to_price_frame

PROJECT_ROOT = Path("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")
frame = pd.read_csv(
    PROJECT_ROOT / "alphalens_forecast/data/cache/EUR_USD/15min.csv",
    parse_dates=["datetime"],
).set_index("datetime")

close_series = frame["close"].dropna()
train, _, test = time_split(close_series)

# EGARCH s’entraîne/évalue sur les log-returns |r|.
price_test = series_to_price_frame(test)
actual_vol = price_test["log_return"].abs().reindex(test.index, fill_value=0.0)

model = load_model("egarch", "EUR/USD", "15min")
preds = test_model("egarch", model, test, "15min", train_series=train)  # renvoie sigma

# Visualisation + métriques/conf intervalle (sur sigma vs |r|)
plot_forecast_vs_real(preds, actual_vol, show_metrics=True, show_confidence=True)

# Histogrammes + courbes (en coupant les 1000 premiers points)
cut = 100
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

axes[0, 0].hist(preds[cut:], bins=30, color="tab:orange", edgecolor="black")
axes[0, 0].set_title("Prédictions σ")

axes[0, 1].hist(actual_vol[cut:], bins=30, color="tab:blue", edgecolor="black")
axes[0, 1].set_title("Volatilité réalisée |r|")

axes[1, 0].plot(preds[cut:], color="tab:orange", linewidth=2)
axes[1, 0].set_title("Prédictions σ")

axes[1, 1].plot(actual_vol[cut:], color="tab:blue", linewidth=2)
axes[1, 1].set_title("Volatilité réalisée |r|")

plt.tight_layout()
plt.show()


# %%

#**********************************************************************************
#*                                                                                *
#*                               TEST VOL EGARCH                                  *
#*                                                                                *
#**********************************************************************************

# %%
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from alphalens_forecast.evaluation import (
    load_model,
    test_model,
    time_split,
    plot_forecast_vs_real,
)
from alphalens_forecast.utils.timeseries import series_to_price_frame

PROJECT_ROOT = Path("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")
PROJECT_ROOT = Path(r"C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation")
frame = pd.read_csv(
    PROJECT_ROOT / "alphalens_forecast/data/cache/EUR_USD/15min.csv",
    parse_dates=["datetime"],
).set_index("datetime")

close_series = frame["close"].dropna().astype(float)
train, _, test = time_split(close_series)

model = load_model("garch", "EUR/USD", "15min")
preds = test_model("garch", model, test, "15min", train_series=train)  # renvoie sigma

price_test = series_to_price_frame(test)
actual_vol = price_test["log_return"].abs().reindex(test.index, fill_value=0.0)

plot_forecast_vs_real(preds, actual_vol, show_metrics=True, show_confidence=True)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(preds.index, preds, label="σ_t prédite", color="tab:orange")
axes[0].plot(actual_vol.index, actual_vol, label="|log_return|", color="tab:blue", alpha=0.6)
axes[0].legend()
axes[0].set_title("Volatilité conditionnelle vs réalisée")

axes[1].plot(train.index[-1000:], train[-1000:], label="Train (close)", color="tab:gray")
axes[1].plot(test.index[:1000], test[:1000:], label="Test (close)", color="tab:green")
axes[1].legend()
axes[1].set_title("Fenêtre de prix utilisée")
plt.tight_layout()
plt.show()


# Diagnostic in-sample : sigma_t vs |return_t|
sigma_t = model._engine._result.conditional_volatility / 1000.0  # rescale
abs_returns = frame["return"].abs().reindex(sigma_t.index, fill_value=0.0)

plt.figure(figsize=(14, 5))
plt.plot(sigma_t.index, sigma_t, label="sigma_t (conditional)", color="tab:orange")
plt.plot(abs_returns.index, abs_returns, label="|return_t|", color="tab:blue", alpha=0.6)
plt.title("Sigma_t vs realised absolute returns")
plt.legend()
plt.tight_layout()
plt.show()


# %%

#**********************************************************************************
#*                                                                                * 
#*                        TEST TARGET PROB CURB RUNNER                           *
#*                                                                                *             
#**********************************************************************************
from alphalens_forecast.config import get_config                         
from alphalens_forecast.data import DataProvider                         
from alphalens_forecast.models import ModelRouter                        
from alphalens_forecast.forecasting import ForecastEngine                
                                                                        
config = get_config()                                                    
provider = DataProvider(config.twelve_data, auto_refresh=True)           
router = ModelRouter()  # ou ModelRouter(Path("models")) si tu veux forcer un dossier                                                        
                                                                        
engine = ForecastEngine(config, provider, router)                        
                                                                        
symbol = "EUR/USD"                                                       
timeframe = "15min"                                                      
horizons = [24]  # en heures                                   
paths = 3000                                                             
                                                                        
# Charge explicitement TES modèles (optionnel, mais pratique pour forcer un type)                                                                 
mean_model = router.load_model("prophet", symbol, timeframe)               
vol_model = router.load_egarch(symbol, timeframe)                        
                                                                        
result = engine.forecast(                                                
    symbol=symbol,                                                       
    timeframe=timeframe,                                                 
    horizons=horizons,                                                   
    paths=paths,                                                         
    use_montecarlo=True,                                                 
    trade_mode="spot",                                                   
    refresh_data=True,           # True = met à jour les données         
    force_retrain=False,         # False = utilise tes modèles existants 
    mean_model_override=mean_model,                                      
    vol_model_override=vol_model,                                        
    execution_price=None,        # mets un live price ici si tu veux     
    execution_price_source=None,                                      
)                                                                        
                                                                        
result = result  # payload complet (TP/SL, risk, etc.)  

# %%
import matplotlib.pyplot as plt
import numpy as np


plt.plot(result.predictions["24h"])


# %%
