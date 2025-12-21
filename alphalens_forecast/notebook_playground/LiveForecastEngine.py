# %%
import sys
from pathlib import Path

sys.path.append(r"C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation")

from alphalens_forecast.forecasting import ForecastEngine
from alphalens_forecast.config import AppConfig, get_config
from alphalens_forecast.data import DataProvider
from alphalens_forecast.models import ModelRouter

# 1. Load config (or instantiate AppConfig() manually)
config = get_config()          # or simply AppConfig()

# Prepare shared services
data_provider = DataProvider()
model_router = ModelRouter()

# 2. Create the engine (uses DataProvider + ModelRouter under the hood)
engine = ForecastEngine(config=config, data_provider=data_provider, model_router=model_router)

# 3. Run a live forecast for a symbol/timeframe
result = engine.forecast(
    symbol="XAU/USD",
    timeframe="15min",
    horizons=[4, 6, 8],
    paths=config.monte_carlo.paths,
    use_montecarlo=config.monte_carlo.use_montecarlo,
    show_progress=False,
)


# 4. Inspect the payload
print("1-step ahead forecasts:")
print(result.payload)          # TP/SL, quantiles, probabilities
print("2-step ahead forecasts:")
print(result.predictions)      # mean-model forecast series
print("3-step ahead forecasts:")
print(result.volatility.sigma) # GARCH sigma path

# 5. (optional) Visualize the forecast
import matplotlib.pyplot as plt
import pandas as pd

# df_preds_results = pd.DataFrame.from_dict(result.predictions)
df_preds = pd.concat(
    {h: df["yhat"] for h, df in result.predictions.items()},
    axis=1
)
# df_preds.columns -> MultiIndex, niveau 0 = horizon

# plt.plot(result.predictions['mean_forecast'])
# plt.title("Live Forecast for EUR/USD (15min)")
# plt.xlabel("Time")  
# plt.ylabel("Price")

# (optional) Save or push the payload downstream

# # %%
# import kagglehub

# # Download selected version
# path = kagglehub.dataset_download("mczielinski/bitcoin-historical-data/versions/450")

# print("Path to dataset files:", path)

# # %%
# import pandas as pd
# import matplotlib.pyplot as plt

# path = r"C:/Users/Labry/.cache/kagglehub/datasets/mczielinski/bitcoin-historical-data/versions/450/btcusd_1-min_data.csv"
# df_BTC = pd.read_csv(path, parse_dates=['Timestamp']) 
# plt.plot(df_BTC['Timestamp'], df_BTC['Close'])

# # %%

# %%
#**********************************************************************************
#*                                                                                *
#*                        LIVE PROD VS REAL DATAS                                 *
#*                                                                                *
#**********************************************************************************

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# sys.path.append(r"C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation")

PROJECT_ROOT = Path(r"C:/Users/Labry/Documents/ALPHALENS_PRJOECT_FORECAST/alphalens_trade_generation/alphalens_forecast")


path_asset_price = PROJECT_ROOT / "data" / "cache" / "EUR_USD" / "15min.csv"
df_asset_price = pd.read_csv(path_asset_price, parse_dates=['datetime'])
df_asset_price["datetime"] = pd.to_datetime(df_asset_price["datetime"], utc=True).dt.tz_convert(None)
df_preds.index = pd.to_datetime(df_preds.index, utc=True).tz_convert(None)

df_preds_min, df_preds_max = df_preds.index.min(), df_preds.index.max()
print(f"Preds range: {df_preds_min} - {df_preds_max}")
df_sub = df_asset_price[df_asset_price["datetime"].between(df_preds_min, df_preds_max)]

# download up to date real data
from alphalens_forecast.data import DataProvider

df_preds_min = pd.Timestamp(df_preds_min).floor("15min")
df_preds_max = pd.Timestamp(df_preds_max).ceil("15min")
data_provider = DataProvider()
df_real = data_provider.load_data(
    symbol="EUR_USD",
    timeframe="15min",
    start=df_preds_min,
    end=df_preds_max,
    range_cache="separate",
)
print(df_real.shape, df_real.index.min(), df_real.index.max())

# %%
#**********************************************************************************
#*                                                                                *
#*                    LIVE PROD VS REAL DATAS : EVALUATION                        *
#*                                                                                *
#**********************************************************************************

# plt.plot(df_real.index, df_real['close'], label='Real Close Price', color='blue')
# plt.plot(df_preds.index, df_preds["4h"], label='Forecast Horizon 4', color='orange')
plt.plot(df_preds.index, df_preds["6h"], label='Forecast Horizon 6', color='green')
# plt.plot(df_preds.index, df_preds["8h"], label='Forecast Horizon 8', color ='red')
plt.title("Live Forecast vs Real Data for EUR/USD (15min)")


# %%
# df_preds : DataFrame avec index = dates (ou convertible en dates)
# df_real  : DataFrame avec index datetime (UTC) et colonne "close"

df_out = df_preds.copy()
df_out.index = pd.to_datetime(df_out.index, utc=True).tz_convert(None)

df_real_close = df_real[["close"]].copy()
df_real_close.index = pd.to_datetime(df_real_close.index, utc=True).tz_convert(None)

df_merged = pd.concat(
    [df_out, df_real_close.rename(columns={"close": "close_real"})],
    axis=1,
).sort_index()

# %%
