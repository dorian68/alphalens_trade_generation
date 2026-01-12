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
asset_list = ["EUR/USD","GBP/USD","ETH/USD","XAU/USD","XLM/USD","BTC/USD",]
# asset_list = ["EUR/USD",]
for symbol in asset_list:
    for timeframe in ("15min","30min","1h","4h"):
        frame = provider.load_data(symbol, timeframe, refresh=True,)
        train_nhits(symbol, timeframe, model_router=router,device=DEVICE,price_frame=frame)
        train_neuralprophet(symbol, timeframe, model_router=router,device=DEVICE,price_frame=frame)
        train_prophet(symbol, timeframe, model_router=router,device=DEVICE,price_frame=frame)
        train_egarch(symbol, timeframe, model_router=router,price_frame=frame)

print("---- Training just ended ----")

# %% 
