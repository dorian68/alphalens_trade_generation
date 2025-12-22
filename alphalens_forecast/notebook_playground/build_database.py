# %%
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")
import alphalens_forecast

from alphalens_forecast.data import DataProvider

asset_list = ["BTC/USD"]

provider = DataProvider()  # nécessite TWELVE_DATA_API_KEY dans l'env
for asset in asset_list:
    frame = provider.load_data(asset, "15min", refresh=True, max_points=500000)
# %%
