# %%
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")
from pathlib import Path
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
    symbol="EUR/USD",
    timeframe="15min",
    horizons=[6, 12, 24],
    paths=config.monte_carlo.paths,
    use_montecarlo=config.monte_carlo.use_montecarlo,
    show_progress=False,
)


# 4. Inspect the payload
print(result.payload)          # TP/SL, quantiles, probabilities
print(result.predictions)      # mean-model forecast series
print(result.volatility.sigma) # GARCH sigma path

# (optional) Save or push the payload downstream

# %%
