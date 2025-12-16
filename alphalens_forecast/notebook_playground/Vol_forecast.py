# %%
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_pacf

# %%
path_file = "/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST/alphalens_forecast/data/cache/EUR_USD/15min.csv"
df = pd.read_csv(path_file, parse_dates=['datetime'], index_col='datetime')
df = df.sort_index()

plt.plot(df['close'])
plt.title('EUR/USD Close Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()


# %%
plt.plot(df["return"])
plt.title('EUR/USD Returns')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.show()  

# %%
std_return = df["return"].std()
mean_return = np.mean(df["return"])
print(f"Mean: {mean_return}, Std Dev: {std_return}")
actual_var = [(x - mean_return) ** 2 for x in df['return']][-365:]

# %%
plot_pacf(df["return"]**2)
plt.show()


# %%
# GARCH Normal distribution

from arch import arch_model
am = arch_model(df["return"] * 1000, p=1,q=1, mean= "ar", vol="GARCH", dist="skewt")

normal_result = am.fit()

print(normal_result.summary())

# reset = normal_result 
normal_result.plot()
plt.show()

# Diagnostic: compare conditional sigma_t to realised |returns|
sigma_t = normal_result.conditional_volatility / 1000.0  # rescale back to original units
abs_returns = df["return"].abs().reindex(sigma_t.index, fill_value=0.0)
plt.figure(figsize=(14, 5))
plt.plot(sigma_t.index, sigma_t, label="sigma_t (conditional)", color="tab:orange")
plt.plot(abs_returns.index, abs_returns, label="|return_t|", color="tab:blue", alpha=0.6)
plt.title("GARCH conditional volatility vs realised absolute returns")
plt.legend()
plt.tight_layout()
plt.show()

# %%
horizon = 3000  # nb de pas à prévoir
forecast = normal_result.forecast(horizon=horizon, reindex=False)
sigma = np.sqrt(forecast.variance.iloc[-1]) / 1000  # tu avais multiplié les returns par 1000, donc on redivise

# %%
# method = "simulation" to get simulated paths
sim_analytic = normal_result.forecast(horizon=horizon, reindex=False, method="analytic", simulations=1000)
paths_sigma_analytics = np.sqrt(sim_analytic.variance.mean(axis=0)) / 1000
params = normal_result.params
print("omega, alpha, beta:", params["omega"], params["alpha[1]"], params["beta[1]"])
plt.plot(paths_sigma_analytics, color='green', label='Analytic Forecast')

# method = "analytic" to get analytic forecast
sim_simulation = normal_result.forecast(horizon=horizon, reindex=False, method="simulation", simulations=1000)
paths_sigma_simulation = np.sqrt(sim_simulation.variance.mean(axis=0)) / 1000
params = normal_result.params
print("omega, alpha, beta:", params["omega"], params["alpha[1]"], params["beta[1]"])
plt.plot(paths_sigma_simulation, color='red', label='Simulation Forecast')

# method = "bootstrap" to get bootstrap forecast
sim_bootstrap = normal_result.forecast(horizon=horizon, reindex=False, method="bootstrap", simulations=1000)
paths_sigma_bootstrap = np.sqrt(sim_bootstrap.variance.mean(axis=0)) / 1000
params = normal_result.params
print("omega, alpha, beta:", params["omega"], params["alpha[1]"], params["beta[1]"])
plt.plot(paths_sigma_bootstrap, color='blue', label='Bootstrap Forecast')


# %%
# EGARCH Normal distribution

am_egarch = arch_model(df["return"] * 1000, p=2,q=1,o=1, vol="EGARCH", dist="t")
egarch_result = am_egarch.fit()
print(egarch_result.summary())

egarch_result.plot()
plt.show()


# %%
# GJR
am = arch_model(df['return'], p = 2, q = 2, o = 2, mean='ar', vol = 'GARCH', dist = 't')

gjr_result = am.fit()

print(gjr_result.summary())

gjr_result.plot()
plt.show()

# %%
#************************************************************
#*  
#*          BACKTESTING on historical data
#*
#************************************************************
import sys
sys.path.append("/home/ubuntu/.vscode-server/projects/alphalens_forecast/ALPHALENS_FORECAST")

from dotenv import load_dotenv
load_dotenv()
from alphalens_forecast.training import train_nhits, train_neuralprophet, train_prophet, train_egarch
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.data import DataProvider   
from alphalens_forecast.evaluation import load_model, test_model, time_split, plot_forecast_vs_real
import pandas as pd
from arch import arch_model

data_provider = DataProvider()
df = data_provider.load_data("EUR_USD", "15min")
r = df["return"].dropna()
train, _ , test = time_split(r, train_ratio=0.8)

garch_model = arch_model(train * 1000, p=1,q=1, mean="ar", vol="GARCH", dist="skewt")
garch_trained = garch_model.fit()

garch_trained.summary()

forecast_vol = garch_trained.forecast(horizon=len(test),reindex=False,method="analytic")



# %%
