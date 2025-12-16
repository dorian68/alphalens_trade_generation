
# %%
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import os
import numpy as np


def moving_average(serie: list, l: int) -> list:
    """Compute the moving average of a series.

    Args:
        serie (list): The input series.
        l (int): The window length for the moving average.

    Returns:
        list: The moving average of the input series.
    """
    if l <= 0:
        raise ValueError("Window length must be positive")
    if l > len(serie):
        raise ValueError("Window length must not be greater than the length of the series")

    ma = []
    for i in range(len(serie)):
        if i < l - 1:
            ma.append(None)  # Not enough data to compute MA
        else:
            window = serie[i - l + 1:i + 1]
            ma.append(sum(window) / l)
    return ma
# %%
if __name__ == "__main__":

    csv_path = r"C:\Users\Do\Downloads\ALPHALENS_PREDICT\data\cache\BTC_USD\30min.csv"

    print(os.path.exists(csv_path))  # doit afficher True
    df_dodo = pd.read_csv(csv_path)
    df_dodo["logX"] = df_dodo["close"].apply(lambda x: np.log(x))
    df_dodo["MA"] = moving_average(df_dodo["logX"].tolist(), l=100)
    df_dodo["variations"] = df_dodo["logX"].diff()
      
    # example usage
    print(df_dodo.columns)
    #plt.plot(df_dodo['logX'])
    #plt.plot(df_dodo['MA'], color='red')
    plt.plot(df_dodo['variations'], color='green')
    plt.show()
    print(len(df_dodo["variations"]))
# %%

# %%
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    y_train = df_dodo[df_dodo["datetime"] < "2025-10-30"]
    y_test = df_dodo[df_dodo["datetime"] >= "2025-10-30"]
    y_train["unique_id"] = "serie_btc_usd"
    y_train["datetime"] = pd.to_datetime(y_train["datetime"])

    horizon = len(df_dodo["variations"])
    input_size = horizon // 2

    model = NHITS(
        h=horizon,
        input_size=input_size,
        max_steps=2000,
        scaler_type='standard'
    )

    nf = NeuralForecast(models=[model], freq='30min')

# %%
    y_train["variations"].fillna(0, inplace=True)
    nf.fit(df=y_train[['unique_id','datetime', 'variations']].rename(columns={'datetime': 'ds', 'variations': 'y'}))
    # print(y_train)

# %%
    import pickle

    save_path = r"C:\Users\Do\model_nhits.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(nf, f)

    print(f"Modèle sauvegardé dans {save_path}")

# %%
    with open(save_path, "rb") as f:
    nf_loaded = pickle.load(f)

    Y_hat = nf_loaded.predict()

# %%
    Y_hat = nf.predict()   # prédictions sur la suite de la série après train
    print(Y_hat.tail())


# %%
    print(np.max(Y_hat["NHITS"].values))
    plt.plot(y_train['datetime'],y_train['close'].values, color='blue')
    plt.plot(Y_hat["ds"],Y_hat['NHITS'].values, color='orange')
# %%
    # Reconstruction de la serie predite
    last_val = y_train['logX'].iloc[-1]
    reconstructed_series = [last_val + Y_hat['NHITS'].iloc[0]]
    for i in range(1, len(Y_hat)):
        next_val = reconstructed_series[-1] + Y_hat['NHITS'].iloc[i]
        reconstructed_series.append(next_val)

    plt.plot(np.exp(y_test['logX'].values), label='True Log Prices', color='blue')
    plt.plot(np.exp(reconstructed_series), label='Predicted Log Prices', color='orange')

# %%
    # ============================================================
# 1) On extrait uniquement les prédictions FUTURES
# ============================================================

# Convertir les dates
Y_hat['ds'] = pd.to_datetime(Y_hat['ds'])
y_test['datetime'] = pd.to_datetime(y_test['datetime'])

# On garde la partie des prédictions correspondant à y_test
Y_future = Y_hat[Y_hat['ds'] >= y_test['datetime'].iloc[0]].copy()

# Le nombre de points de prédiction doit correspondre à y_test
Y_future = Y_future.iloc[:len(y_test)]

# ============================================================
# 2) Inverse scaling automatique via NF
#    (NHITS fait déjà inverse_transform)
# ============================================================

scaled_preds = Y_future['NHITS'].values.reshape(-1, 1)
scaler = nf.models[0].scaler

preds_unscaled = scaler.inverse_transform(scaled_preds).flatten()

# ============================================================
# 3) Reconstruction des log-prix à partir des variations
# ============================================================

last_log_price = y_train['logX'].iloc[-1]

reconstructed_log_price = [last_log_price]

for diff in preds_unscaled:
    next_val = reconstructed_log_price[-1] + diff
    reconstructed_log_price.append(next_val)

# enlever la valeur initiale
reconstructed_log_price = reconstructed_log_price[1:]

# ============================================================
# 4) Superposition graphique
# ============================================================

plt.figure(figsize=(14, 6))

# Vraies valeurs test
plt.plot(
    y_test['datetime'].values[:len(reconstructed_log_price)],
    y_test['logX'].values[:len(reconstructed_log_price)],
    label='True Future Log Prices',
    color='blue'
)

# Prédictions reconstruite
plt.plot(
    y_test['datetime'].values[:len(reconstructed_log_price)],
    reconstructed_log_price,
    label='Predicted Log Prices (NHITS)',
    color='orange'
)

# Historique train pour contexte
plt.plot(
    y_train['datetime'].values[-200:],   # les 200 derniers points
    y_train['logX'].values[-200:], 
    label='Train (last 200 pts)',
    color='green'
)

plt.legend()
plt.title("Comparaison des log-prix futurs vs prédiction NHITS")
plt.xlabel("Date")
plt.ylabel("Log-Price")
plt.grid(True)
plt.show()

# %%
