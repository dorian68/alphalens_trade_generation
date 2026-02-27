# %%
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphalens_forecast.config import TwelveDataConfig
from alphalens_forecast.data.provider import DataProvider
from alphalens_forecast.regime_detection import (
    RegimeConfig,
    fit_hmm_regime_model,
    apply_hmm_regime_model,
    HMMRegimeStore,
    load_hmm_regime_model,
)
import alphalens_forecast.regime_detection.deterministic as det

import numpy as np

try:
    from IPython.display import display as _display
except Exception:  # pragma: no cover - optional for notebooks
    def _display(obj) -> None:  # type: ignore[no-redef]
        print(obj)


def _plot_regime_series(df: pd.DataFrame, series: pd.DataFrame, title: str) -> None:
    if df.empty or series.empty:
        print("No data to plot.")
        return

    color_map = {
        "TREND_UP": "#2ca02c",
        "TREND_DOWN": "#d62728",
        "RANGE": "#1f77b4",
        "BREAKOUT_VOL_EXPANSION": "#ff7f0e",
        "STRESS_CHOP": "#9467bd",
    }

    aligned = series.reindex(df.index)
    colors = aligned["regime"].map(color_map).fillna("#7f7f7f")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["close"], color="black", linewidth=1.2, label="Close")
    ax.scatter(df.index, df["close"], c=colors, s=10, alpha=0.7, label="Regime")

    handles = [ax.lines[0]]
    for regime, color in color_map.items():
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", label=regime, markerfacecolor=color, markersize=6)
        )

    ax.legend(handles=handles, loc="best", frameon=False)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Close")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_hmm(
    *,
    symbol: str = "BTC/USD",
    timeframe: str = "15min",
    train_start: pd.Timestamp = pd.Timestamp("2025-01-02", tz="UTC"),
    train_end: pd.Timestamp = pd.Timestamp("2025-11-30", tz="UTC"),
    test_start: pd.Timestamp = pd.Timestamp("2026-02-10", tz="UTC"),
    test_end: pd.Timestamp = pd.Timestamp("2026-02-15", tz="UTC"),
    train_lookback: int = 2000,
    plot: bool = True,
) -> dict:
    provider = DataProvider(config=TwelveDataConfig(), auto_refresh=False)
    df = provider.load_data(
        symbol=symbol,
        timeframe=timeframe,
        start=train_start,
        end=test_end,
        range_cache="none",
    )

    print(f"deterministic.py = {det.__file__}")
    cleaned = df[["open", "high", "low", "close"]].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Raw rows: {len(df)} | Cleaned rows: {len(cleaned)}")

    train_df = cleaned.loc[train_start:train_end]
    test_df = cleaned.loc[test_start:test_end]

    config = RegimeConfig(
        regime_mode="hmm",
        lookback=train_lookback,
        hmm_states=5,
        hmm_n_iter=150,
        hmm_covariance="diag",
    )

    train_fit = train_df.iloc[-train_lookback:] if len(train_df) > train_lookback else train_df
    print(f"Training HMM on {len(train_fit)} bars.")
    hmm_model = fit_hmm_regime_model(train_fit, config=config)

    store = HMMRegimeStore()
    store.save(symbol, timeframe, hmm_model, metadata={"train_start": str(train_start), "train_end": str(train_end)})

    loaded = load_hmm_regime_model(symbol, timeframe, store=store)
    if loaded is None:
        raise RuntimeError("Failed to reload stored HMM regime model.")

    applied = apply_hmm_regime_model(test_df, loaded)
    print(f"Applied window rows: {len(applied)}")
    print(f"Latest regime: {applied.iloc[-1]['regime']} | Confidence: {applied.iloc[-1]['confidence']:.3f}")

    if plot:
        _plot_regime_series(test_df, applied, f"{symbol} ({timeframe}) HMM regimes (applied window)")

    # expose for notebook analysis cells
    globals()["train_df"] = train_df
    globals()["test_df"] = test_df
    globals()["hmm_model"] = hmm_model
    globals()["loaded_model"] = loaded
    globals()["applied"] = applied

    return {
        "train_df": train_df,
        "test_df": test_df,
        "hmm_model": hmm_model,
        "loaded_model": loaded,
        "applied": applied,
    }


def main() -> None:
    run_hmm()


if __name__ == "__main__":
    main()

# %% Notebook run cell (execute this before analysis if needed)
results = run_hmm()
test_df = results["test_df"]
applied = results["applied"]


# %% Analysis cell: test-set inspection
# Run this cell after main() in a notebook context.
# It shows the test window, summary stats, and regime distribution.
if "test_df" in globals() and "applied" in globals():
    print("\nTest set snapshot (tail):")
    _display(test_df.tail(10))

    print("\nTest set basic stats (close):")
    _display(test_df["close"].describe())

    print("\nApplied regime distribution (%):")
    _display(applied["regime"].value_counts(normalize=True).mul(100).round(2))

    print("\nApplied confidence summary:")
    _display(applied["confidence"].describe())

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    applied["confidence"].hist(ax=axes[0], bins=30, color="#4c78a8", alpha=0.8)
    axes[0].set_title("Confidence Distribution (Test Set)")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.2)

    applied["regime"].value_counts().plot(
        kind="bar",
        ax=axes[1],
        color="#f58518",
        alpha=0.8,
    )
    axes[1].set_title("Regime Counts (Test Set)")
    axes[1].set_xlabel("Regime")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, axis="y", alpha=0.2)

    axes[2].plot(test_df.index, test_df["close"], color="black", linewidth=1.0)
    axes[2].set_title("Close Price (Test Set)")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Close")
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()
else:
    print("Run main() first so test_df and applied are available.")


# %%
