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
from alphalens_forecast.regime_detection.deterministic import (
    DeterministicRegimeDetector,
    RegimeConfig,
)


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
    ax.scatter(df.index, df["close"], c=colors, s=12, alpha=0.7, label="Regime")

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


def _current_run_length(regimes: pd.Series) -> int:
    if regimes.empty:
        return 0
    last = regimes.iloc[-1]
    run = 0
    for value in reversed(regimes.tolist()):
        if value == last:
            run += 1
        else:
            break
    return run


def _print_summary(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    result,
    series: pd.DataFrame,
) -> None:
    print("\nRegime detection summary (score smoothing)")
    print(f"Symbol: {symbol} | Timeframe: {timeframe}")
    if df.empty:
        print("No data available.")
        return

    start = df.index.min()
    end = df.index.max()
    print(f"Rows: {len(df)} | Range: {start} -> {end}")
    print(f"Latest regime: {result.regime} | Confidence: {result.confidence:.3f}")

    if series.empty:
        return

    counts = series["regime"].value_counts(normalize=True).mul(100).round(1)
    avg_conf = series["confidence"].mean()
    run_len = _current_run_length(series["regime"])

    print(f"Average confidence: {avg_conf:.3f}")
    print(f"Current regime run length: {run_len} bars")
    print("Regime distribution (% of bars):")
    for regime, pct in counts.items():
        print(f"- {regime}: {pct}%")


def main() -> None:
    symbol = "BTC/USD"
    timeframe = "15min"
    start = pd.Timestamp("2025-08-02", tz="UTC")
    end = pd.Timestamp("2025-08-08", tz="UTC")

    provider = DataProvider(config=TwelveDataConfig(), auto_refresh=False)
    df = provider.load_data(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        range_cache="none",
    )

    detector = DeterministicRegimeDetector(
        RegimeConfig(
            regime_mode="heuristic",
            enable_score_smoothing=True,
            score_smoothing_alpha=0.15,
        )
    )
    result = detector.detect(df)
    series = detector.detect_series(df)

    _plot_regime_series(df, series, f"{symbol} ({timeframe}) regime detection - smoothing")
    _print_summary(symbol, timeframe, df, result, series)


if __name__ == "__main__":
    main()

# %%
