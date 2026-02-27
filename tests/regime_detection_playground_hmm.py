# %%
from __future__ import annotations

from dataclasses import replace
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
    fit_hmm_regime_model,
    apply_hmm_regime_model,
)
import alphalens_forecast.regime_detection.deterministic as det

import numpy as np


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


def _plot_state_probabilities(index: pd.Index, posterior: pd.DataFrame, title: str) -> None:
    if posterior is None or posterior.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    for col in posterior.columns:
        ax.plot(index, posterior[col], linewidth=1.2, label=col)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()


def _plot_transition_matrix(transmat: np.ndarray, title: str) -> None:
    if transmat is None or transmat.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transmat, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(range(transmat.shape[1]))
    ax.set_yticks(range(transmat.shape[0]))
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    print("\nRegime detection summary (HMM)")
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


def _fit_hmm_diagnostics(features: pd.DataFrame, config: RegimeConfig):
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        print("hmmlearn not available; skipping HMM diagnostics.")
        return None

    obs_cols = [
        "slope_norm",
        "chop_ratio",
        "atr_pct",
        "breakout_strength",
        "vol_expansion",
        "sign_flip_rate",
        "tail_score",
        "candle_score",
    ]
    missing = [col for col in obs_cols if col not in features.columns]
    if missing:
        print(f"Missing HMM features: {missing}")
        return None

    obs = features[obs_cols].to_numpy(dtype=float)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    if len(obs) < max(20, config.hmm_states * 10):
        print(f"Insufficient data for HMM (n={len(obs)})")
        return None

    lookback = min(len(obs), max(config.lookback, config.hmm_states * 10))
    train_obs = obs[-lookback:]
    mean = train_obs.mean(axis=0)
    std = train_obs.std(axis=0)
    std[~np.isfinite(std) | (std <= 0)] = 1.0
    obs_scaled = (obs - mean) / std
    train_scaled = obs_scaled[-lookback:]

    model = GaussianHMM(
        n_components=max(2, int(config.hmm_states)),
        covariance_type=config.hmm_covariance,
        n_iter=max(10, int(config.hmm_n_iter)),
        random_state=int(config.hmm_random_state),
    )
    model.fit(train_scaled)
    posterior = model.predict_proba(obs_scaled)
    state_path = model.predict(train_scaled)

    regime_map = None
    if hasattr(det, "_hmm_state_means") and hasattr(det, "_label_hmm_states"):
        base_config = replace(
            config,
            enable_walk_forward_calib=False,
            enable_score_smoothing=False,
            enable_vol_mom_confirm=False,
            regime_mode="heuristic",
        )
        train_features = features.iloc[-lookback:]
        state_means = det._hmm_state_means(train_features, state_path, posterior[-lookback:])
        regime_map = det._label_hmm_states(state_means, base_config)

    return {
        "posterior": posterior,
        "transmat": model.transmat_,
        "regime_map": regime_map,
    }


def main() -> None:
    symbol = "BTC/USD"
    timeframe = "15min"
    start = pd.Timestamp("2025-01-02", tz="UTC")
    end = pd.Timestamp("2026-02-08", tz="UTC")

    provider = DataProvider(config=TwelveDataConfig(), auto_refresh=False)
    df = provider.load_data(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        range_cache="none",
    )

    print(f"deterministic.py = {det.__file__}")
    cleaned = df[["open", "high", "low", "close"]].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"Raw rows: {len(df)} | Cleaned rows: {len(cleaned)}")

    config = RegimeConfig(
        regime_mode="hmm",
        lookback=2000,
        hmm_states=5,
        hmm_n_iter=100,
        hmm_covariance="diag",
    )
    detector = DeterministicRegimeDetector(config)
    result = detector.detect(cleaned)
    series = detector.detect_series(cleaned)

    _plot_regime_series(cleaned, series, f"{symbol} ({timeframe}) regime detection - hmm")
    _print_summary(symbol, timeframe, cleaned, result, series)

    train_lookback = 2000
    if len(cleaned) > train_lookback:
        train_df = cleaned.iloc[-train_lookback:]
        print(f"Training HMM on last {train_lookback} bars.")
    else:
        train_df = cleaned
        print("Training HMM on full window.")

    hmm_model = fit_hmm_regime_model(train_df, config=config)
    applied = apply_hmm_regime_model(cleaned, hmm_model)

    applied_result = applied.iloc[-1]
    print(f"Applied latest regime: {applied_result['regime']} | Confidence: {applied_result['confidence']:.3f}")

    features = det._compute_features(cleaned, detector.config)
    diagnostics = _fit_hmm_diagnostics(features, detector.config)
    if diagnostics is not None:
        posterior = diagnostics["posterior"]
        transmat = diagnostics["transmat"]
        regime_map = diagnostics["regime_map"]

        if regime_map:
            print("HMM state -> regime mapping:")
            for state_idx, regime in sorted(regime_map.items()):
                print(f"- state {state_idx}: {regime}")
        else:
            print("HMM state -> regime mapping: unavailable (private helpers not found).")

        posterior_df = pd.DataFrame(
            posterior,
            index=features.index,
            columns=[f"state_{i}" for i in range(posterior.shape[1])],
        )
        _plot_state_probabilities(
            posterior_df.index,
            posterior_df,
            "HMM state posterior probabilities",
        )
        _plot_transition_matrix(transmat, "HMM transition matrix")


if __name__ == "__main__":
    main()

# %%
