#!/usr/bin/env python
"""Standalone Monte Carlo test that mirrors the forecasting pipeline inputs.

Example:
  python alphalens_forecast/notebook_playground/montecarlo_test.py \
    --current-price 1.078 \
    --median-price 1.081 \
    --sigma 0.0012 \
    --dof 8 \
    --timeframe 15min \
    --horizon-hours 24 \
    --paths 5000 \
    --save-paths mc_paths.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphalens_forecast.core.montecarlo import MonteCarloSimulator
from alphalens_forecast.core.volatility_bridge import horizon_to_steps, interval_to_hours


def compute_student_t_quantiles(median_log: float, sigma: float, dof: float) -> dict[str, float]:
    """Match forecasting.compute_student_t_quantiles (p20/p50/p80 on log prices)."""
    dist = t(df=dof)
    q15_log = median_log + dist.ppf(0.15) * sigma
    q50_log = median_log
    q85_log = median_log + dist.ppf(0.85) * sigma
    return {
        "p20": float(np.exp(q15_log)),
        "p50": float(np.exp(q50_log)),
        "p80": float(np.exp(q85_log)),
    }


def load_sigma_path(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if "sigma" in archive:
            data = archive["sigma"]
        else:
            data = archive[archive.files[0]]
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = np.loadtxt(path, delimiter=",")
    array = np.asarray(data, dtype=float).reshape(-1)
    return array


def build_sigma_array(
    *,
    steps: int,
    sigma: float,
    sigma_path: Optional[str],
    sigma_start: Optional[float],
    sigma_end: Optional[float],
) -> np.ndarray:
    if sigma_path:
        array = load_sigma_path(Path(sigma_path))
    elif sigma_start is not None or sigma_end is not None:
        start = sigma if sigma_start is None else sigma_start
        end = start if sigma_end is None else sigma_end
        array = np.linspace(start, end, steps)
    else:
        array = np.full(steps, float(sigma))

    array = np.asarray(array, dtype=float).reshape(-1)
    if array.size < steps:
        raise ValueError(f"Sigma path has {array.size} values, expected at least {steps}.")
    return array[:steps]


def generate_paths(
    *,
    current_price: float,
    drift: float,
    sigma_array: np.ndarray,
    dof: float,
    steps: int,
    step_hours: float,
    paths: int,
    seed: Optional[int],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scaled_sigma = sigma_array * np.sqrt(step_hours)
    t_scale = np.sqrt((dof - 2) / dof)
    shocks = rng.standard_t(dof, size=(paths, steps)) * t_scale
    log_returns = drift * step_hours + scaled_sigma * shocks
    cumulative = np.cumsum(log_returns, axis=1)
    prices = current_price * np.exp(cumulative)
    return prices


def probability_hit_tp_before_sl(prices: np.ndarray, tp: float, sl: float, steps: int) -> float:
    tp_mask = prices >= tp
    sl_mask = prices <= sl

    tp_hit = tp_mask.any(axis=1)
    sl_hit = sl_mask.any(axis=1)

    tp_first = np.where(tp_hit, tp_mask.argmax(axis=1), steps + 1)
    sl_first = np.where(sl_hit, sl_mask.argmax(axis=1), steps + 1)

    hit_tp_before_sl = tp_hit & ((tp_first < sl_first) | ~sl_hit)
    return float(np.mean(hit_tp_before_sl))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Monte Carlo test matching the forecasting pipeline inputs.",
    )
    parser.add_argument("--current-price", type=float, default=1.0)
    parser.add_argument("--median-price", type=float, default=1.0)
    parser.add_argument("--dof", type=float, default=8.0)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--sigma-path", type=str, default=None)
    parser.add_argument("--sigma-start", type=float, default=None)
    parser.add_argument("--sigma-end", type=float, default=None)
    parser.add_argument("--paths", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--horizon-hours", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--save-paths", type=str, default=None)
    args = parser.parse_args()

    steps = args.steps
    if args.horizon_hours is not None:
        steps = horizon_to_steps(args.horizon_hours, args.timeframe)

    if args.current_price <= 0 or args.median_price <= 0:
        raise ValueError("Prices must be strictly positive.")
    if steps <= 0:
        raise ValueError("Steps must be positive.")

    last_log = float(np.log(args.current_price))
    median_log = float(np.log(args.median_price))
    drift = (median_log - last_log) / steps

    sigma_array = build_sigma_array(
        steps=steps,
        sigma=args.sigma,
        sigma_path=args.sigma_path,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
    )
    if not np.isfinite(sigma_array).all():
        raise ValueError("Sigma path contains non-finite values.")

    sigma_h = float(np.sqrt(np.sum(np.square(sigma_array))))
    quantiles = compute_student_t_quantiles(median_log=median_log, sigma=sigma_h, dof=args.dof)

    direction = "long" if quantiles["p50"] >= args.current_price else "short"
    if direction == "long":
        tp = quantiles["p80"]
        sl = quantiles["p20"]
    else:
        tp = quantiles["p20"]
        sl = quantiles["p80"]

    step_hours = interval_to_hours(args.timeframe)

    mc = MonteCarloSimulator(paths=args.paths, seed=args.seed, show_progress=not args.no_progress)
    mc_result = mc.simulate(
        current_price=args.current_price,
        drift=drift,
        sigma=sigma_array,
        dof=args.dof,
        tp=tp,
        sl=sl,
        steps=steps,
        step_hours=step_hours,
    )

    summary: dict[str, object] = {
        "inputs": {
            "current_price": float(args.current_price),
            "median_price": float(args.median_price),
            "dof": float(args.dof),
            "steps": int(steps),
            "timeframe": args.timeframe,
            "step_hours": float(step_hours),
            "paths": int(args.paths),
            "seed": int(args.seed),
        },
        "sigma": {
            "per_step_min": float(np.min(sigma_array)),
            "per_step_max": float(np.max(sigma_array)),
            "sigma_h": float(sigma_h),
        },
        "direction": direction,
        "tp": float(tp),
        "sl": float(sl),
        "student_t_quantiles": quantiles,
        "mc_quantiles": mc_result.quantiles,
        "prob_hit_tp_before_sl": float(mc_result.probability_hit_tp_before_sl),
    }

    if args.forward:
        forward_mc = MonteCarloSimulator(paths=args.paths, seed=args.seed, show_progress=False)
        forward_result = forward_mc.simulate(
            current_price=quantiles["p50"],
            drift=drift,
            sigma=sigma_array,
            dof=args.dof,
            tp=tp,
            sl=sl,
            steps=steps,
            step_hours=step_hours,
        )
        summary["forward_prob_hit_tp_before_sl"] = float(
            forward_result.probability_hit_tp_before_sl
        )

    if args.save_paths:
        prices = generate_paths(
            current_price=args.current_price,
            drift=drift,
            sigma_array=sigma_array,
            dof=args.dof,
            steps=steps,
            step_hours=step_hours,
            paths=args.paths,
            seed=args.seed,
        )
        np.savez_compressed(args.save_paths, prices=prices)
        summary["paths_output"] = str(args.save_paths)
        summary["paths_quantiles"] = {
            "p20": float(np.quantile(prices[:, -1], 0.20)),
            "p50": float(np.quantile(prices[:, -1], 0.50)),
            "p80": float(np.quantile(prices[:, -1], 0.80)),
        }
        summary["paths_prob_hit_tp_before_sl"] = probability_hit_tp_before_sl(
            prices, tp, sl, steps
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


# %%
import numpy as np
from matplotlib import pyplot as plt

data_mc = np.load(r"C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation\mc_paths.npz")

for i in range(200):#len(data_mc["prices"])):
    plt.plot(data_mc["prices"][i])
# %%
# ************************************************************************
# * ALPHALENS FORECAST TESTING TP COMPUTE                                *
# *   Copyright (c) 2024 AlphaLens Labs Inc.                             *
# ************************************************************************

# ============================================================
# TP / SL calibration under probabilistic constraint
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from alphalens_forecast.core import MonteCarloSimulator
import matplotlib.pyplot as plt


# ============================================================
# 1) Market / trade context
# ============================================================

@dataclass(frozen=True)
class TradeContext:
    """
    Immutable container describing the probabilistic state
    of the market at trade entry.
    """

    current_price: float
    drift: float
    sigma: Union[float, np.ndarray]
    dof: float
    steps: int
    step_hours: float
    skew: float


# ============================================================
# 2) Economic stop-loss definition
# ============================================================

def compute_sl_price(
    ctx: TradeContext,
    sl_sigma: float,
) -> float:
    """
    Compute an economic stop-loss expressed as a multiple of volatility.
    """
    sigma_ref = float(np.mean(ctx.sigma))
    return ctx.current_price * (1.0 - sl_sigma * sigma_ref)


# ============================================================
# 3) TP search under probabilistic constraint
# ============================================================

def find_tp_for_target_prob(
    mc: MonteCarloSimulator,
    ctx: TradeContext,
    *,
    sl_price: float,
    target_prob: float = 0.75,
    tp_min_sigma: float = 0.05,
    tp_max_sigma: float = 5.0,
    tol: float = 1e-3,
    max_iter: int = 40,
    output=None
) -> Optional[float]:
    """
    Find the maximal TP price such that:
        P(hit TP before SL) >= target_prob
    """

    if not (0.0 < target_prob < 1.0):
        raise ValueError("target_prob must be in (0, 1).")

    sigma_ref = float(np.mean(ctx.sigma))

    tp_low = ctx.current_price * (1.0 + tp_min_sigma * sigma_ref)
    tp_high = ctx.current_price * (1.0 + tp_max_sigma * sigma_ref)
    # print(f"Initial TP search in [{tp_low:.5f}, {tp_high:.5f}]")

    best_tp: Optional[float] = None

    for i in range(max_iter):
        # print(f"Iteration {i+1}: TP search in [{tp_low:.5f}, {tp_high:.5f}]")
        tp_mid = 0.5 * (tp_low + tp_high)

        mc_result = mc.simulate(
            current_price=ctx.current_price,
            drift=ctx.drift,
            sigma=ctx.sigma,
            dof=ctx.dof,
            skew=ctx.skew,
            tp=tp_mid,
            sl=sl_price,
            steps=ctx.steps,
            step_hours=ctx.step_hours,
        )

        p = mc_result.probability_hit_tp_before_sl
        plt.plot(mc_result.final_prices)
        # print(f"Expected PnL: {mc_result.expected_pnl}")
        # print(f"probability hit tp before sl: {p}")

        if p >= target_prob:
            best_tp = tp_mid
            tp_low = tp_mid      # try more ambitious TP
        else:
            tp_high = tp_mid    # TP too far
        # if best_tp is None:
        #     best_tp = np.nan
        # print(f"Iteration {i+1}: TP search in [{tp_low:.5f}, {tp_high:.5f}]")
        # print(f"  -> P(hit TP before SL) = {p:.4f} ")
        # print(f"  -> Current best TP = {best_tp:.5f} ")
        # print(f"  -> TP range = {abs(tp_high - tp_low) / ctx.current_price}")
        if abs(tp_high - tp_low) / ctx.current_price < tol:
            break
        if output is not None:
            return mc_result.expected_pnl
    return best_tp


# ============================================================
# 4) Example usage (script entry point)
# ============================================================

def gbl_function(current_price, sl_sigma, target_prob, mc, sigma=0.00015, output=None):
    ctx = TradeContext(
        current_price=current_price,
        drift=0.0,
        sigma=sigma,
        dof=3.0,
        skew=-0.35,
        steps=96,
        step_hours=1,
    )
    # print(f"current price is {ctx.current_price}")

    sl_price = compute_sl_price(ctx, sl_sigma)

    tp_price = find_tp_for_target_prob(
        mc,
        ctx,
        sl_price=sl_price,
        target_prob=target_prob,
        output=output,
    )

    if tp_price is None:
        return np.nan

    if output is not None:
        return tp_price

    return (
        (tp_price - ctx.current_price)
        / ctx.current_price
        / float(np.mean(ctx.sigma))
    )


# %%
if __name__ == "__main__":

    mc = MonteCarloSimulator(
    paths=1000,
    seed=34,
    show_progress=False,
    debug=True,
)

    # current_price=1.77893,
    # drift=0.5,
    # sigma=0.00025,
    # dof=3.0,
    # skew=-0.35,
    # steps=96,
    # step_hours=0.25,

    x = np.linspace(0.10, 0.99, 10) # target_prob
    y = np.linspace(0.02, 10.0, 10) # sl_sigma
    X, Y = np.meshgrid(x, y)
    Z = np.array([[gbl_function(1.77893, sl_sigma=y_val, target_prob=x_val, mc=mc,sigma=0.00035,output="pnl") for x_val in x] for y_val in y])

    from matplotlib import colors

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    Z_masked = np.ma.masked_invalid(Z)
    norm = colors.Normalize(vmin=Z_masked.min(), vmax=Z_masked.max())

    surf = ax.plot_surface(
        Y, X, Z_masked,
        cmap="plasma",
        norm=norm,
        edgecolor="none",
        antialiased=True
    )

    ax.set_xlabel("stop-loss (in sigma)")
    ax.set_ylabel("target probability")
    ax.set_zlabel("take-profit (in sigma)")

    ax.view_init(elev=30, azim=140)

    fig.colorbar(surf, shrink=0.6, aspect=12, label="take-profit (sigma)")
    plt.show()
# %%
