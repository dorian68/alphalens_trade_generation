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
from tqdm import tqdm


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
    # print(f"sl price is {ctx.current_price * (1.0 - sl_sigma * sigma_ref)} and current price is {ctx.current_price}, sl sigma is {sl_sigma}, sigma ref is {sigma_ref}")
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
    max_iter: int = 40
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
    return best_tp, mc_result.expected_pnl


# ============================================================
# 4) Example usage (script entry point)
# ============================================================

def gbl_function(current_price, sl_sigma, target_prob, mc, sigma=0.00015):
    ctx = TradeContext(
        current_price=current_price,
        drift=0.0,
        sigma=0.003834107612171475,
        dof=3.0,
        skew=-0.2598382021455602,
        steps=96,
        step_hours=1,
    )
    # print(f"current price is {ctx.current_price}")

    sl_price = compute_sl_price(ctx, sl_sigma)
    scn_pnls = []

    tp_price, scn_pnl = find_tp_for_target_prob(
        mc,
        ctx,
        sl_price=sl_price,
        target_prob=target_prob,
    )
    if scn_pnl is not None:
        scn_pnls.append(scn_pnl)

    if tp_price is None:
        return np.nan

    return (
        (tp_price - ctx.current_price)
        / ctx.current_price
        / float(np.mean(ctx.sigma))
    )


# %%
if __name__ == "__main__":

    mc = MonteCarloSimulator(
    paths=500,
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

    x = np.linspace(0.10, 0.99, 60) # target_prob
    y = np.linspace(0.02, 10.0, 20) # sl_sigma
    X, Y = np.meshgrid(x, y)
    Z = np.array([
        [
        
        gbl_function(1.176893, sl_sigma=y_val, target_prob=x_val, mc=mc,sigma=0.00096) 
        
        for x_val in x] 
        for y_val in tqdm(y,desc="avancement de y")])

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