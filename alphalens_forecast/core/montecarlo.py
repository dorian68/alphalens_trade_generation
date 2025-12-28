"""Monte Carlo simulation utilities for TP/SL probability estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from tqdm.auto import tqdm


# ============================================================
# Result container
# ============================================================

@dataclass
class MonteCarloResult:
    """
    Container for Monte Carlo simulation outputs.
    """

    # --- probabilistic quantities ---
    probability_hit_tp_before_sl: float
    prob_sl_before_tp: float

    # --- economic quantities ---
    expected_pnl: float

    # --- distribution diagnostics ---
    quantiles: Dict[str, float]
    final_prices: np.ndarray

    # --- optional debug info ---
    debug: Optional[Dict[str, Any]] = None


# ============================================================
# Monte Carlo engine
# ============================================================

class MonteCarloSimulator:
    """
    Simulate heavy-tailed geometric price processes
    and evaluate TP / SL stopping behaviour.
    """

    def __init__(
        self,
        paths: int,
        seed: Optional[int] = None,
        show_progress: bool = True,
        debug: bool = False,
    ) -> None:
        if paths <= 0:
            raise ValueError("Monte Carlo paths must be positive.")

        self.paths = int(paths)
        self._rng = np.random.default_rng(seed)
        self._show_progress = bool(show_progress)
        self._debug = bool(debug)
        self._seed = seed

    # --------------------------------------------------------

    def simulate(
        self,
        current_price: float,
        drift: float,
        sigma: Union[float, np.ndarray],
        dof: float,
        skew: float,
        tp: float,
        sl: float,
        steps: int,
        step_hours: float,
        debug: Optional[bool] = None,
    ) -> MonteCarloResult:
        """
        Run the Monte Carlo simulation.
        """

        # -------------------------------
        # Sanity checks
        # -------------------------------

        if current_price <= 0 or tp <= 0 or sl <= 0:
            raise ValueError("Prices must be strictly positive.")
        if steps <= 0:
            raise ValueError("Simulation requires positive steps.")
        if dof <= 2:
            raise ValueError("Student-t dof must exceed 2.")

        debug_enabled = self._debug if debug is None else bool(debug)

        # -------------------------------
        # Volatility handling
        # -------------------------------

        if isinstance(sigma, (list, tuple, np.ndarray)):
            sigma_array = np.asarray(sigma, dtype=float)
            if sigma_array.size != steps:
                raise ValueError("Sigma path length must match steps.")
            sigma_mode = "path"
        else:
            sigma_array = np.full(steps, float(sigma))
            sigma_mode = "scalar"

        scaled_sigma = sigma_array * np.sqrt(step_hours)

        # Student-t variance normalization
        t_scale = np.sqrt((dof - 2.0) / dof)

        debug_info: Optional[Dict[str, Any]] = None
        if debug_enabled:
            debug_info = {
                "paths": self.paths,
                "steps": steps,
                "seed": self._seed,
                "dof": dof,
                "skew": skew,
                "drift": drift,
                "step_hours": step_hours,
                "sigma_mode": sigma_mode,
                "sigma_mean": float(np.mean(sigma_array)),
                "t_scale": float(t_scale),
                "tp": float(tp),
                "sl": float(sl),
                "entry": float(current_price),
            }

        # -------------------------------
        # Monte Carlo generation
        # -------------------------------

        with tqdm(
            total=self.paths,
            desc="Monte Carlo paths",
            disable=not self._show_progress,
        ):
            # --- innovations ---
            raw = self._rng.standard_t(dof, size=(self.paths, steps))

            if skew != 0.0:
                raw = np.where(
                    raw >= 0.0,
                    raw * (1.0 + skew),
                    raw * (1.0 - skew),
                )

            # normalize innovations
            raw = raw - np.mean(raw)
            raw = raw / np.std(raw)

            shocks = raw * t_scale

            # --- log-price dynamics ---
            log_returns = drift * step_hours + scaled_sigma * shocks
            cumulative = np.cumsum(log_returns, axis=1)
            prices = current_price * np.exp(cumulative)

        # -------------------------------
        # TP / SL stopping logic
        # -------------------------------

        tp_mask = prices >= tp
        sl_mask = prices <= sl

        tp_hit = tp_mask.any(axis=1)
        sl_hit = sl_mask.any(axis=1)

        tp_first = np.where(tp_hit, tp_mask.argmax(axis=1), steps + 1)
        sl_first = np.where(sl_hit, sl_mask.argmax(axis=1), steps + 1)

        tp_before_sl = tp_hit & ((tp_first < sl_first) | ~sl_hit)
        sl_before_tp = sl_hit & ((sl_first < tp_first) | ~tp_hit)

        probability_tp = float(np.mean(tp_before_sl))
        probability_sl = float(np.mean(sl_before_tp))

        # -------------------------------
        # Economic PnL definition
        # -------------------------------

        pnl = np.empty(self.paths, dtype=float)
        entry = current_price

        pnl[tp_before_sl] = tp - entry
        pnl[sl_before_tp] = -(entry - sl)

        neither = ~(tp_before_sl | sl_before_tp)
        pnl[neither] = prices[neither, -1] - entry

        expected_pnl = float(np.mean(pnl))

        # -------------------------------
        # Distribution diagnostics
        # -------------------------------

        final_prices = prices[:, -1]
        quantiles = {
            "p20": float(np.quantile(final_prices, 0.20)),
            "p50": float(np.quantile(final_prices, 0.50)),
            "p80": float(np.quantile(final_prices, 0.80)),
        }

        if debug_enabled and debug_info is not None:
            debug_info.update(
                {
                    "tp_hit_rate": float(np.mean(tp_hit)),
                    "sl_hit_rate": float(np.mean(sl_hit)),
                    "tp_before_sl_rate": probability_tp,
                    "expected_pnl": expected_pnl,
                    "pnl_mean": float(np.mean(pnl)),
                    "pnl_std": float(np.std(pnl)),
                }
            )

        return MonteCarloResult(
            probability_hit_tp_before_sl=probability_tp,
            prob_sl_before_tp=probability_sl,
            expected_pnl=expected_pnl,
            quantiles=quantiles,
            final_prices=final_prices,
            debug=debug_info,
        )
