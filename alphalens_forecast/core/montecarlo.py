"""Monte Carlo simulation utilities for TP/SL probability estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
from tqdm.auto import tqdm


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation outputs."""

    probability_hit_tp_before_sl: float
    quantiles: Dict[str, float]
    final_prices: np.ndarray


class MonteCarloSimulator:
    """Simulate heavy-tailed geometric processes to estimate TP/SL probabilities."""

    def __init__(
        self,
        paths: int,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        if paths <= 0:
            raise ValueError("Monte Carlo paths must be positive.")
        self.paths = paths
        self._rng = np.random.default_rng(seed)
        self._show_progress = show_progress

    def simulate(
        self,
        current_price: float,
        drift: float,
        sigma: Union[float, np.ndarray],
        dof: float,
        tp: float,
        sl: float,
        steps: int,
        step_hours: float,
    ) -> MonteCarloResult:
        """
        Run the Monte Carlo simulation.

        Parameters
        ----------
        current_price:
            Latest observed price.
        drift:
            Expected log-return per step.
        sigma:
            Volatility per step (log-return standard deviation).
        dof:
            Degrees of freedom of the Student-t innovation.
        tp:
            Take-profit level.
        sl:
            Stop-loss level.
        steps:
            Number of steps to simulate.
        step_hours:
            Duration in hours for each step (used for drift integration).
        """
        if current_price <= 0 or tp <= 0 or sl <= 0:
            raise ValueError("Prices must be strictly positive for simulation.")
        if steps <= 0:
            raise ValueError("Simulation requires a positive number of steps.")
        if dof <= 2:
            raise ValueError("Student-t degrees of freedom must exceed 2.")

        if isinstance(sigma, (list, tuple, np.ndarray)):
            sigma_array = np.asarray(sigma, dtype=float)
            if sigma_array.size != steps:
                raise ValueError(
                    "Sigma path length must match the number of simulation steps."
                )
        else:
            sigma_array = np.full(steps, float(sigma))

        scaled_sigma = sigma_array * np.sqrt(step_hours)
        # Stabilise Student-t to unit variance.
        t_scale = np.sqrt((dof - 2) / dof)

        with tqdm(
            total=self.paths,
            desc="Monte Carlo paths",
            disable=not self._show_progress,
        ) as progress:
            shocks = self._rng.standard_t(dof, size=(self.paths, steps)) * t_scale
            log_returns = drift * step_hours + scaled_sigma * shocks
            cumulative = np.cumsum(log_returns, axis=1)
            prices = current_price * np.exp(cumulative)
            progress.update(self.paths)

        tp_mask = prices >= tp
        sl_mask = prices <= sl

        tp_hit = tp_mask.any(axis=1)
        sl_hit = sl_mask.any(axis=1)

        tp_first = np.where(tp_hit, tp_mask.argmax(axis=1), steps + 1)
        sl_first = np.where(sl_hit, sl_mask.argmax(axis=1), steps + 1)

        hit_tp_before_sl = tp_hit & ((tp_first < sl_first) | ~sl_hit)
        probability = float(np.mean(hit_tp_before_sl))

        final_prices = prices[:, -1]
        quantiles = {
            "p20": float(np.quantile(final_prices, 0.20)),
            "p50": float(np.quantile(final_prices, 0.50)),
            "p80": float(np.quantile(final_prices, 0.80)),
        }

        return MonteCarloResult(
            probability_hit_tp_before_sl=probability,
            quantiles=quantiles,
            final_prices=final_prices,
        )
