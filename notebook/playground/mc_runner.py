"""Monte Carlo execution runner with optional multiprocessing and batching.

This module keeps MonteCarloSimulator as the numerical engine and adds orchestration:
- execution mode selection (single or multiprocessing)
- optional path batching per run
- optional path reuse across TP/SL evaluations
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import copy
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from alphalens_forecast.core.montecarlo import MonteCarloResult, MonteCarloSimulator

SigmaInput = Union[float, Sequence[float], np.ndarray]


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _coerce_tp_sl_grid(tp_sl_grid: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    for item in tp_sl_grid:
        if len(item) != 2:
            raise ValueError("Each tp_sl_grid entry must contain exactly two values.")
        tp, sl = float(item[0]), float(item[1])
        pairs.append((tp, sl))
    return pairs


def _get_rng_state(simulator: MonteCarloSimulator) -> Dict[str, Any]:
    return copy.deepcopy(simulator._rng.bit_generator.state)


def _set_rng_state(simulator: MonteCarloSimulator, state: Dict[str, Any]) -> None:
    simulator._rng.bit_generator.state = copy.deepcopy(state)


def _set_simulator_seed(simulator: MonteCarloSimulator, seed: int) -> None:
    simulator._rng = np.random.default_rng(seed)
    if hasattr(simulator, "_seed"):
        simulator._seed = seed


def _split_paths(total_paths: int, batch_paths: Optional[int]) -> List[int]:
    if batch_paths is None or batch_paths <= 0 or batch_paths >= total_paths:
        return [int(total_paths)]
    batches: List[int] = []
    remaining = int(total_paths)
    while remaining > 0:
        count = min(batch_paths, remaining)
        batches.append(int(count))
        remaining -= count
    return batches


def _combine_results(
    results: List[MonteCarloResult],
    counts: List[int],
) -> MonteCarloResult:
    if len(results) != len(counts):
        raise ValueError("Results and counts must have matching lengths.")
    total = int(sum(counts))
    if total <= 0:
        raise ValueError("Total paths must be positive.")

    prob_tp = sum(
        res.probability_hit_tp_before_sl * count for res, count in zip(results, counts)
    ) / total
    prob_sl = sum(
        res.prob_sl_before_tp * count for res, count in zip(results, counts)
    ) / total
    expected_pnl = sum(
        res.expected_pnl * count for res, count in zip(results, counts)
    ) / total

    final_prices = np.concatenate([res.final_prices for res in results], axis=0)
    quantiles = {
        "p20": float(np.quantile(final_prices, 0.20)),
        "p50": float(np.quantile(final_prices, 0.50)),
        "p80": float(np.quantile(final_prices, 0.80)),
    }

    return MonteCarloResult(
        probability_hit_tp_before_sl=float(prob_tp),
        prob_sl_before_tp=float(prob_sl),
        expected_pnl=float(expected_pnl),
        quantiles=quantiles,
        final_prices=final_prices,
        debug=None,
    )


def _simulate_with_optional_batch(
    *,
    paths: int,
    batch_paths: Optional[int],
    seed: Optional[int],
    rng_state: Optional[Dict[str, Any]],
    show_progress: bool,
    current_price: float,
    drift: float,
    sigma: SigmaInput,
    dof: float,
    skew: float,
    tp: float,
    sl: float,
    steps: int,
    step_hours: float,
) -> MonteCarloResult:
    batches = _split_paths(paths, batch_paths)
    if len(batches) == 1:
        simulator = MonteCarloSimulator(
            paths=paths,
            seed=seed,
            show_progress=show_progress,
        )
        if rng_state is not None:
            simulator._rng.bit_generator.state = copy.deepcopy(rng_state)
        return simulator.simulate(
            current_price=current_price,
            drift=drift,
            sigma=sigma,
            dof=dof,
            skew=skew,
            tp=tp,
            sl=sl,
            steps=steps,
            step_hours=step_hours,
            debug=None,
        )

    simulator = MonteCarloSimulator(
        paths=batches[0],
        seed=seed,
        show_progress=False,
    )
    if rng_state is not None:
        simulator._rng.bit_generator.state = copy.deepcopy(rng_state)

    results: List[MonteCarloResult] = []
    for count in batches:
        simulator.paths = int(count)
        results.append(
            simulator.simulate(
                current_price=current_price,
                drift=drift,
                sigma=sigma,
                dof=dof,
                skew=skew,
                tp=tp,
                sl=sl,
                steps=steps,
                step_hours=step_hours,
                debug=None,
            )
        )
    return _combine_results(results, batches)


def _run_tp_sl_worker(
    args: Tuple[
        float,
        float,
        float,
        float,
        SigmaInput,
        float,
        float,
        int,
        float,
        int,
        Optional[int],
        Optional[Dict[str, Any]],
        Optional[int],
    ]
) -> Tuple[Tuple[float, float], MonteCarloResult]:
    (
        tp,
        sl,
        current_price,
        drift,
        sigma,
        dof,
        skew,
        steps,
        step_hours,
        paths,
        seed,
        rng_state,
        batch_paths,
    ) = args
    result = _simulate_with_optional_batch(
        paths=paths,
        batch_paths=batch_paths,
        seed=seed,
        rng_state=rng_state,
        show_progress=False,
        current_price=current_price,
        drift=drift,
        sigma=sigma,
        dof=dof,
        skew=skew,
        tp=tp,
        sl=sl,
        steps=steps,
        step_hours=step_hours,
    )
    return (tp, sl), result


class MonteCarloRunner:
    """Orchestrate Monte Carlo TP/SL sweeps with optional parallelism and batching."""

    def __init__(
        self,
        *,
        current_price: float,
        drift: float,
        sigma: SigmaInput,
        dof: float,
        steps: int,
        step_hours: float,
        skew: float = 0.0,
        batch_paths: Optional[int] = None,
    ) -> None:
        _validate_positive("current_price", float(current_price))
        _validate_positive("step_hours", float(step_hours))
        if steps <= 0:
            raise ValueError("steps must be positive.")
        self._current_price = float(current_price)
        self._drift = float(drift)
        self._sigma = sigma
        self._dof = float(dof)
        self._steps = int(steps)
        self._step_hours = float(step_hours)
        self._skew = float(skew)
        if batch_paths is not None and batch_paths <= 0:
            raise ValueError("batch_paths must be positive when provided.")
        self._batch_paths = int(batch_paths) if batch_paths is not None else None

    def run(
        self,
        simulator: MonteCarloSimulator,
        *,
        tp_sl_grid: Iterable[Sequence[float]],
        execution_mode: str = "single",
        n_workers: Optional[int] = None,
        reuse_paths: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[Tuple[float, float], MonteCarloResult]:
        """
        Parameters
        ----------
        simulator : MonteCarloSimulator
            An EXISTING instance imported from montecarlo.py

        tp_sl_grid : iterable
            Iterable of (tp, sl) pairs to evaluate

        execution_mode : str
            One of:
              - "single" (default, identical behavior to current usage)
              - "mp"     (multiprocessing)

        n_workers : int | None
            Number of worker processes (mp only)

        reuse_paths : bool
            If True:
              - simulate price paths ONCE
              - reuse them across all TP/SL evaluations
            If False:
              - fall back to current per-call simulation behavior

        seed : int | None
            Optional seed for reproducibility

        Returns
        -------
        dict
            Mapping (tp, sl) -> result
        """
        if execution_mode not in {"single", "mp"}:
            raise ValueError("execution_mode must be 'single' or 'mp'.")
        if n_workers is not None and n_workers <= 0:
            raise ValueError("n_workers must be positive when provided.")

        pairs = _coerce_tp_sl_grid(tp_sl_grid)
        results: Dict[Tuple[float, float], MonteCarloResult] = {}

        paths = int(getattr(simulator, "paths", 0))
        if paths <= 0:
            raise ValueError("simulator.paths must be positive.")

        batch_paths = self._batch_paths
        if batch_paths is not None and batch_paths >= paths:
            batch_paths = None

        show_progress = bool(getattr(simulator, "_show_progress", True))

        if execution_mode == "single":
            base_state: Optional[Dict[str, Any]] = None
            if reuse_paths:
                if seed is not None:
                    _set_simulator_seed(simulator, seed)
                base_state = _get_rng_state(simulator)
            else:
                if seed is not None:
                    _set_simulator_seed(simulator, seed)

            for index, (tp, sl) in enumerate(pairs):
                if reuse_paths and base_state is not None:
                    _set_rng_state(simulator, base_state)

                if batch_paths is None:
                    result = simulator.simulate(
                        current_price=self._current_price,
                        drift=self._drift,
                        sigma=self._sigma,
                        dof=self._dof,
                        skew=self._skew,
                        tp=float(tp),
                        sl=float(sl),
                        steps=self._steps,
                        step_hours=self._step_hours,
                        debug=None,
                    )
                else:
                    if reuse_paths:
                        batch_seed = seed
                        rng_state = base_state if seed is None else None
                    else:
                        batch_seed = None if seed is None else int(seed) + index
                        rng_state = None
                    result = _simulate_with_optional_batch(
                        paths=paths,
                        batch_paths=batch_paths,
                        seed=batch_seed,
                        rng_state=rng_state,
                        show_progress=False,
                        current_price=self._current_price,
                        drift=self._drift,
                        sigma=self._sigma,
                        dof=self._dof,
                        skew=self._skew,
                        tp=float(tp),
                        sl=float(sl),
                        steps=self._steps,
                        step_hours=self._step_hours,
                    )

                results[(tp, sl)] = result
            return results

        if n_workers is None:
            n_workers = os.cpu_count() or 1

        base_seed = seed
        base_state = None
        if reuse_paths and seed is None:
            base_state = _get_rng_state(simulator)

        ctx = mp.get_context("spawn")
        tasks = []
        for index, (tp, sl) in enumerate(pairs):
            if reuse_paths:
                task_seed = base_seed
                task_state = base_state
            else:
                task_seed = None if base_seed is None else int(base_seed) + index
                task_state = None
            tasks.append(
                (
                    float(tp),
                    float(sl),
                    self._current_price,
                    self._drift,
                    self._sigma,
                    self._dof,
                    self._skew,
                    self._steps,
                    self._step_hours,
                    paths,
                    task_seed,
                    task_state,
                    batch_paths,
                )
            )

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_run_tp_sl_worker, task): task for task in tasks}
            for future in as_completed(futures):
                key, result = future.result()
                results[key] = result

        return results


__all__ = ["MonteCarloRunner"]
