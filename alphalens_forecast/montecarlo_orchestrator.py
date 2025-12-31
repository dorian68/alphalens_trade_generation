"""Batch orchestration helpers for the Monte Carlo simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union, Literal, TYPE_CHECKING

import hashlib
import time

import numpy as np

from alphalens_forecast.core.montecarlo import MonteCarloResult, MonteCarloSimulator

if TYPE_CHECKING:
    from alphalens_forecast.config import AppConfig

SigmaInput = Union[float, Sequence[float], np.ndarray]
SeedStrategy = Literal["fixed", "increment", "hash"]


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _coerce_sigma(sigma: SigmaInput, steps: int) -> SigmaInput:
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigma_array = np.asarray(sigma, dtype=float)
        if sigma_array.ndim != 1:
            raise ValueError("Sigma path must be one-dimensional.")
        if sigma_array.size == 0:
            raise ValueError("Sigma path must not be empty.")
        if sigma_array.size != steps:
            raise ValueError("Sigma path length must match steps.")
        if np.any(sigma_array < 0):
            raise ValueError("Sigma path values must be non-negative.")
        return sigma_array
    sigma_value = float(sigma)
    if sigma_value < 0:
        raise ValueError("Sigma must be non-negative.")
    return sigma_value


def _summarize_sigma(sigma: SigmaInput) -> Dict[str, Any]:
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigma_array = np.asarray(sigma, dtype=float)
        return {
            "mode": "path",
            "len": float(sigma_array.size),
            "mean": float(np.mean(sigma_array)),
            "min": float(np.min(sigma_array)),
            "max": float(np.max(sigma_array)),
        }
    return {"mode": "scalar", "value": float(sigma)}


def _summarize_result(result: MonteCarloResult) -> "MonteCarloSummary":
    return MonteCarloSummary(
        probability_hit_tp_before_sl=float(result.probability_hit_tp_before_sl),
        prob_sl_before_tp=float(result.prob_sl_before_tp),
        expected_pnl=float(result.expected_pnl),
        quantiles={str(k): float(v) for k, v in result.quantiles.items()},
    )


def _resolve_label(label: Optional[str], index: int) -> str:
    if label:
        return str(label)
    return f"job_{index}"


def _hash_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


@dataclass(frozen=True)
class MonteCarloJob:
    """Specification for a single Monte Carlo run."""

    current_price: float
    drift: float
    sigma: SigmaInput
    dof: float
    tp: float
    sl: float
    steps: int
    step_hours: float
    skew: float = 0.0
    label: Optional[str] = None
    seed: Optional[int] = None
    paths: Optional[int] = None
    forward_entry_price: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "current_price": float(self.current_price),
            "drift": float(self.drift),
            "sigma_summary": _summarize_sigma(self.sigma),
            "dof": float(self.dof),
            "tp": float(self.tp),
            "sl": float(self.sl),
            "steps": int(self.steps),
            "step_hours": float(self.step_hours),
            "skew": float(self.skew),
            "seed": self.seed,
            "paths": self.paths,
            "forward_entry_price": self.forward_entry_price,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MonteCarloSummary:
    """Condensed Monte Carlo metrics for reporting."""

    probability_hit_tp_before_sl: float
    prob_sl_before_tp: float
    expected_pnl: float
    quantiles: Dict[str, float]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "probability_hit_tp_before_sl": float(self.probability_hit_tp_before_sl),
            "prob_sl_before_tp": float(self.prob_sl_before_tp),
            "expected_pnl": float(self.expected_pnl),
            "quantiles": {str(k): float(v) for k, v in self.quantiles.items()},
        }


@dataclass
class MonteCarloRun:
    """Output for a single Monte Carlo job."""

    label: str
    job: MonteCarloJob
    summary: MonteCarloSummary
    duration_seconds: float
    paths: int
    seed: Optional[int]
    result: Optional[MonteCarloResult] = None
    forward_summary: Optional[MonteCarloSummary] = None
    forward_duration_seconds: Optional[float] = None
    forward_result: Optional[MonteCarloResult] = None

    def to_payload(self, *, include_paths: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "label": self.label,
            "paths": int(self.paths),
            "seed": self.seed,
            "duration_seconds": float(self.duration_seconds),
            "job": self.job.to_payload(),
            "summary": self.summary.to_payload(),
        }
        if self.forward_summary is not None:
            payload["forward_summary"] = self.forward_summary.to_payload()
            payload["forward_duration_seconds"] = (
                float(self.forward_duration_seconds) if self.forward_duration_seconds is not None else None
            )
        if include_paths and self.result is not None:
            payload["final_prices"] = self.result.final_prices.tolist()
        if include_paths and self.forward_result is not None:
            payload["forward_final_prices"] = self.forward_result.final_prices.tolist()
        return payload


@dataclass
class MonteCarloBatchResult:
    """Aggregate output for a collection of Monte Carlo jobs."""

    runs: List[MonteCarloRun]
    total_seconds: float
    paths: int
    seed: Optional[int]
    seed_strategy: SeedStrategy

    def to_payload(self, *, include_paths: bool = False) -> Dict[str, Any]:
        return {
            "total_seconds": float(self.total_seconds),
            "paths": int(self.paths),
            "seed": self.seed,
            "seed_strategy": self.seed_strategy,
            "runs": [run.to_payload(include_paths=include_paths) for run in self.runs],
        }

    def summaries_by_label(self) -> Dict[str, MonteCarloSummary]:
        return {run.label: run.summary for run in self.runs}


class MonteCarloOrchestrator:
    """Coordinate multiple Monte Carlo simulations with consistent settings."""

    def __init__(
        self,
        *,
        paths: int,
        seed: Optional[int] = None,
        show_progress: bool = True,
        debug: bool = False,
        seed_strategy: SeedStrategy = "increment",
        retain_results: bool = True,
    ) -> None:
        if paths <= 0:
            raise ValueError("paths must be positive.")
        if seed_strategy not in {"fixed", "increment", "hash"}:
            raise ValueError("seed_strategy must be one of: fixed, increment, hash.")
        self._paths = int(paths)
        self._seed = seed
        self._show_progress = bool(show_progress)
        self._debug = bool(debug)
        self._seed_strategy: SeedStrategy = seed_strategy
        self._retain_results = bool(retain_results)

    @classmethod
    def from_config(
        cls,
        config: "AppConfig",
        *,
        paths: Optional[int] = None,
        show_progress: bool = True,
        debug: bool = False,
        seed_strategy: SeedStrategy = "increment",
        retain_results: bool = True,
    ) -> "MonteCarloOrchestrator":
        resolved_paths = paths if paths is not None else config.monte_carlo.paths
        return cls(
            paths=resolved_paths,
            seed=config.monte_carlo.seed,
            show_progress=show_progress,
            debug=debug,
            seed_strategy=seed_strategy,
            retain_results=retain_results,
        )

    def run(
        self,
        job: MonteCarloJob,
        *,
        index: int = 0,
        show_progress: Optional[bool] = None,
    ) -> MonteCarloRun:
        _validate_positive("current_price", float(job.current_price))
        _validate_positive("tp", float(job.tp))
        _validate_positive("sl", float(job.sl))
        _validate_positive("step_hours", float(job.step_hours))
        if job.steps <= 0:
            raise ValueError("steps must be positive.")
        if float(job.dof) <= 2:
            raise ValueError("dof must exceed 2.")
        if job.paths is not None and int(job.paths) <= 0:
            raise ValueError("job.paths must be positive.")
        if job.forward_entry_price is not None:
            _validate_positive("forward_entry_price", float(job.forward_entry_price))

        label = _resolve_label(job.label, index)
        paths = int(job.paths) if job.paths is not None else self._paths
        sigma = _coerce_sigma(job.sigma, job.steps)
        seed = self._resolve_seed(job, label, index)
        progress = self._show_progress if show_progress is None else bool(show_progress)

        simulator = MonteCarloSimulator(
            paths=paths,
            seed=seed,
            show_progress=progress,
            debug=self._debug,
        )
        start = time.perf_counter()
        result = simulator.simulate(
            current_price=float(job.current_price),
            drift=float(job.drift),
            sigma=sigma,
            dof=float(job.dof),
            skew=float(job.skew),
            tp=float(job.tp),
            sl=float(job.sl),
            steps=int(job.steps),
            step_hours=float(job.step_hours),
            debug=self._debug,
        )
        duration = time.perf_counter() - start
        summary = _summarize_result(result)

        forward_summary = None
        forward_duration = None
        forward_result = None
        if job.forward_entry_price is not None:
            forward_simulator = MonteCarloSimulator(
                paths=paths,
                seed=seed,
                show_progress=False,
                debug=self._debug,
            )
            forward_start = time.perf_counter()
            forward_result = forward_simulator.simulate(
                current_price=float(job.forward_entry_price),
                drift=float(job.drift),
                sigma=sigma,
                dof=float(job.dof),
                skew=float(job.skew),
                tp=float(job.tp),
                sl=float(job.sl),
                steps=int(job.steps),
                step_hours=float(job.step_hours),
                debug=self._debug,
            )
            forward_duration = time.perf_counter() - forward_start
            forward_summary = _summarize_result(forward_result)

        if not self._retain_results:
            result = None
            forward_result = None

        return MonteCarloRun(
            label=label,
            job=job,
            summary=summary,
            duration_seconds=duration,
            paths=paths,
            seed=seed,
            result=result,
            forward_summary=forward_summary,
            forward_duration_seconds=forward_duration,
            forward_result=forward_result,
        )

    def run_batch(
        self,
        jobs: Iterable[MonteCarloJob],
        *,
        show_progress: Optional[bool] = None,
    ) -> MonteCarloBatchResult:
        job_list = list(jobs)
        progress = self._show_progress if show_progress is None else bool(show_progress)
        if show_progress is None and len(job_list) > 1:
            progress = False

        start = time.perf_counter()
        runs = [
            self.run(job, index=idx, show_progress=progress)
            for idx, job in enumerate(job_list)
        ]
        total = time.perf_counter() - start
        return MonteCarloBatchResult(
            runs=runs,
            total_seconds=total,
            paths=self._paths,
            seed=self._seed,
            seed_strategy=self._seed_strategy,
        )

    def _resolve_seed(self, job: MonteCarloJob, label: str, index: int) -> Optional[int]:
        if job.seed is not None:
            return int(job.seed)
        if self._seed is None:
            return None
        base_seed = int(self._seed)
        if self._seed_strategy == "fixed":
            return base_seed
        if self._seed_strategy == "increment":
            return base_seed + int(index)
        return _hash_seed(base_seed, label)


__all__ = [
    "MonteCarloBatchResult",
    "MonteCarloJob",
    "MonteCarloOrchestrator",
    "MonteCarloRun",
    "MonteCarloSummary",
]
