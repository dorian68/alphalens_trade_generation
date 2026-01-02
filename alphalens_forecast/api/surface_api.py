"""FastAPI surface API for TargetProbabilityCurve."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from alphalens_forecast.config import get_config
from alphalens_forecast.core import (
    MonteCarloSimulator,
    TPFindConfig,
    TargetProbabilityCurve,
    get_log_returns,
    horizon_to_steps,
    interval_to_hours,
)
from alphalens_forecast.data import DataProvider


class RangeSpec(BaseModel):
    min: float
    max: float
    steps: int = Field(..., gt=1)


class SurfaceRequest(BaseModel):
    symbol: str
    timeframe: str
    horizon_hours: Optional[float] = Field(default=None, gt=0)
    steps: Optional[int] = Field(default=None, gt=0)
    paths: int = Field(default=3000, gt=0)
    dof: float = Field(default=3.0, gt=2.0)
    skew: float = 0.0
    direction: Literal["long", "short"] = "long"
    target_prob: RangeSpec
    sl_sigma: RangeSpec
    n_workers: Optional[int] = None
    output_path: Optional[str] = None


class SurfacePayload(BaseModel):
    target_probs: list[float]
    sl_sigma: list[float]
    tp_sigma: list[list[float]]


class SurfaceResponse(BaseModel):
    symbol: str
    timeframe: str
    horizon_hours: float
    sigma_ref: float
    entry_price: float
    surface: SurfacePayload


app = FastAPI(title="AlphaLens Surface API")


def _seed_from_request(request: SurfaceRequest) -> int:
    parts = (
        request.symbol,
        request.timeframe,
        request.horizon_hours,
        request.steps,
        request.paths,
        request.dof,
        request.skew,
        request.direction,
        request.target_prob.min,
        request.target_prob.max,
        request.target_prob.steps,
        request.sl_sigma.min,
        request.sl_sigma.max,
        request.sl_sigma.steps,
    )
    payload = "|".join("" if value is None else str(value) for value in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _resolve_steps_and_horizon(request: SurfaceRequest, step_hours: float) -> tuple[int, float]:
    if request.horizon_hours is not None and request.steps is not None:
        raise HTTPException(status_code=400, detail="Provide horizon_hours or steps, not both.")
    if request.horizon_hours is None and request.steps is None:
        raise HTTPException(status_code=400, detail="horizon_hours is required when steps is not provided.")
    if request.horizon_hours is not None:
        horizon_hours = float(request.horizon_hours)
        steps = horizon_to_steps(horizon_hours, request.timeframe)
        return steps, horizon_hours
    steps = int(request.steps or 0)
    if steps <= 0:
        raise HTTPException(status_code=400, detail="steps must be positive.")
    return steps, float(steps * step_hours)


def _build_grid(spec: RangeSpec, *, name: str) -> np.ndarray:
    if spec.steps <= 1:
        raise HTTPException(status_code=400, detail=f"{name}.steps must be greater than 1.")
    if spec.min >= spec.max:
        raise HTTPException(status_code=400, detail=f"{name}.min must be less than {name}.max.")
    return np.linspace(spec.min, spec.max, spec.steps)


def _load_price_frame(symbol: str, timeframe: str):
    config = get_config()
    provider = DataProvider(config.twelve_data)
    frame = provider.load_data(symbol, timeframe)
    if frame.empty:
        raise HTTPException(status_code=400, detail="No data returned by DataProvider.")
    return frame.sort_index()


def _resolve_entry_price(frame) -> float:
    if "close" not in frame.columns:
        raise HTTPException(status_code=400, detail="Price frame must include a 'close' column.")
    entry_price = float(frame["close"].iloc[-1])
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise HTTPException(status_code=400, detail="entry_price must be positive and finite.")
    return entry_price


def _estimate_sigma_ref(frame) -> float:
    log_returns = get_log_returns(frame)
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if log_returns.empty:
        raise HTTPException(status_code=400, detail="Cannot compute returns from the provided data.")
    sigma_ref = float(log_returns.std())
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        raise HTTPException(status_code=400, detail="sigma_ref must be positive and finite.")
    return sigma_ref


def _save_surface(surface, output_path: Optional[str]) -> None:
    if output_path is None:
        return
    output_path = output_path.strip()
    if not output_path:
        raise HTTPException(status_code=400, detail="output_path must be a non-empty string.")
    path = Path(output_path).expanduser()
    parent = path.parent
    if parent and not parent.exists():
        raise HTTPException(
            status_code=400,
            detail=f"output_path parent directory does not exist: {parent}",
        )
    try:
        np.savez_compressed(
            path,
            target_probs=surface.target_probs,
            sl_sigma=surface.sl_sigma,
            tp_prices=surface.tp_prices,
            tp_sigma=surface.tp_sigma,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save surface to {path}: {exc}",
        ) from exc


@app.post("/surface", response_model=SurfaceResponse)
def build_surface(request: SurfaceRequest) -> SurfaceResponse:
    try:
        step_hours = interval_to_hours(request.timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    steps, horizon_hours = _resolve_steps_and_horizon(request, step_hours)
    target_probs = _build_grid(request.target_prob, name="target_prob")
    sl_sigma = _build_grid(request.sl_sigma, name="sl_sigma")

    if not np.all((target_probs > 0.0) & (target_probs < 1.0)):
        raise HTTPException(status_code=400, detail="target_prob values must be in (0, 1).")
    if not np.all(sl_sigma > 0.0):
        raise HTTPException(status_code=400, detail="sl_sigma values must be positive.")

    frame = _load_price_frame(request.symbol, request.timeframe)
    entry_price = _resolve_entry_price(frame)
    sigma_ref = _estimate_sigma_ref(frame)

    drift_per_hour = 0.0
    sigma_per_hour = sigma_ref / np.sqrt(step_hours)

    seed = _seed_from_request(request)
    simulator = MonteCarloSimulator(
        paths=request.paths,
        seed=seed,
        show_progress=False,
    )
    prices = simulator.simulate_paths(
        current_price=entry_price,
        drift=drift_per_hour,
        sigma=sigma_per_hour,
        dof=request.dof,
        skew=request.skew,
        steps=steps,
        step_hours=step_hours,
    )

    cfg = TPFindConfig()
    curve = TargetProbabilityCurve(
        prices,
        entry_price=entry_price,
        sigma_ref=sigma_ref,
        direction=request.direction,
        default_cfg=cfg,
    )
    surface = curve.build_surface(
        target_probs=target_probs,
        sl_sigma=sl_sigma,
        cfg=cfg,
        n_workers=request.n_workers,
        plot=False,
    )
    _save_surface(surface, request.output_path)

    payload = SurfacePayload(
        target_probs=surface.target_probs.tolist(),
        sl_sigma=surface.sl_sigma.tolist(),
        tp_sigma=surface.tp_sigma.tolist(),
    )
    return SurfaceResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        horizon_hours=horizon_hours,
        sigma_ref=float(sigma_ref),
        entry_price=float(entry_price),
        surface=payload,
    )
