"""FastAPI surface API for TargetProbabilityCurve."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
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
    skew: Optional[float] = None
    methodology: Literal["legacy", "research"] = "legacy"
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
    atr: float
    entry_price: float
    methodology: Optional[str] = None
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


def _estimate_sigma_ref_and_skew(
    frame,
    requested_skew: Optional[float],
) -> tuple[float, float]:
    log_returns = get_log_returns(frame)
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if log_returns.empty:
        raise HTTPException(status_code=400, detail="Cannot compute returns from the provided data.")
    window = 1
    if len(log_returns) > 1:
        deltas = log_returns.index.to_series().diff().dropna()
        if not deltas.empty:
            median_delta = deltas.median()
            if hasattr(median_delta, "total_seconds"):
                bar_seconds = median_delta.total_seconds()
            else:
                try:
                    bar_seconds = float(median_delta / np.timedelta64(1, "s"))
                except Exception:
                    bar_seconds = None
            if bar_seconds and np.isfinite(bar_seconds) and bar_seconds > 0:
                window = max(1, int((24 * 60 * 60) / bar_seconds))
    if window > len(log_returns):
        window = len(log_returns)
    recent = log_returns.tail(window)
    sigma_ref = float(recent.ewm(span=window, adjust=False).std().iloc[-1])
    if not np.isfinite(sigma_ref) or sigma_ref <= 0:
        raise HTTPException(status_code=400, detail="sigma_ref must be positive and finite.")
    if requested_skew is None:
        skew = float(recent.skew(skipna=True))
    else:
        skew = float(requested_skew)
    if not np.isfinite(skew):
        skew = 0.0
    return sigma_ref, skew


def _estimate_drift_and_sigma(
    frame: pd.DataFrame,
    timeframe: str,
    methodology: str,
) -> tuple[float, float]:
    step_hours = interval_to_hours(timeframe)
    if methodology == "legacy":
        sigma_ref = frame.attrs.get("_legacy_sigma_ref")
        if sigma_ref is None:
            sigma_ref, _ = _estimate_sigma_ref_and_skew(frame, None)
        drift_per_hour = 0.0
        sigma_per_hour = sigma_ref / np.sqrt(step_hours)
        return drift_per_hour, sigma_per_hour
    if methodology == "research":
        log_returns = frame.attrs.get("_research_log_returns")
        if log_returns is None:
            log_returns = get_log_returns(frame)
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if log_returns.empty:
            raise HTTPException(status_code=400, detail="Cannot compute returns from the provided data.")
        drift_per_step = float(log_returns.mean())
        sigma_per_step = float(log_returns.std(ddof=0))
        if not np.isfinite(sigma_per_step) or sigma_per_step <= 0:
            raise HTTPException(status_code=400, detail="sigma_ref must be positive and finite.")
        drift_per_hour = drift_per_step / step_hours
        sigma_per_hour = sigma_per_step / np.sqrt(step_hours)
        return drift_per_hour, sigma_per_hour
    raise HTTPException(status_code=400, detail="methodology must be 'legacy' or 'research'.")


def _estimate_atr(frame) -> float:
    required = {"high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail="Price frame must include 'high', 'low', and 'close' columns.",
        )
    highs = frame["high"].astype(float)
    lows = frame["low"].astype(float)
    closes = frame["close"].astype(float)
    prev_close = closes.shift(1)
    true_range = pd.concat(
        [
            (highs - lows).abs(),
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range = true_range.replace([np.inf, -np.inf], np.nan).dropna()
    if true_range.empty:
        raise HTTPException(status_code=400, detail="Cannot compute ATR from the provided data.")

    period = 14
    if period > len(true_range):
        period = len(true_range)
    recent = true_range.tail(period)
    atr = float(recent.mean())
    if not np.isfinite(atr) or atr <= 0:
        raise HTTPException(status_code=400, detail="atr must be positive and finite.")
    return atr


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
    atr = _estimate_atr(frame)
    if request.methodology == "research":
        log_returns = get_log_returns(frame)
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if log_returns.empty:
            raise HTTPException(status_code=400, detail="Cannot compute returns from the provided data.")
        frame.attrs["_research_log_returns"] = log_returns
        sigma_ref = float(log_returns.std(ddof=0))
        if not np.isfinite(sigma_ref) or sigma_ref <= 0:
            raise HTTPException(status_code=400, detail="sigma_ref must be positive and finite.")
        if request.skew is None:
            skew = float(log_returns.skew(skipna=True))
        else:
            skew = float(request.skew)
        if not np.isfinite(skew):
            skew = 0.0
    else:
        sigma_ref, skew = _estimate_sigma_ref_and_skew(frame, request.skew)
        frame.attrs["_legacy_sigma_ref"] = sigma_ref

    drift_per_hour, sigma_per_hour = _estimate_drift_and_sigma(
        frame,
        request.timeframe,
        request.methodology,
    )

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
        skew=skew,
        steps=steps,
        step_hours=step_hours,
    )

    if request.methodology == "research":
        cfg = TPFindConfig(max_iter=35, rel_tol=1e-4)
    else:
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
        atr=float(atr),
        entry_price=float(entry_price),
        methodology=request.methodology,
        surface=payload,
    )
