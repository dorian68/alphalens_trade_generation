"""Ready-to-run TargetProbabilityCurve script (no CLI)."""
# %%
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphalens_forecast.config import AppConfig, get_config
from alphalens_forecast.core import (
    MonteCarloSimulator,
    TPFindConfig,
    TargetProbabilityCurve,
    get_log_returns,
    horizon_to_steps,
    interval_to_hours,
)
from alphalens_forecast.data import DataProvider


@dataclass(frozen=True)
class RunConfig:
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    refresh: bool = False
    max_points: Optional[int] = None
    range_cache: str = "none"
    data_cache_dir: Optional[Path] = None
    auto_refresh: bool = False
    paths: int = 3000
    steps: int = 96
    horizon_hours: Optional[int] = None
    seed: Optional[int] = None
    show_progress: bool = False
    dof: float = 5.0
    skew: Optional[float] = None
    entry_price: Optional[float] = None
    sigma_ref: Optional[float] = None
    drift_per_step: Optional[float] = None
    direction: str = "long"
    target_probs: Optional[Sequence[float]] = None
    sl_sigma: Optional[Sequence[float]] = None
    target_prob_min: float = 0.10
    target_prob_max: float = 0.99
    target_prob_steps: int = 30
    sl_sigma_min: float = 0.02
    sl_sigma_max: float = 10.0
    sl_sigma_steps: int = 30
    tp_min_mult: float = 0.05
    tp_max_mult: float = 8.0
    max_iter: int = 35
    rel_tol: float = 1e-4
    n_workers: Optional[int] = None
    plot: bool = False
    z_kind: str = "sigma"
    cmap: str = "plasma"
    elev: float = 30.0
    azim: float = 140.0
    output: Optional[Path] = None


RUN_CONFIG = RunConfig(
    symbol="XLM/USD",
    timeframe="15min",
    horizon_hours=72,
    paths=1000,
    output=Path("tp_surface.npz"),
    plot=True,
    # skew=-0.25,
)


def _build_grid(
    explicit: Optional[Sequence[float]],
    *,
    min_val: float,
    max_val: float,
    steps: int,
) -> np.ndarray:
    if explicit is not None:
        values = np.asarray(list(explicit), dtype=float)
        if values.size == 0:
            raise ValueError("List values must be non-empty.")
        return values
    if steps <= 1:
        raise ValueError("steps must be greater than 1 when building a grid.")
    return np.linspace(min_val, max_val, steps)


def _resolve_symbol_timeframe(config: RunConfig, app_config: AppConfig) -> tuple[str, str]:
    symbol = config.symbol or app_config.twelve_data.symbol
    timeframe = config.timeframe or app_config.twelve_data.interval
    if not symbol:
        raise ValueError("symbol must not be empty.")
    if not timeframe:
        raise ValueError("timeframe must not be empty.")
    return symbol, timeframe


def _resolve_steps(config: RunConfig, timeframe: str) -> int:
    if config.horizon_hours is not None:
        steps = horizon_to_steps(int(config.horizon_hours), timeframe)
    else:
        steps = int(config.steps)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    return steps


def _load_price_frame(
    config: RunConfig,
    app_config: AppConfig,
    symbol: str,
    timeframe: str,
):
    cache_dir = config.data_cache_dir.expanduser() if config.data_cache_dir else None
    provider = DataProvider(
        app_config.twelve_data,
        cache_dir=cache_dir,
        auto_refresh=config.auto_refresh,
    )
    frame = provider.load_data(
        symbol,
        timeframe,
        refresh=config.refresh,
        max_points=config.max_points,
        start=config.start,
        end=config.end,
        range_cache=config.range_cache,
    )
    if frame.empty:
        raise ValueError("No data returned by DataProvider.")
    return frame.sort_index()


def _resolve_entry_price(frame, config: RunConfig) -> float:
    if config.entry_price is not None:
        entry_price = float(config.entry_price)
    else:
        if "close" not in frame.columns:
            raise ValueError("Price frame must include a 'close' column.")
        entry_price = float(frame["close"].iloc[-1])
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price must be positive and finite.")
    return entry_price


def _estimate_return_stats(frame, config: RunConfig, step_hours: float) -> tuple[float, float, float, float]:
    log_returns = get_log_returns(frame)
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if log_returns.empty:
        raise ValueError("Cannot compute returns from the provided data.")
    drift_per_step = float(config.drift_per_step) if config.drift_per_step is not None else float(log_returns.mean())
    sigma_per_step = float(config.sigma_ref) if config.sigma_ref is not None else float(log_returns.std())
    if not np.isfinite(sigma_per_step) or sigma_per_step <= 0:
        raise ValueError("sigma_ref must be positive and finite.")
    if config.skew is None:
        skew = float(log_returns.skew(skipna=True))
    else:
        skew = float(config.skew)
    if not np.isfinite(skew):
        skew = 0.0
    drift_per_hour = drift_per_step / step_hours
    sigma_per_hour = sigma_per_step / np.sqrt(step_hours)
    return drift_per_hour, sigma_per_hour, sigma_per_step, skew


def _simulate_price_paths(
    *,
    config: RunConfig,
    entry_price: float,
    drift_per_hour: float,
    sigma_per_hour: float,
    skew: float,
    steps: int,
    step_hours: float,
) -> np.ndarray:
    simulator = MonteCarloSimulator(
        paths=config.paths,
        seed=config.seed,
        show_progress=config.show_progress,
    )
    return simulator.simulate_paths(
        current_price=entry_price,
        drift=drift_per_hour,
        sigma=sigma_per_hour,
        dof=config.dof,
        skew=skew,
        steps=steps,
        step_hours=step_hours,
    )


def run(config: RunConfig) -> np.ndarray:
    app_config = get_config()
    symbol, timeframe = _resolve_symbol_timeframe(config, app_config)
    steps = _resolve_steps(config, timeframe)
    step_hours = interval_to_hours(timeframe)

    frame = _load_price_frame(config, app_config, symbol, timeframe)
    entry_price = _resolve_entry_price(frame, config)
    drift_per_hour, sigma_per_hour, sigma_ref, skew = _estimate_return_stats(frame, config, step_hours)
    print(f"drift_per_hour, sigma_per_hour, sigma_ref = ({drift_per_hour}, {sigma_per_hour}, {sigma_ref})")

    prices = _simulate_price_paths(
        config=config,
        entry_price=entry_price,
        drift_per_hour=drift_per_hour,
        sigma_per_hour=sigma_per_hour,
        skew=skew,
        steps=steps,
        step_hours=step_hours,
    )

    target_probs = _build_grid(
        config.target_probs,
        min_val=config.target_prob_min,
        max_val=config.target_prob_max,
        steps=config.target_prob_steps,
    )
    sl_sigma = _build_grid(
        config.sl_sigma,
        min_val=config.sl_sigma_min,
        max_val=config.sl_sigma_max,
        steps=config.sl_sigma_steps,
    )

    cfg = TPFindConfig(
        max_iter=config.max_iter,
        rel_tol=config.rel_tol,
        tp_min_mult=config.tp_min_mult,
        tp_max_mult=config.tp_max_mult,
    )

    curve = TargetProbabilityCurve(
        prices,
        entry_price=entry_price,
        sigma_ref=sigma_ref,
        direction=config.direction,
        default_cfg=cfg,
    )

    surface = curve.build_surface(
        target_probs=target_probs,
        sl_sigma=sl_sigma,
        cfg=cfg,
        n_workers=config.n_workers,
        plot=False,
    )
    if config.plot:
        curve.plot_surface(
            surface,
            z_kind=config.z_kind,
            cmap=config.cmap,
            elev=config.elev,
            azim=config.azim,
        )

    if config.output is not None:
        output_path = config.output
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        np.savez_compressed(
            output_path,
            target_probs=surface.target_probs,
            sl_sigma=surface.sl_sigma,
            tp_prices=surface.tp_prices,
            tp_sigma=surface.tp_sigma,
        )
        print(f"Saved surface to {output_path}")

    print(
        "surface shape:",
        surface.tp_sigma.shape,
        "nan count:",
        int(np.isnan(surface.tp_sigma).sum()),
    )
    return surface.tp_sigma


def main() -> np.ndarray:
    return run(RUN_CONFIG)


if __name__ == "__main__":
    main()

# %%
                                                                   
import numpy as np                                                    
from alphalens_forecast.core.target_prob_curve import TargetProbSurface, TargetProbabilityCurve                             
                                                                    
data = np.load(r"C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation\alphalens_forecast\notebook_playground\tp_surface_2.npz")                                                      
surface = TargetProbSurface(                                          
    target_probs=data["target_probs"],                                
    sl_sigma=data["sl_sigma"],                                        
    tp_prices=data["tp_prices"],                                      
    tp_sigma=data["tp_sigma"],                                        
)                                                                     
                                                                    
# Instance “dummy” juste pour appeler plot_surface                    
curve = TargetProbabilityCurve(prices=np.ones((1, 2)),                
entry_price=1.0, sigma_ref=1.0)                                       
curve.plot_surface(surface, z_kind="sigma", cmap="plasma", elev=30,   
azim=140)                                                             
    
# %%
