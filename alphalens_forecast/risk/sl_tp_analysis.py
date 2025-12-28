"""TP/SL quantile analysis utilities for AlphaLens trade forecasts."""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union, Literal, Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import t

Direction = Literal["long", "short"]
SigmaInput = Union[float, Sequence[float], np.ndarray]
QuantileInput = Union[float, Sequence[float], Tuple[float, float]]


def _normalize_direction(direction: str) -> Direction:
    if direction is None:
        raise ValueError("Direction must be provided as 'long' or 'short'.")
    normalized = str(direction).strip().lower()
    if normalized not in {"long", "short"}:
        raise ValueError("Direction must be one of: long, short")
    return normalized  # type: ignore[return-value]


def _validate_quantiles(q_low: float, q_high: float) -> None:
    if not (0.0 < q_low < 1.0) or not (0.0 < q_high < 1.0):
        raise ValueError("Quantiles must be within (0, 1).")
    if q_low >= q_high:
        raise ValueError("q_low must be strictly less than q_high.")


def _resolve_median_log(median_value: float, median_is_log: bool) -> float:
    if median_is_log:
        return float(median_value)
    if median_value <= 0:
        raise ValueError("Median price must be positive to compute log values.")
    return float(np.log(median_value))


def _coerce_sigma_path(sigma_path: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if sigma_path is None:
        return None
    sigma_array = np.asarray(sigma_path, dtype=float)
    if sigma_array.ndim != 1:
        raise ValueError("Sigma path must be one-dimensional.")
    if sigma_array.size == 0:
        raise ValueError("Sigma path must contain at least one value.")
    if np.any(sigma_array < 0):
        raise ValueError("Sigma path values must be non-negative.")
    return sigma_array


def aggregate_sigma(sigma: SigmaInput, horizon: Optional[int] = None) -> float:
    """
    Aggregate a per-step sigma path into a horizon-level sigma.

    Parameters
    ----------
    sigma:
        Scalar sigma or per-step sigma path.
    horizon:
        Optional number of steps to aggregate. When omitted, all steps are used.
    """
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigma_array = np.asarray(sigma, dtype=float)
        if sigma_array.ndim != 1:
            raise ValueError("Sigma path must be one-dimensional.")
        if sigma_array.size == 0:
            raise ValueError("Sigma path must contain at least one value.")
        if np.any(sigma_array < 0):
            raise ValueError("Sigma path values must be non-negative.")
        if horizon is not None:
            if horizon <= 0:
                raise ValueError("Horizon must be positive when aggregating sigma.")
            if horizon > sigma_array.size:
                raise ValueError("Horizon exceeds available sigma path length.")
            sigma_array = sigma_array[:horizon]
        return float(np.sqrt(np.sum(np.square(sigma_array))))
    sigma_value = float(sigma)
    if sigma_value < 0:
        raise ValueError("Sigma must be non-negative.")
    return sigma_value


def compute_sl_tp_from_quantiles(
    last_price: float,
    median_log_price: float,
    sigma_h: SigmaInput,
    dof: float,
    direction: str,
    q_low: float,
    q_high: float,
    *,
    median_is_log: bool = True,
    standardize_t: bool = False,
    horizon: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute SL/TP levels from Student-t quantiles of the log-price distribution.

    Parameters
    ----------
    last_price:
        Entry price for the trade (typically the latest price).
    median_log_price:
        Median log-price forecast (or median price if median_is_log=False).
    sigma_h:
        Aggregated sigma for the horizon, or a per-step sigma path.
    dof:
        Student-t degrees of freedom.
    direction:
        Trade direction ("long" or "short").
    q_low:
        Lower quantile (e.g., 0.2).
    q_high:
        Upper quantile (e.g., 0.8).
    median_is_log:
        Whether median_log_price is already in log space.
    standardize_t:
        If True, rescales the Student-t so sigma_h represents log-price stdev.
    horizon:
        Optional horizon length when aggregating sigma paths.

    Returns
    -------
    dict
        SL/TP levels and distances to entry (absolute and percentage).
    """
    entry_price = float(last_price)
    if entry_price <= 0:
        raise ValueError("last_price must be positive.")
    direction_norm = _normalize_direction(direction)
    _validate_quantiles(q_low, q_high)
    sigma_value = aggregate_sigma(sigma_h, horizon=horizon)
    if dof <= 0:
        raise ValueError("Student-t degrees of freedom must be positive.")

    median_log = _resolve_median_log(median_log_price, median_is_log)
    dist = t(df=dof)
    scale = sigma_value
    if standardize_t:
        if dof <= 2:
            raise ValueError("standardize_t requires dof > 2.")
        scale = sigma_value * np.sqrt((dof - 2.0) / dof)

    low_log = median_log + dist.ppf(q_low) * scale
    high_log = median_log + dist.ppf(q_high) * scale

    low_price = float(np.exp(low_log))
    high_price = float(np.exp(high_log))

    if direction_norm == "long":
        sl = low_price
        tp = high_price
    else:
        sl = high_price
        tp = low_price

    sl_dist_abs = abs(entry_price - sl)
    tp_dist_abs = abs(tp - entry_price)
    sl_dist_pct = sl_dist_abs / entry_price
    tp_dist_pct = tp_dist_abs / entry_price

    return {
        "sl": float(sl),
        "tp": float(tp),
        "entry_price": entry_price,
        "median_price": float(np.exp(median_log)),
        "sl_distance_abs": float(sl_dist_abs),
        "tp_distance_abs": float(tp_dist_abs),
        "sl_distance_pct": float(sl_dist_pct),
        "tp_distance_pct": float(tp_dist_pct),
    }


def probability_hit_tp_before_sl(
    trajectories: np.ndarray,
    tp: float,
    sl: float,
    *,
    direction: Optional[str] = None,
    steps: Optional[int] = None,
) -> float:
    """
    Estimate the probability that TP is hit before SL on simulated price paths.

    Parameters
    ----------
    trajectories:
        Array shaped (paths, steps) with simulated prices.
    tp:
        Take-profit level.
    sl:
        Stop-loss level.
    direction:
        Optional trade direction. If omitted, it is inferred from tp/sl ordering.
    steps:
        Optional steps to consider from each trajectory.
    """
    prices = np.asarray(trajectories, dtype=float)
    if prices.ndim == 1:
        prices = prices[None, :]
    if prices.ndim != 2:
        raise ValueError("trajectories must be a 1D or 2D array of prices.")
    if prices.shape[1] == 0:
        raise ValueError("trajectories must include at least one step.")

    if steps is None:
        steps = prices.shape[1]
    steps = int(steps)
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if steps > prices.shape[1]:
        raise ValueError("steps exceeds trajectory length.")
    prices = prices[:, :steps]

    inferred = direction
    if inferred is None:
        inferred = "long" if tp >= sl else "short"
    direction_norm = _normalize_direction(inferred)

    if direction_norm == "long":
        tp_mask = prices >= tp
        sl_mask = prices <= sl
    else:
        tp_mask = prices <= tp
        sl_mask = prices >= sl

    tp_hit = tp_mask.any(axis=1)
    sl_hit = sl_mask.any(axis=1)
    tp_first = np.where(tp_hit, tp_mask.argmax(axis=1), steps + 1)
    sl_first = np.where(sl_hit, sl_mask.argmax(axis=1), steps + 1)
    hit_tp_before_sl = tp_hit & ((tp_first < sl_first) | ~sl_hit)
    return float(np.mean(hit_tp_before_sl))


def attach_probability_hit_tp_before_sl(
    sl_tp: Mapping[str, float],
    trajectories: Optional[np.ndarray],
    *,
    tp_key: str = "tp",
    sl_key: str = "sl",
    direction: Optional[str] = None,
    steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Attach a Monte Carlo TP/SL hit probability to a SL/TP dict if trajectories exist.

    Parameters
    ----------
    sl_tp:
        Mapping with TP/SL levels.
    trajectories:
        Optional Monte Carlo price paths.
    tp_key:
        Key name for TP in sl_tp.
    sl_key:
        Key name for SL in sl_tp.
    direction:
        Optional trade direction.
    steps:
        Optional steps to evaluate per trajectory.
    """
    result: Dict[str, float] = dict(sl_tp)
    if trajectories is None:
        return result
    if tp_key not in sl_tp or sl_key not in sl_tp:
        raise KeyError("sl_tp must contain tp_key and sl_key values.")
    probability = probability_hit_tp_before_sl(
        trajectories,
        tp=float(sl_tp[tp_key]),
        sl=float(sl_tp[sl_key]),
        direction=direction,
        steps=steps,
    )
    result["probability_hit_tp_before_sl"] = probability
    return result


def interpret_sl_tp(
    last_price: float,
    sl: float,
    tp: float,
    sigma_h: float,
    dof: float,
    direction: str,
    *,
    tight_sigma: float = 1.0,
    loose_sigma: float = 2.5,
    ambitious_sigma: float = 2.0,
) -> Dict[str, object]:
    """
    Provide interpretability metrics for a TP/SL pair.

    Returns
    -------
    dict
        sl_in_sigma, tp_in_sigma, fat_tail_risk, and tight/ambition flags.
    """
    _ = _normalize_direction(direction)
    entry_price = float(last_price)
    if entry_price <= 0 or sl <= 0 or tp <= 0:
        raise ValueError("Prices must be positive to interpret TP/SL.")
    entry_log = np.log(entry_price)
    sl_log_dist = abs(np.log(sl) - entry_log)
    tp_log_dist = abs(np.log(tp) - entry_log)

    if sigma_h <= 0:
        sl_in_sigma = float("inf") if sl_log_dist > 0 else 0.0
        tp_in_sigma = float("inf") if tp_log_dist > 0 else 0.0
    else:
        sl_in_sigma = sl_log_dist / sigma_h
        tp_in_sigma = tp_log_dist / sigma_h

    if dof <= 4:
        fat_tail_risk = "high"
    elif dof <= 8:
        fat_tail_risk = "medium"
    else:
        fat_tail_risk = "low"

    tight_threshold = tight_sigma * (1.2 if dof <= 4 else 1.0)
    sl_too_tight = sl_in_sigma < tight_threshold
    sl_too_loose = sl_in_sigma > loose_sigma
    tp_ambitious = tp_in_sigma > ambitious_sigma
    tp_conservative = tp_in_sigma < tight_sigma

    return {
        "sl_in_sigma": float(sl_in_sigma),
        "tp_in_sigma": float(tp_in_sigma),
        "fat_tail_risk": fat_tail_risk,
        "sl_too_tight": bool(sl_too_tight),
        "sl_too_loose": bool(sl_too_loose),
        "tp_ambitious": bool(tp_ambitious),
        "tp_conservative": bool(tp_conservative),
    }


def _get_param(base: object, key: str, default: Optional[Any] = None) -> Any:
    if isinstance(base, Mapping):
        return base.get(key, default)
    return getattr(base, key, default)


def _resolve_quantile_pair(value: QuantileInput) -> Tuple[float, float]:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        q_low, q_high = float(value[0]), float(value[1])
    else:
        q_val = float(value)
        if q_val < 0.5:
            q_low, q_high = q_val, 1.0 - q_val
        else:
            q_low, q_high = 1.0 - q_val, q_val
    _validate_quantiles(q_low, q_high)
    return q_low, q_high


def _sigma_for_horizon(
    *,
    base_sigma_h: float,
    sigma_path: Optional[np.ndarray],
    sigma_per_step: Optional[float],
    base_horizon: Optional[int],
    horizon: int,
) -> float:
    if sigma_path is not None:
        return aggregate_sigma(sigma_path, horizon=horizon)
    if sigma_per_step is not None:
        return float(sigma_per_step) * np.sqrt(horizon)
    if base_horizon and base_horizon > 0:
        return float(base_sigma_h) * np.sqrt(horizon / base_horizon)
    return float(base_sigma_h)


def _median_log_for_horizon(
    *,
    base_median_log: float,
    last_price: float,
    base_horizon: Optional[int],
    horizon: int,
) -> float:
    if base_horizon and base_horizon > 0:
        entry_log = np.log(last_price)
        drift = (base_median_log - entry_log) / base_horizon
        return float(entry_log + drift * horizon)
    return float(base_median_log)


def analyze_sl_tp_sensitivity(
    base_trade_params: object,
    vary: Literal["sigma", "dof", "horizon", "quantiles"],
    grid: Iterable[QuantileInput],
) -> pd.DataFrame:
    """
    Analyze how SL/TP levels change as model parameters vary.

    Parameters
    ----------
    base_trade_params:
        Trade-like object or mapping with fields such as last_price, median_log_price or
        median, sigma_h/sigma_path, dof, horizon, direction, q_low, q_high.
    vary:
        Which parameter to vary: "sigma", "dof", "horizon", or "quantiles".
    grid:
        Values to test. For "sigma" this is treated as a scale multiplier.
        For "quantiles", each entry can be a float (symmetric tails) or (q_low, q_high).
    """
    last_price = _get_param(base_trade_params, "last_price", _get_param(base_trade_params, "entry_price"))
    if last_price is None:
        raise ValueError("base_trade_params must provide last_price or entry_price.")
    last_price = float(last_price)
    if last_price <= 0:
        raise ValueError("last_price must be positive.")

    median_log = _get_param(base_trade_params, "median_log_price")
    if median_log is None:
        median_price = _get_param(base_trade_params, "median_price", _get_param(base_trade_params, "median"))
        if median_price is None:
            raise ValueError("base_trade_params must provide median_log_price or median price.")
        median_log = _resolve_median_log(float(median_price), median_is_log=False)
    else:
        median_log = _resolve_median_log(float(median_log), median_is_log=True)

    dof = _get_param(base_trade_params, "dof", _get_param(base_trade_params, "degrees_of_freedom"))
    if dof is None:
        raise ValueError("base_trade_params must provide dof or degrees_of_freedom.")
    dof = float(dof)

    horizon = _get_param(base_trade_params, "horizon", _get_param(base_trade_params, "steps"))
    if horizon is not None:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("horizon must be positive.")

    sigma_path = _coerce_sigma_path(_get_param(base_trade_params, "sigma_path"))
    sigma_per_step = _get_param(base_trade_params, "sigma_per_step")
    if sigma_per_step is not None:
        sigma_per_step = float(sigma_per_step)

    sigma_h_input = _get_param(base_trade_params, "sigma_h", _get_param(base_trade_params, "sigma"))
    if sigma_path is None and sigma_per_step is None and sigma_h_input is None:
        raise ValueError("base_trade_params must provide sigma_h, sigma_path, or sigma_per_step.")

    if sigma_path is not None:
        base_sigma_h = aggregate_sigma(sigma_path, horizon=horizon)
    elif sigma_per_step is not None:
        if horizon is None:
            raise ValueError("sigma_per_step requires a base horizon.")
        base_sigma_h = float(sigma_per_step) * np.sqrt(horizon)
    else:
        base_sigma_h = aggregate_sigma(sigma_h_input, horizon=horizon)

    q_low = _get_param(base_trade_params, "q_low", 0.2)
    q_high = _get_param(base_trade_params, "q_high", 0.8)
    _validate_quantiles(float(q_low), float(q_high))

    direction = _get_param(base_trade_params, "direction")
    if direction is None:
        median_price = float(np.exp(median_log))
        direction = "long" if median_price >= last_price else "short"
    direction = _normalize_direction(str(direction))

    standardize_t = bool(_get_param(base_trade_params, "standardize_t", False))

    rows = []
    for value in grid:
        sigma_h = base_sigma_h
        dof_value = dof
        horizon_value = horizon
        q_low_value = float(q_low)
        q_high_value = float(q_high)
        median_log_value = median_log
        vary_value: float

        if vary == "sigma":
            vary_value = float(value)
            sigma_h = base_sigma_h * vary_value
        elif vary == "dof":
            vary_value = float(value)
            dof_value = float(value)
        elif vary == "horizon":
            vary_value = float(value)
            horizon_value = int(value)
            if horizon_value <= 0:
                raise ValueError("horizon values must be positive.")
            sigma_h = _sigma_for_horizon(
                base_sigma_h=base_sigma_h,
                sigma_path=sigma_path,
                sigma_per_step=sigma_per_step,
                base_horizon=horizon,
                horizon=horizon_value,
            )
            median_log_value = _median_log_for_horizon(
                base_median_log=median_log,
                last_price=last_price,
                base_horizon=horizon,
                horizon=horizon_value,
            )
        elif vary == "quantiles":
            q_low_value, q_high_value = _resolve_quantile_pair(value)
            vary_value = float(q_high_value)
        else:
            raise ValueError("vary must be one of: sigma, dof, horizon, quantiles")

        sl_tp = compute_sl_tp_from_quantiles(
            last_price=last_price,
            median_log_price=median_log_value,
            sigma_h=sigma_h,
            dof=dof_value,
            direction=direction,
            q_low=q_low_value,
            q_high=q_high_value,
            median_is_log=True,
            standardize_t=standardize_t,
            horizon=horizon_value,
        )
        denominator = max(sl_tp["sl_distance_abs"], 1e-12)
        tp_sl_ratio = sl_tp["tp_distance_abs"] / denominator
        rows.append(
            {
                "vary": vary,
                "value": float(vary_value),
                "direction": direction,
                "entry_price": float(last_price),
                "median_price": float(np.exp(median_log_value)),
                "sigma_h": float(sigma_h),
                "dof": float(dof_value),
                "horizon": float(horizon_value) if horizon_value is not None else float("nan"),
                "q_low": float(q_low_value),
                "q_high": float(q_high_value),
                "sl": sl_tp["sl"],
                "tp": sl_tp["tp"],
                "sl_distance_abs": sl_tp["sl_distance_abs"],
                "tp_distance_abs": sl_tp["tp_distance_abs"],
                "sl_distance_pct": sl_tp["sl_distance_pct"],
                "tp_distance_pct": sl_tp["tp_distance_pct"],
                "tp_sl_ratio": float(tp_sl_ratio),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "aggregate_sigma",
    "compute_sl_tp_from_quantiles",
    "analyze_sl_tp_sensitivity",
    "probability_hit_tp_before_sl",
    "attach_probability_hit_tp_before_sl",
    "interpret_sl_tp",
]
