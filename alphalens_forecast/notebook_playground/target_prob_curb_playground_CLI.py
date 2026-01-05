"""Playground entry point for TargetProbabilityCurve."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphalens_forecast.core.target_prob_curve import TPFindConfig, TargetProbabilityCurve


def _parse_float_list(raw: Optional[str]) -> Optional[np.ndarray]:
    if raw is None:
        return None
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise ValueError("List values must be non-empty.")
    return np.asarray([float(item) for item in parts], dtype=float)


def _load_prices(path: Path, key: Optional[str]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if key:
                if key not in archive:
                    raise KeyError(f"Key '{key}' not found in {path}.")
                data = archive[key]
            else:
                data = archive["prices"] if "prices" in archive else archive[archive.files[0]]
    elif suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Unsupported file format. Use .npy, .npz, or .csv.")

    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("Loaded prices must be 2D (paths, steps).")
    if array.shape[1] == 0:
        raise ValueError("Loaded prices must have at least one step.")
    return array


def _build_grid(
    explicit: Optional[np.ndarray],
    *,
    min_val: float,
    max_val: float,
    steps: int,
) -> np.ndarray:
    if explicit is not None:
        return explicit
    if steps <= 1:
        raise ValueError("steps must be greater than 1 when building a grid.")
    return np.linspace(min_val, max_val, steps)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the target probability curve toolkit on saved Monte Carlo paths.",
    )
    parser.add_argument("--prices-path", type=str, required=True)
    parser.add_argument("--prices-key", type=str, default=None)
    parser.add_argument("--entry-price", type=float, required=True)
    parser.add_argument("--sigma-ref", type=float, required=True)
    parser.add_argument("--direction", type=str, default="long", choices=("long", "short"))
    parser.add_argument("--target-probs", type=str, default=None)
    parser.add_argument("--sl-sigma", type=str, default=None)
    parser.add_argument("--target-prob-min", type=float, default=0.10)
    parser.add_argument("--target-prob-max", type=float, default=0.99)
    parser.add_argument("--target-prob-steps", type=int, default=30)
    parser.add_argument("--sl-sigma-min", type=float, default=0.02)
    parser.add_argument("--sl-sigma-max", type=float, default=10.0)
    parser.add_argument("--sl-sigma-steps", type=int, default=30)
    parser.add_argument("--tp-min-mult", type=float, default=0.05)
    parser.add_argument("--tp-max-mult", type=float, default=8.0)
    parser.add_argument("--max-iter", type=int, default=35)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--z-kind", type=str, default="sigma", choices=("sigma", "price"))
    parser.add_argument("--cmap", type=str, default="plasma")
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=140.0)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args(argv)

    prices = _load_prices(Path(args.prices_path), key=args.prices_key)
    target_probs = _build_grid(
        _parse_float_list(args.target_probs),
        min_val=args.target_prob_min,
        max_val=args.target_prob_max,
        steps=args.target_prob_steps,
    )
    sl_sigma = _build_grid(
        _parse_float_list(args.sl_sigma),
        min_val=args.sl_sigma_min,
        max_val=args.sl_sigma_max,
        steps=args.sl_sigma_steps,
    )

    cfg = TPFindConfig(
        max_iter=args.max_iter,
        rel_tol=args.rel_tol,
        tp_min_mult=args.tp_min_mult,
        tp_max_mult=args.tp_max_mult,
    )

    curve = TargetProbabilityCurve(
        prices,
        entry_price=args.entry_price,
        sigma_ref=args.sigma_ref,
        direction=args.direction,
        default_cfg=cfg,
    )

    plot_kwargs = {
        "z_kind": args.z_kind,
        "cmap": args.cmap,
        "elev": args.elev,
        "azim": args.azim,
    }

    surface = curve.build_surface(
        target_probs=target_probs,
        sl_sigma=sl_sigma,
        cfg=cfg,
        n_workers=args.n_workers,
        plot=args.plot,
        plot_kwargs=plot_kwargs,
    )

    if args.output:
        np.savez_compressed(
            args.output,
            target_probs=surface.target_probs,
            sl_sigma=surface.sl_sigma,
            tp_prices=surface.tp_prices,
            tp_sigma=surface.tp_sigma,
        )
        print(f"Saved surface to {args.output}")

    print(
        "surface shape:",
        surface.tp_sigma.shape,
        "nan count:",
        int(np.isnan(surface.tp_sigma).sum()),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
