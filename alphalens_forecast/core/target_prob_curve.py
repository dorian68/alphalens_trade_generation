"""Target probability curve utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import numpy as np

Direction = Literal["long", "short"]


def _normalize_direction(direction: str) -> Direction:
    if direction is None:
        raise ValueError("direction must be provided as 'long' or 'short'.")
    normalized = str(direction).strip().lower()
    if normalized not in {"long", "short"}:
        raise ValueError("direction must be one of: long, short")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class TPFindConfig:
    max_iter: int = 35
    rel_tol: float = 1e-4
    tp_min_mult: float = 0.05
    tp_max_mult: float = 8.0

    def validate(self) -> None:
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.rel_tol <= 0:
            raise ValueError("rel_tol must be positive.")
        if self.tp_min_mult <= 0 or self.tp_max_mult <= 0:
            raise ValueError("tp_min_mult and tp_max_mult must be positive.")
        if self.tp_min_mult >= self.tp_max_mult:
            raise ValueError("tp_min_mult must be less than tp_max_mult.")


@dataclass(frozen=True)
class TargetProbSurface:
    target_probs: np.ndarray
    sl_sigma: np.ndarray
    tp_prices: np.ndarray
    tp_sigma: np.ndarray


def _ensure_1d(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _initial_tp_bounds(
    *,
    entry_price: float,
    sigma_ref: float,
    direction: Direction,
    cfg: TPFindConfig,
) -> Tuple[float, float]:
    if sigma_ref <= 0:
        raise ValueError("sigma_ref must be positive.")

    if direction == "long":
        near = entry_price * (1.0 + cfg.tp_min_mult * sigma_ref)
        far = entry_price * (1.0 + cfg.tp_max_mult * sigma_ref)
    else:
        near = entry_price * (1.0 - cfg.tp_min_mult * sigma_ref)
        far = entry_price * (1.0 - cfg.tp_max_mult * sigma_ref)

    if near <= 0 or far <= 0:
        raise ValueError("TP bounds must stay positive. Check sigma_ref and bounds.")

    return near, far


def _sl_price_from_sigma(
    *,
    entry_price: float,
    sigma_ref: float,
    sl_sigma: float,
    direction: Direction,
) -> float:
    if direction == "long":
        sl_price = entry_price * (1.0 - sl_sigma * sigma_ref)
    else:
        sl_price = entry_price * (1.0 + sl_sigma * sigma_ref)
    if sl_price <= 0:
        raise ValueError("SL price must be positive. Check sl_sigma and sigma_ref.")
    return sl_price


def _prob_hit_tp_before_sl(
    prices: np.ndarray,
    *,
    tp: np.ndarray,
    sl: float,
    direction: Direction,
) -> np.ndarray:
    prices = prices[:, :, None]
    tp = np.asarray(tp, dtype=float)[None, None, :]

    if direction == "long":
        hit_tp = prices >= tp
        hit_sl = prices <= sl
    else:
        hit_tp = prices <= tp
        hit_sl = prices >= sl

    first_tp = hit_tp.argmax(axis=1).astype(float)
    first_sl = hit_sl.argmax(axis=1).astype(float)

    tp_any = hit_tp.any(axis=1)
    sl_any = hit_sl.any(axis=1)
    first_tp[~tp_any] = np.inf
    first_sl[~sl_any] = np.inf

    return np.mean(first_tp < first_sl, axis=0)


def _find_tp_for_all_target_probs(
    prices: np.ndarray,
    *,
    entry_price: float,
    sigma_ref: float,
    target_probs: np.ndarray,
    sl_price: float,
    direction: Direction,
    cfg: TPFindConfig,
) -> np.ndarray:
    cfg.validate()
    if not np.all((target_probs > 0.0) & (target_probs < 1.0)):
        raise ValueError("target_probs must be in (0, 1).")

    tp_near, tp_far = _initial_tp_bounds(
        entry_price=entry_price,
        sigma_ref=sigma_ref,
        direction=direction,
        cfg=cfg,
    )

    tp_near = np.full_like(target_probs, tp_near, dtype=float)
    tp_far = np.full_like(target_probs, tp_far, dtype=float)
    active = np.ones_like(target_probs, dtype=bool)

    for _ in range(cfg.max_iter):
        if not active.any():
            break

        tp_mid = 0.5 * (tp_near + tp_far)
        probs_mid = _prob_hit_tp_before_sl(
            prices,
            tp=tp_mid,
            sl=sl_price,
            direction=direction,
        )

        too_easy = probs_mid >= target_probs
        tp_near[active & too_easy] = tp_mid[active & too_easy]
        tp_far[active & ~too_easy] = tp_mid[active & ~too_easy]

        converged = (np.abs(tp_far - tp_near) / max(entry_price, 1e-12)) < cfg.rel_tol
        active &= ~converged

    return 0.5 * (tp_near + tp_far)


_MP_STATE: dict[str, object] = {}


def _init_mp_state(
    prices: np.ndarray,
    entry_price: float,
    sigma_ref: float,
    target_probs: np.ndarray,
    direction: Direction,
    cfg: TPFindConfig,
) -> None:
    global _MP_STATE
    _MP_STATE = {
        "prices": prices,
        "entry_price": entry_price,
        "sigma_ref": sigma_ref,
        "target_probs": target_probs,
        "direction": direction,
        "cfg": cfg,
    }


def _compute_tp_row(sl_sigma: float) -> np.ndarray:
    state = _MP_STATE
    sl_price = _sl_price_from_sigma(
        entry_price=float(state["entry_price"]),
        sigma_ref=float(state["sigma_ref"]),
        sl_sigma=float(sl_sigma),
        direction=state["direction"],
    )
    return _find_tp_for_all_target_probs(
        state["prices"],
        entry_price=float(state["entry_price"]),
        sigma_ref=float(state["sigma_ref"]),
        target_probs=state["target_probs"],
        sl_price=sl_price,
        direction=state["direction"],
        cfg=state["cfg"],
    )


class TargetProbabilityCurve:
    """Compute TP levels that satisfy target hit probabilities."""

    def __init__(
        self,
        prices: np.ndarray,
        *,
        entry_price: float,
        sigma_ref: float,
        direction: str = "long",
        default_cfg: Optional[TPFindConfig] = None,
    ) -> None:
        array = np.asarray(prices, dtype=float)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError("prices must be 2D (paths, steps).")
        if array.shape[1] == 0:
            raise ValueError("prices must have at least one step.")
        if entry_price <= 0:
            raise ValueError("entry_price must be positive.")
        if sigma_ref <= 0:
            raise ValueError("sigma_ref must be positive.")

        self._prices = array
        self._entry_price = float(entry_price)
        self._sigma_ref = float(sigma_ref)
        self._direction: Direction = _normalize_direction(direction)
        self._default_cfg = default_cfg or TPFindConfig()

    @property
    def prices(self) -> np.ndarray:
        return self._prices

    @property
    def entry_price(self) -> float:
        return self._entry_price

    @property
    def sigma_ref(self) -> float:
        return self._sigma_ref

    @property
    def direction(self) -> Direction:
        return self._direction

    def find_tp_for_target_probs(
        self,
        *,
        sl_price: float,
        target_probs: Sequence[float],
        cfg: Optional[TPFindConfig] = None,
    ) -> np.ndarray:
        probs = _ensure_1d("target_probs", target_probs)
        if sl_price <= 0:
            raise ValueError("sl_price must be positive.")
        cfg = cfg or self._default_cfg
        return _find_tp_for_all_target_probs(
            self._prices,
            entry_price=self._entry_price,
            sigma_ref=self._sigma_ref,
            target_probs=probs,
            sl_price=float(sl_price),
            direction=self._direction,
            cfg=cfg,
        )

    def find_tp_for_target_prob(
        self,
        *,
        sl_price: float,
        target_prob: float,
        cfg: Optional[TPFindConfig] = None,
    ) -> float:
        tp_prices = self.find_tp_for_target_probs(
            sl_price=sl_price,
            target_probs=[target_prob],
            cfg=cfg,
        )
        return float(tp_prices[0])

    def build_surface(
        self,
        *,
        target_probs: Sequence[float],
        sl_sigma: Sequence[float],
        cfg: Optional[TPFindConfig] = None,
        n_workers: Optional[int] = None,
        plot: bool = False,
        plot_kwargs: Optional[dict] = None,
    ) -> TargetProbSurface:
        x_probs = _ensure_1d("target_probs", target_probs)
        y_sigma = _ensure_1d("sl_sigma", sl_sigma)

        if not np.all((x_probs > 0.0) & (x_probs < 1.0)):
            raise ValueError("target_probs must be in (0, 1).")
        if not np.all(y_sigma > 0.0):
            raise ValueError("sl_sigma must be positive.")

        cfg = cfg or self._default_cfg
        cfg.validate()

        if n_workers is not None and n_workers > 1:
            import multiprocessing as mp

            with mp.Pool(
                processes=n_workers,
                initializer=_init_mp_state,
                initargs=(
                    self._prices,
                    self._entry_price,
                    self._sigma_ref,
                    x_probs,
                    self._direction,
                    cfg,
                ),
            ) as pool:
                rows = pool.map(_compute_tp_row, y_sigma)
            tp_prices = np.vstack(rows)
        else:
            rows = []
            for sl_value in y_sigma:
                sl_price = _sl_price_from_sigma(
                    entry_price=self._entry_price,
                    sigma_ref=self._sigma_ref,
                    sl_sigma=float(sl_value),
                    direction=self._direction,
                )
                tp_row = _find_tp_for_all_target_probs(
                    self._prices,
                    entry_price=self._entry_price,
                    sigma_ref=self._sigma_ref,
                    target_probs=x_probs,
                    sl_price=sl_price,
                    direction=self._direction,
                    cfg=cfg,
                )
                rows.append(tp_row)
            tp_prices = np.vstack(rows)

        tp_sigma = self._tp_prices_to_sigma(tp_prices)
        surface = TargetProbSurface(
            target_probs=x_probs,
            sl_sigma=y_sigma,
            tp_prices=tp_prices,
            tp_sigma=tp_sigma,
        )

        if plot:
            self.plot_surface(surface, **(plot_kwargs or {}))

        return surface

    def plot_surface(
        self,
        surface: TargetProbSurface,
        *,
        z_kind: str = "sigma",
        figsize: Tuple[float, float] = (9.0, 7.0),
        cmap: str = "plasma",
        elev: float = 30.0,
        azim: float = 140.0,
        show: bool = True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import colors

        z_kind_normalized = z_kind.strip().lower()
        if z_kind_normalized == "sigma":
            z_data = surface.tp_sigma
            z_label = "take-profit (in sigma)"
        elif z_kind_normalized == "price":
            z_data = surface.tp_prices
            z_label = "take-profit (price)"
        else:
            raise ValueError("z_kind must be 'sigma' or 'price'.")

        finite = np.isfinite(z_data)
        if not np.any(finite):
            raise ValueError("Surface has no finite values to plot.")

        x_grid, y_grid = np.meshgrid(surface.target_probs, surface.sl_sigma)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        z_masked = np.ma.masked_invalid(z_data)
        norm = colors.Normalize(vmin=z_masked.min(), vmax=z_masked.max())
        surf = ax.plot_surface(
            y_grid,
            x_grid,
            z_masked,
            cmap=cmap,
            norm=norm,
            edgecolor="none",
            antialiased=True,
        )

        ax.set_xlabel("stop-loss (in sigma)")
        ax.set_ylabel("target probability")
        ax.set_zlabel(z_label)
        ax.view_init(elev=elev, azim=azim)

        fig.colorbar(surf, shrink=0.6, aspect=12, label=z_label)
        if show:
            plt.show()

        return fig, ax

    def _tp_prices_to_sigma(self, tp_prices: np.ndarray) -> np.ndarray:
        if self._direction == "long":
            return (tp_prices / self._entry_price - 1.0) / self._sigma_ref
        return (1.0 - tp_prices / self._entry_price) / self._sigma_ref


__all__ = ["TargetProbabilityCurve", "TPFindConfig", "TargetProbSurface"]
