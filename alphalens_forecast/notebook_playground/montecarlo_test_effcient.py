# %%
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from alphalens_forecast.core import montecarlo as mc
from typing import Union
import multiprocessing as mp
import time
from tqdm import tqdm

_SHARED_PRICES = None
_SHARED_ENTRY_PRICE = None
_SHARED_SIGMA_REF = None
_SHARED_X_TARGET_PROBS = None
_SHARED_DIRECTION = None

@dataclass(frozen=True)
class TradeContext:
    """
    Immutable container describing the probabilistic state
    of the market at trade entry.
    """

    current_price: float
    drift: float
    sigma: Union[float, np.ndarray]
    dof: float
    steps: int
    step_hours: float
    skew: float

def find_tp_for_all_target_probs(
    prices,
    *,
    entry_price,
    sl_price,
    target_probs,      # shape (N,)
    sigma_ref,
    direction="long",
    max_iter=35,
    rel_tol=1e-4,
):
    N = len(target_probs)

    # bornes TP initiales
    tp_low = np.full(N, entry_price * (1.0 + 0.05 * sigma_ref))
    tp_high = np.full(N, entry_price * (1.0 + 8.0 * sigma_ref))

    # masque des TP encore actifs
    active = np.ones(N, dtype=bool)

    for _ in range(max_iter):

        if not active.any():
            break

        tp_mid = 0.5 * (tp_low + tp_high)

        probs_mid = prob_hit_tp_before_sl(
            prices,
            tp=tp_mid,
            sl=sl_price,
            direction=direction,
        )

        # condition dichotomie
        too_easy = probs_mid >= target_probs

        tp_low[active & too_easy] = tp_mid[active & too_easy]
        tp_high[active & ~too_easy] = tp_mid[active & ~too_easy]

        # critère d’arrêt
        converged = (np.abs(tp_high - tp_low) / entry_price) < rel_tol
        active &= ~converged

    return 0.5 * (tp_low + tp_high)

def _init_worker(
    prices,
    entry_price,
    sigma_ref,
    x_target_probs,
    direction,
):
    global _SHARED_PRICES
    global _SHARED_ENTRY_PRICE
    global _SHARED_SIGMA_REF
    global _SHARED_X_TARGET_PROBS
    global _SHARED_DIRECTION

    _SHARED_PRICES = prices
    _SHARED_ENTRY_PRICE = entry_price
    _SHARED_SIGMA_REF = sigma_ref
    _SHARED_X_TARGET_PROBS = x_target_probs
    _SHARED_DIRECTION = direction

def _compute_Z_row(sl_sigma):
    a = time.time()
    sl_price = _SHARED_ENTRY_PRICE * (1.0 - sl_sigma * _SHARED_SIGMA_REF)

    tp_vec = find_tp_for_all_target_probs(
        _SHARED_PRICES,
        entry_price=_SHARED_ENTRY_PRICE,
        sl_price=sl_price,
        target_probs=_SHARED_X_TARGET_PROBS,
        sigma_ref=_SHARED_SIGMA_REF,
        direction=_SHARED_DIRECTION,
    )
    print(f"loop took {time.time() - a} seconds", flush=True)
    return (tp_vec / _SHARED_ENTRY_PRICE - 1.0) / _SHARED_SIGMA_REF


def build_surface_Z_mp(
    prices: np.ndarray,
    *,
    entry_price: float,
    sigma_ref: float,
    x_target_probs: np.ndarray,
    y_sl_sigma: np.ndarray,
    direction: str = "long",
    n_workers: int | None = None,
) -> np.ndarray:

    print(f"nb of workers is {mp.cpu_count()}",flush=True)
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    tasks = [
        (
            sl_sigma,
            prices,
            entry_price,
            sigma_ref,
            x_target_probs,
            direction,
        )
        for sl_sigma in y_sl_sigma
    ]

    print("task defined",flush=True)
    rows = []

    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            prices,
            entry_price,
            sigma_ref,
            x_target_probs,
            direction,
        ),
    ) as pool:

        for row in tqdm(
            pool.imap_unordered(_compute_Z_row, y_sl_sigma),
            total=len(y_sl_sigma),
            desc="Building Z surface",
        ):
            rows.append(row)


    return np.vstack(rows)

# -----------------------------
# Helpers: extraction des paths
# -----------------------------
def get_prices_from_mc_result(mc_result: Any) -> np.ndarray:
    """
    Tente d'extraire une matrice de prix (n_paths, n_steps) depuis le retour
    de ton moteur Monte Carlo.

    Adapte cette fonction au format exact de alphalens_forecast.core.montecarlo
    si besoin.
    """
    return mc_result.final_prices


# -----------------------------
# Métriques: probas TP avant SL
# -----------------------------
INF = np.int32(2_000_000_000)

def prob_hit_tp_before_sl_old(
    prices: np.ndarray,
    *,
    tp: float,
    sl: float,
    direction: str = "long",
) -> float:
    """
    Calcule P(hit TP before SL) en respectant l'ordre temporel, à partir des paths.
    prices: (n_paths, n_steps)

    Convention:
    - long: TP = prix >= tp, SL = prix <= sl
    - short: TP = prix <= tp, SL = prix >= sl
    """
    if prices.ndim != 2:
        raise ValueError("prices must be 2D (paths, steps)")

    if direction not in ("long", "short"):
        raise ValueError("direction must be 'long' or 'short'")

    if direction == "long":
        hit_tp = prices >= tp
        hit_sl = prices <= sl
    else:
        hit_tp = prices <= tp
        hit_sl = prices >= sl

    has_tp = hit_tp.any(axis=1)
    has_sl = hit_sl.any(axis=1)

    first_tp = np.where(has_tp, hit_tp.argmax(axis=1).astype(np.int32), INF)
    first_sl = np.where(has_sl, hit_sl.argmax(axis=1).astype(np.int32), INF)

    # TP gagné si touché avant SL
    win = first_tp < first_sl
    return float(win.mean())

def prob_hit_tp_before_sl(prices, tp, sl, direction="long"):
    """
    prices: (paths, steps)
    tp: float ou np.ndarray (N,)
    retourne: float ou np.ndarray (N,)
    """

    prices = prices[:, :, None]          # (paths, steps, 1)
    tp = np.asarray(tp)[None, None, :]   # (1, 1, N)

    if direction == "long":
        hit_tp = prices >= tp
        hit_sl = prices <= sl
    else:
        hit_tp = prices <= tp
        hit_sl = prices >= sl

    first_tp = hit_tp.argmax(axis=1)
    first_sl = hit_sl.argmax(axis=1)

    # gestion "jamais touché"
    first_tp[~hit_tp.any(axis=1)] = np.inf
    first_sl[~hit_sl.any(axis=1)] = np.inf

    return np.mean(first_tp < first_sl, axis=0)

# -----------------------------
# Recherche: TP pour target_prob
# -----------------------------
@dataclass(frozen=True)
class TPFindConfig:
    max_iter: int = 35
    rel_tol: float = 1e-4           # tolérance relative sur le TP
    tp_min_mult: float = 0.05       # en "sigma" / multiplicateur (à adapter)
    tp_max_mult: float = 8.0        # bornes max du TP


def find_tp_for_target_prob(
    prices: np.ndarray,
    *,
    entry_price: float,
    sl_price: float,
    target_prob: float,
    sigma_ref: float,
    direction: str = "long",
    cfg: TPFindConfig = TPFindConfig(),
) -> Optional[float]:
    """
    Trouve un TP tel que P(TP before SL) ≈ target_prob, via dichotomie.
    - Aucun re-MonteCarlo ici : uniquement des évaluations rapides sur `prices`.
    - Retourne le TP (prix) ou None si pas de solution plausible.
    """
    if not (0.0 < target_prob < 1.0):
        raise ValueError("target_prob must be in (0,1)")

    if sigma_ref <= 0:
        raise ValueError("sigma_ref must be > 0")

    # bornes TP en prix à partir de entry_price et sigma_ref
    # NOTE: tu peux remplacer par ta formule exacte (sigma->prix) sans casser l’idée.
    if direction == "long":
        tp_low = entry_price * (1.0 + cfg.tp_min_mult * sigma_ref)
        tp_high = entry_price * (1.0 + cfg.tp_max_mult * sigma_ref)
    else:
        tp_low = entry_price * (1.0 - cfg.tp_min_mult * sigma_ref)
        tp_high = entry_price * (1.0 - cfg.tp_max_mult * sigma_ref)

    # quick sanity: si la proba au TP_high est encore >= target => on sature
    p_high = prob_hit_tp_before_sl(prices, tp=tp_high, sl=sl_price, direction=direction)
    p_low = prob_hit_tp_before_sl(prices, tp=tp_low, sl=sl_price, direction=direction)

    # On veut une proba proche target_prob. Selon monotonicité:
    # - long: plus TP est loin, plus proba baisse (généralement)
    # - short: idem (en valeur absolue)
    # Donc p_low >= p_high en pratique.
    # Si target > p_low: impossible (même TP proche ne suffit pas)
    if target_prob > p_low + 1e-12:
        return None

    # Si target <= p_high: même TP très loin donne encore >= target => tp_high OK
    if target_prob <= p_high + 1e-12:
        return tp_high

    # dichotomie
    best = None
    for _ in range(cfg.max_iter):
        tp_mid = 0.5 * (tp_low + tp_high)
        p_mid = prob_hit_tp_before_sl(prices, tp=tp_mid, sl=sl_price, direction=direction)

        # On cherche p_mid ≈ target_prob.
        # Si p_mid >= target => TP trop proche (trop facile) => éloigner TP
        if p_mid >= target_prob:
            best = tp_mid
            # éloigner TP (augmenter pour long, diminuer pour short)
            if direction == "long":
                tp_low = tp_mid
            else:
                tp_low = tp_mid  # pour short, tp_low est "moins loin" (plus proche de entry)
        else:
            # TP trop loin => rapprocher
            tp_high = tp_mid

        if abs(tp_high - tp_low) / max(entry_price, 1e-12) < cfg.rel_tol:
            break

    return best


# -----------------------------
# Construction de la surface Z
# -----------------------------
def build_surface_Z(
    prices: np.ndarray,
    *,
    entry_price: float,
    sigma_ref: float,
    x_target_probs: np.ndarray,
    y_sl_sigma: np.ndarray,
    direction: str = "long",
) -> np.ndarray:
    """
    Renvoie Z (len(y), len(x)) où chaque cellule = TP_sigma (ou TP_price selon ce que tu veux).
    Ici, je renvoie le TP exprimé en "sigma_ref" relatif au entry_price, pour coller à ton graphique.
    """

    Z = np.empty((len(y_sl_sigma), len(x_target_probs)))

    for i, sl_sigma in enumerate(y_sl_sigma):

        sl_price = entry_price * (1.0 - sl_sigma * sigma_ref)

        # 🔥 ICI : UNE SEULE DICHOTOMIE VECTORISÉE
        tp_vec = find_tp_for_all_target_probs(
            prices,
            entry_price=entry_price,
            sl_price=sl_price,
            target_probs=x_target_probs,
            sigma_ref=sigma_ref,
            direction="long",
        )

        Z[i, :] = (tp_vec / entry_price - 1.0) / sigma_ref

    return Z

# %%
# -----------------------------
# Exemple d'usage (à adapter)
# -----------------------------
if __name__ == "__main__":
    # Ton grid
    x = np.linspace(0.10, 0.99, 30)   # mets 200 pour tester vite, puis 1000
    y = np.linspace(0.02, 10.0, 30)   # idem
    X, Y = np.meshgrid(x, y)

    # Paramètres (exemples)
    entry_price = 1.176893
    sigma_ref = 0.00060  # si c'est une vol "par step", garde cohérent avec ton simulateur
    direction = "long"

    # 1) MONTE CARLO une fois
    mc_simu = mc.MonteCarloSimulator(
        paths=50,
        seed=42,
    )

    ctx = TradeContext(
        current_price=entry_price,
        drift=0.0,
        sigma=0.00060,
        dof=3.0,
        skew=-0.35,
        steps=96,
        step_hours=1,
    )

    # mc_result = mc_simu.simulate(            
    #     current_price=ctx.current_price,
    #     drift=ctx.drift,
    #     sigma=ctx.sigma,
    #     dof=ctx.dof,
    #     skew=ctx.skew,
    #     tp=1.780000,
    #     sl=1.790000,
    #     steps=ctx.steps,
    #     step_hours=ctx.step_hours,
    #     )


    # prices = get_prices_from_mc_result(mc_result)

    prices = mc_simu.simulate_paths(
        current_price=ctx.current_price,
        drift=ctx.drift,
        sigma=ctx.sigma,
        dof=ctx.dof,
        skew=ctx.skew,
        steps=ctx.steps,
        step_hours=ctx.step_hours,
    )

    print("hit 1")
    print(prices)
    print(prices.shape)

    Z = build_surface_Z_mp(
        prices,
        entry_price=entry_price,
        sigma_ref=sigma_ref,
        x_target_probs=x,
        y_sl_sigma=y,
        direction=direction,
        n_workers=4, 
    )

    print("hit 2")
    print(Z)

    print("Z shape:", Z.shape, "nan count:", np.isnan(Z).sum())

# %%
# ==================================================================
#
#                           PLOT CURVE
#
# ==================================================================

import matplotlib.pyplot as plt
from matplotlib import colors

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

Z_masked = np.ma.masked_invalid(Z)
norm = colors.Normalize(vmin=Z_masked.min(), vmax=Z_masked.max())
surf = ax.plot_surface(
    Y, X, Z_masked,
    cmap="plasma",
    norm=norm,
    edgecolor="none",
    antialiased=True
)

ax.set_xlabel("stop-loss (in sigma)")
ax.set_ylabel("target probability")
ax.set_zlabel("take-profit (in sigma)")

ax.view_init(elev=30, azim=140)

fig.colorbar(surf, shrink=0.6, aspect=12, label="take-profit (sigma)")
plt.show()
# %%
