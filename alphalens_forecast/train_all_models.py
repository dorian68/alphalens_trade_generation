"""CLI helper to train regime baselines per instrument/timeframe/regime."""
from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Optional

from alphalens_forecast.config import AppConfig, get_instrument_universe
from alphalens_forecast.data import DataProvider
from alphalens_forecast.models.router import ModelRouter
from alphalens_forecast.models.training import train_regime_models
from alphalens_forecast.regime_detection.deterministic import (
    REGIME_BREAKOUT,
    REGIME_RANGE,
    REGIME_STRESS_CHOP,
)

logger = logging.getLogger(__name__)


def _parse_csv(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items


def _resolve_regimes(raw: Optional[str]) -> List[str]:
    if raw is None or not raw.strip():
        return [REGIME_RANGE, REGIME_STRESS_CHOP, REGIME_BREAKOUT]
    mapping = {
        "range": REGIME_RANGE,
        "stress": REGIME_STRESS_CHOP,
        "stress_chop": REGIME_STRESS_CHOP,
        "breakout": REGIME_BREAKOUT,
        "breakout_vol_expansion": REGIME_BREAKOUT,
    }
    resolved: List[str] = []
    for entry in _parse_csv(raw):
        key = entry.strip().lower()
        resolved.append(mapping.get(key, entry))
    return resolved


def _iter_instruments(
    instrument_filter: Iterable[str],
    timeframe_filter: Iterable[str],
) -> List[tuple[str, str]]:
    universe = get_instrument_universe()
    wanted_symbols = {sym.upper() for sym in instrument_filter} if instrument_filter else set()
    wanted_tfs = {tf.lower() for tf in timeframe_filter} if timeframe_filter else set()
    pairs: List[tuple[str, str]] = []
    for instrument in universe.instruments:
        symbol_norm = instrument.symbol.upper()
        if wanted_symbols and symbol_norm not in wanted_symbols:
            continue
        for tf in instrument.timeframes:
            if wanted_tfs and tf.lower() not in wanted_tfs:
                continue
            pairs.append((instrument.symbol, tf))
    return pairs


def main() -> None:
    config = AppConfig()
    parser = argparse.ArgumentParser(description="Train regime baselines per instrument/timeframe.")
    parser.add_argument(
        "--enable-per-instrument-models",
        action="store_true",
        default=False,
        help="Train per-instrument regime baselines (flag required for prod routing).",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        default=False,
        help="Force retraining even if a cached model exists.",
    )
    parser.add_argument(
        "--retrain-instruments",
        type=str,
        default="",
        help="Comma-separated list of symbols to train (e.g. EUR/USD,BTC/USD).",
    )
    parser.add_argument(
        "--retrain-timeframes",
        type=str,
        default="",
        help="Comma-separated list of timeframes to train (e.g. 15min,1h).",
    )
    parser.add_argument(
        "--regimes",
        type=str,
        default="",
        help="Comma-separated regime labels (range,stress,breakout). Default trains all.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        default=False,
        help="Refresh market data before training.",
    )
    args = parser.parse_args()

    if not args.enable_per_instrument_models:
        logger.warning(
            "Per-instrument models are disabled by default. "
            "Enable --enable-per-instrument-models to use these models in routing."
        )

    instruments = _parse_csv(args.retrain_instruments)
    timeframes = _parse_csv(args.retrain_timeframes)
    regimes = _resolve_regimes(args.regimes)

    pairs = _iter_instruments(instruments, timeframes)
    if not pairs:
        logger.warning("No instrument/timeframe pairs selected. Nothing to train.")
        return

    provider = DataProvider(auto_refresh=args.refresh_data)
    router = ModelRouter()

    for symbol, timeframe in pairs:
        for regime in regimes:
            logger.info(
                "Training regime baseline for %s @ %s (regime=%s)",
                symbol,
                timeframe,
                regime,
            )
            train_regime_models(
                symbol,
                timeframe,
                regime,
                data_provider=provider,
                model_router=router,
                config=config,
                force_retrain=args.force_retrain,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    main()
