"""AlphaLens inference API for EC2 deployment."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphalens_forecast.config import get_config
from alphalens_forecast.data import DataProvider
from alphalens_forecast.forecasting import FREQ_MAP, ForecastEngine
from alphalens_forecast.models import ModelRouter, NHiTSForecaster
from alphalens_forecast.models.selection import MODEL_TYPES, resolve_device, select_model_type
from alphalens_forecast.utils import TwelveDataError


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_HOST = os.environ.get("ALPHALENS_API_HOST", "0.0.0.0")
DEFAULT_PORT = _env_int("ALPHALENS_API_PORT", 8000)
LOG_LEVEL = os.environ.get("ALPHALENS_LOG_LEVEL", "INFO").upper()
SERVER_START = time.perf_counter()

logger = logging.getLogger("alphalens_api")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_cache_dir() -> Optional[Path]:
    raw = os.environ.get("ALPHALENS_DATA_CACHE_DIR") or os.environ.get("ALPHALENS_CACHE_DIR")
    if not raw:
        return None
    return Path(raw).expanduser()


def coerce_bool(value: Any, default: bool, label: str, warnings: List[str]) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    warnings.append(f"{label} is invalid; using default {default}.")
    return default


def coerce_int(value: Any, default: int, label: str, warnings: List[str]) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        warnings.append(f"{label} is invalid; using default {default}.")
        return default


def parse_horizons(value: Any, default: List[int]) -> Tuple[List[int], List[Any], bool]:
    if value is None:
        return list(default), [], True
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
        items = list(raw_items)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]

    valid: List[int] = []
    invalid: List[Any] = []
    for item in items:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            invalid.append(item)
            continue
        if parsed <= 0:
            invalid.append(item)
            continue
        valid.append(parsed)

    seen: set[int] = set()
    deduped: List[int] = []
    for value_item in valid:
        if value_item in seen:
            continue
        seen.add(value_item)
        deduped.append(value_item)

    return deduped, invalid, False


def inspect_model_assets(
    router: ModelRouter,
    model_type: str,
    symbol: str,
    timeframe: str,
) -> Dict[str, Any]:
    model_dir = router.get_model_dir(model_type, symbol, timeframe)
    metadata_path = router.get_metadata_path(model_type, symbol, timeframe)
    info: Dict[str, Any] = {
        "model_type": model_type,
        "model_dir": str(model_dir),
        "metadata_path": str(metadata_path),
        "metadata_exists": metadata_path.exists(),
    }
    if metadata_path.exists():
        try:
            manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
            info["storage_format"] = manifest.get("storage_format")
            info["class_path"] = manifest.get("class_path")
            model_file = manifest.get("model_file")
            if model_file:
                artifact_path = model_dir / model_file
                info["artifact_path"] = str(artifact_path)
                info["artifact_exists"] = artifact_path.exists()
        except Exception as exc:  # noqa: BLE001
            info["metadata_error"] = str(exc)
    legacy_path = model_dir / "model.pkl"
    info["legacy_path"] = str(legacy_path)
    info["legacy_exists"] = legacy_path.exists()
    return info


def infer_missing_reason(info: Dict[str, Any]) -> str:
    if info.get("metadata_exists") or info.get("artifact_exists") or info.get("legacy_exists"):
        return "load_failed"
    return "not_found"


def load_models(
    config,
    router: ModelRouter,
    symbol: str,
    timeframe: str,
    model_type: str,
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
    device = resolve_device(config.torch_device, model_type)

    mean_info = inspect_model_assets(router, model_type, symbol, timeframe)
    mean_info["device"] = device
    mean_model = router.load_model(model_type, symbol, timeframe, device=device)
    mean_info["loaded"] = mean_model is not None
    if isinstance(mean_model, NHiTSForecaster) and mean_model.requires_retraining():
        mean_info["loaded"] = False
        mean_info["reason"] = "requires_retraining"
        mean_model = None
    if mean_model is None and "reason" not in mean_info:
        mean_info["reason"] = infer_missing_reason(mean_info)

    vol_info = inspect_model_assets(router, "egarch", symbol, timeframe)
    vol_model = router.load_egarch(symbol, timeframe)
    vol_info["loaded"] = vol_model is not None
    if vol_model is None:
        vol_info["reason"] = infer_missing_reason(vol_info)

    status = {
        "model_type": model_type,
        "device": device,
        "mean": mean_info,
        "vol": vol_info,
    }
    return mean_model, vol_model, status


def serialize_predictions(predictions: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for horizon, frame in predictions.items():
        if frame is None or frame.empty:
            payload[horizon] = []
            continue
        df = frame.copy()
        if "ds" not in df.columns:
            df = df.reset_index().rename(columns={"index": "ds"})
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        payload[horizon] = [
            {
                "ds": ts.isoformat().replace("+00:00", "Z"),
                "yhat": float(val),
            }
            for ts, val in zip(df["ds"], df["yhat"])
        ]
    return payload


def build_error_payload(
    *,
    status: str,
    message: str,
    request_id: str,
    errors: Optional[List[str]] = None,
    details: Optional[Dict[str, Any]] = None,
    hint: Optional[str] = None,
    debug: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "status": status,
        "message": message,
        "request_id": request_id,
        "timestamp": utc_now_iso(),
    }
    if errors:
        payload["errors"] = errors
    if details:
        payload["details"] = details
    if hint:
        payload["hint"] = hint
    if debug:
        payload["debug"] = debug
    return payload


def handle_forecast(
    payload: Dict[str, Any],
    *,
    request_id: str,
    debug: bool,
) -> Tuple[int, Dict[str, Any]]:
    start = time.perf_counter()
    config = get_config()
    warnings: List[str] = []
    errors: List[str] = []

    if not isinstance(payload, dict):
        return 400, build_error_payload(
            status="invalid_request",
            message="JSON body must be an object",
            request_id=request_id,
        )

    symbol = str(payload.get("symbol") or config.twelve_data.symbol or "").strip()
    if not symbol:
        errors.append("symbol is required")

    timeframe = (
        str(
            payload.get("timeframe")
            or config.default_timeframe
            or config.twelve_data.interval
            or ""
        )
        .strip()
    )
    if not timeframe:
        errors.append("timeframe is required")
    elif timeframe.lower() not in FREQ_MAP:
        supported = ", ".join(sorted(FREQ_MAP.keys()))
        errors.append(f"unsupported timeframe '{timeframe}'. supported: {supported}")

    horizons, invalid_horizons, used_default = parse_horizons(
        payload.get("horizons"),
        config.monte_carlo.horizon_hours,
    )
    if invalid_horizons:
        warnings.append(f"ignored invalid horizons: {invalid_horizons}")
    if used_default:
        warnings.append(f"horizons not provided; using defaults {list(config.monte_carlo.horizon_hours)}")
    if not horizons:
        errors.append("horizons must contain positive integers")

    use_montecarlo = coerce_bool(
        payload.get("use_montecarlo"),
        config.monte_carlo.use_montecarlo,
        "use_montecarlo",
        warnings,
    )
    paths = coerce_int(payload.get("paths"), config.monte_carlo.paths, "paths", warnings)
    if use_montecarlo and paths <= 0:
        errors.append("paths must be > 0 when use_montecarlo is true")

    include_predictions = coerce_bool(
        payload.get("include_predictions"),
        False,
        "include_predictions",
        warnings,
    )
    include_metadata = coerce_bool(
        payload.get("include_metadata"),
        True,
        "include_metadata",
        warnings,
    )
    include_model_info = coerce_bool(
        payload.get("include_model_info"),
        True,
        "include_model_info",
        warnings,
    )

    model_type_raw = payload.get("model_type")
    if model_type_raw is None:
        model_type = select_model_type(timeframe) if timeframe else ""
    else:
        model_type = str(model_type_raw).strip().lower()
        if model_type not in MODEL_TYPES:
            options = ", ".join(sorted(MODEL_TYPES.keys()))
            errors.append(f"model_type must be one of: {options}")

    if errors:
        return 400, build_error_payload(
            status="invalid_request",
            message="Request validation failed",
            request_id=request_id,
            errors=errors,
        )

    request_context = {
        "symbol": symbol,
        "timeframe": timeframe,
        "horizons": horizons,
        "use_montecarlo": use_montecarlo,
        "paths": paths,
        "model_type": model_type,
        "include_predictions": include_predictions,
        "include_metadata": include_metadata,
        "include_model_info": include_model_info,
    }

    model_router = ModelRouter()
    mean_model, vol_model, model_status = load_models(
        config,
        model_router,
        symbol,
        timeframe,
        model_type,
    )
    if mean_model is None or vol_model is None:
        missing = []
        if mean_model is None:
            missing.append("mean")
        if vol_model is None:
            missing.append("vol")
        hint = (
            "Train and save models using the CLI, then deploy the models directory to this server. "
            "Example: python -m alphalens_forecast.main --symbol "
            f"{symbol} --timeframe {timeframe} --horizons "
            f"{' '.join(str(h) for h in horizons)} --save-models true"
        )
        details = {
            "missing": missing,
            "model_status": model_status,
            "model_dir": str(model_router.base_dir),
        }
        return 404, build_error_payload(
            status="missing_model",
            message="No trained models available for this request",
            request_id=request_id,
            details=details,
            hint=hint,
        )

    data_provider = DataProvider(
        config.twelve_data,
        cache_dir=resolve_cache_dir(),
        auto_refresh=True,
    )
    engine = ForecastEngine(config, data_provider, model_router)

    try:
        result = engine.forecast(
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
            paths=paths,
            use_montecarlo=use_montecarlo,
            show_progress=False,
            mean_model_override=mean_model,
            vol_model_override=vol_model,
        )
    except TwelveDataError as exc:
        return 502, build_error_payload(
            status="data_provider_error",
            message="Market data fetch failed",
            request_id=request_id,
            details={"error": str(exc)},
            hint="Check Twelve Data API credentials and connectivity",
        )
    except ValueError as exc:
        return 422, build_error_payload(
            status="invalid_request",
            message="Forecast failed due to invalid inputs or data",
            request_id=request_id,
            details={"error": str(exc)},
        )
    except Exception as exc:  # noqa: BLE001
        debug_blob = traceback.format_exc() if debug else None
        return 500, build_error_payload(
            status="error",
            message="Unexpected error while running forecast",
            request_id=request_id,
            details={"error": str(exc)},
            debug=debug_blob,
        )

    payload_out: Dict[str, Any] = {
        "ok": True,
        "status": "ok",
        "message": "forecast completed",
        "request_id": request_id,
        "timestamp": utc_now_iso(),
        "request": request_context,
        "warnings": warnings,
        "data": {
            "payload": result.payload,
            "as_of": result.as_of,
            "data_hash": result.data_hash,
            "durations": result.durations,
        },
    }

    if include_metadata:
        payload_out["data"]["metadata"] = result.metadata
    if include_predictions:
        try:
            payload_out["data"]["predictions"] = serialize_predictions(result.predictions)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to serialize predictions: {exc}")
    if include_model_info:
        payload_out["data"]["model_status"] = model_status

    payload_out["data"]["total_seconds"] = round(time.perf_counter() - start, 3)

    return 200, payload_out


class ForecastAPIHandler(BaseHTTPRequestHandler):
    server_version = "AlphaLensForecastAPI/1.0"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/health"}:
            self._send_json(
                200,
                {
                    "ok": True,
                    "status": "ok",
                    "message": "alphalens inference api",
                    "timestamp": utc_now_iso(),
                    "uptime_seconds": round(time.perf_counter() - SERVER_START, 3),
                    "model_dir": str(ModelRouter().base_dir),
                    "cache_dir": str(resolve_cache_dir() or ""),
                    "supported_timeframes": sorted(FREQ_MAP.keys()),
                    "model_types": sorted(MODEL_TYPES.keys()),
                },
                request_id=None,
            )
            return

        self._send_json(
            404,
            build_error_payload(
                status="not_found",
                message="Route not found",
                request_id=str(uuid.uuid4()),
            ),
            request_id=None,
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/forecast":
            self._send_json(
                404,
                build_error_payload(
                    status="not_found",
                    message="Route not found",
                    request_id=str(uuid.uuid4()),
                ),
                request_id=None,
            )
            return

        request_id = str(uuid.uuid4())
        debug = False
        query = parse_qs(parsed.query or "")
        if "debug" in query:
            debug = coerce_bool(query.get("debug", [None])[0], False, "debug", [])

        payload = {}
        content_length = self.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                length = 0
            if length > 0:
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    self._send_json(
                        400,
                        build_error_payload(
                            status="invalid_request",
                            message="Malformed JSON payload",
                            request_id=request_id,
                        ),
                        request_id=request_id,
                    )
                    return

        status_code, response = handle_forecast(payload, request_id=request_id, debug=debug)
        self._send_json(status_code, response, request_id=request_id)

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status_code: int, payload: Dict[str, Any], request_id: Optional[str]) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if request_id:
            self.send_header("X-Request-Id", request_id)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Starting AlphaLens inference API on %s:%s", DEFAULT_HOST, DEFAULT_PORT)
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), ForecastAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
