"""Read-only immutable context construction for V-Sense."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from furnace_data.config import load_config
from furnace_data.dataset.static_csv import get_static_dataset_path
from furnace_data.runtime_paths import get_repo_root
from furnace_data.vsense.catalog import (
    ALGORITHM_VERSION,
    CATALOG_VERSION,
    DISPLAY_TIMEZONE,
    control_parameter_by_feature,
    feature_for_parameter_id,
    load_vsense_catalog,
    optimization_by_id,
    parameter_by_id,
    target_for_optimization,
)
from furnace_data.vsense.model_bundle import model_readiness


class VSenseContextError(RuntimeError):
    """Context construction failure with an API-stable code."""

    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(message)


def build_context_snapshot(
    *,
    optimization_type_id: str,
    data_mode: str = "live",
    context_id: str | None = None,
    owner_user_id: str | None = None,
    now: datetime | None = None,
    ttl_seconds: int = 1800,
    catalog: dict[str, Any] | None = None,
    control_profile: dict[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    live_values: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a server-owned immutable context from trusted data sources."""

    now_utc = _aware_utc(now)
    catalog_data = catalog or load_vsense_catalog(context_ttl_seconds=ttl_seconds)
    optimizations = optimization_by_id(catalog_data)
    if optimization_type_id not in optimizations:
        raise VSenseContextError(
            "VSENSE_INVALID_OPTIMIZATION_TYPE",
            "Unknown V-Sense optimization type.",
        )
    if data_mode not in {"live", "historical_only"}:
        raise VSenseContextError(
            "VSENSE_INVALID_OPTIMIZATION_TYPE",
            "V-Sense data_mode must be live or historical_only.",
        )

    history, dataset_meta = _load_or_use_history(history_df)
    if history.empty:
        raise VSenseContextError(
            "VSENSE_DATASET_NOT_AVAILABLE",
            "Canonical static ML dataset is not available.",
            status_code=503,
        )
    history = _normalize_history(history)
    latest_index = _latest_timestamp(history, now_utc)
    dataset_meta.update(_dataset_version_metadata(history, latest_index, now_utc))

    controls = _context_controls(
        optimization_type_id,
        history,
        latest_index,
        now_utc,
        live_values=live_values if data_mode == "live" else None,
        catalog=catalog_data,
    )
    inputs = _context_input_groups(
        optimization_type_id,
        history,
        latest_index,
        now_utc,
        catalog=catalog_data,
    )
    target = target_for_optimization(optimization_type_id)
    target_feature = str(target["feature_name"])
    baseline_value = _latest_numeric(history, target_feature, latest_index)
    warnings: list[str] = []
    if baseline_value is None:
        baseline_value = 0.0
        warnings.append(
            f"Target baseline for {target['id']} is unavailable; using neutral 0.0."
        )

    snapshot = {
        "context_id": context_id or f"ctx_{uuid4().hex}",
        "owner_user_id": owner_user_id,
        "created_at": _iso_z(now_utc),
        "expires_at": _iso_z(now_utc + timedelta(seconds=max(1, int(ttl_seconds)))),
        "as_of": _iso_z(latest_index or now_utc),
        "display_timezone": catalog_data["display_timezone"],
        "optimization_type_id": optimization_type_id,
        "catalog_version": CATALOG_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": dataset_meta,
        "models": model_readiness(),
        "control_profile": control_profile or {
            "profile_id": "plant-default",
            "version": 1,
            "parameters": [],
        },
        "controls": controls,
        "input_groups": inputs,
        "target": {
            "parameter_id": target["id"],
            "value": _json_number(baseline_value),
            "source": "historical",
            "source_timestamp": _iso_z(latest_index or now_utc),
            "quality": "good" if baseline_value is not None else "fallback",
        },
        "warnings": warnings,
    }
    return _json_safe(snapshot)


def _load_or_use_history(history_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if history_df is not None:
        return history_df.copy(), {
            "dataset_id": "static_ml_dataset",
            "version": "fixture-history",
        }

    path = _existing_static_dataset_path()
    if path is None:
        raise VSenseContextError(
            "VSENSE_DATASET_NOT_AVAILABLE",
            "Canonical static ML dataset is not available.",
            status_code=503,
        )
    try:
        if path.suffix.lower() in {".pkl", ".pickle"}:
            history = pd.read_pickle(path)
        else:
            history = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        raise VSenseContextError(
            "VSENSE_DATASET_NOT_AVAILABLE",
            "Canonical static ML dataset could not be read.",
            status_code=503,
        ) from exc
    return history, {
        "dataset_id": "static_ml_dataset",
        "version": _path_version(path),
    }


def _existing_static_dataset_path() -> Path | None:
    candidates: list[Path] = []
    try:
        candidates.append(get_static_dataset_path())
    except Exception:
        pass
    try:
        cfg = load_config("setting_ds_dv.yml")
        raw = Path(str(cfg.get("DATA") or ""))
        candidates.append(raw if raw.is_absolute() else get_repo_root() / raw)
    except Exception:
        pass
    try:
        cfg = load_config("setting_vsense.yml")
        raw = Path(str(cfg.get("DATA") or ""))
        candidates.append(raw if raw.is_absolute() else get_repo_root() / raw)
    except Exception:
        pass
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        for column in ("time", "date_time", "timestamp"):
            if column in out.columns:
                out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
                out = out.set_index(column)
                break
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize(timezone.utc)
        else:
            out.index = out.index.tz_convert(timezone.utc)
        out = out[~out.index.isna()]
    return out.sort_index()


def _latest_timestamp(history: pd.DataFrame, now_utc: datetime) -> datetime:
    if isinstance(history.index, pd.DatetimeIndex) and not history.empty:
        last = history.index[-1]
        return _aware_utc(last.to_pydatetime())
    return now_utc


def _dataset_version_metadata(
    history: pd.DataFrame,
    range_end: datetime,
    now_utc: datetime,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(str(tuple(str(c) for c in history.columns)).encode("utf-8"))
    digest.update(str(len(history)).encode("ascii"))
    if not history.empty:
        digest.update(str(history.index[-1]).encode("utf-8"))
    staleness = max(0, int((now_utc - range_end).total_seconds()))
    return {
        "dataset_id": "static_ml_dataset",
        "version": f"dataset-{digest.hexdigest()[:16]}",
        "range_end": _iso_z(range_end),
        "staleness_seconds": staleness,
    }


def _context_controls(
    optimization_type_id: str,
    history: pd.DataFrame,
    latest_index: datetime,
    now_utc: datetime,
    *,
    live_values: dict[str, dict[str, Any]] | None,
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    opt = optimization_by_id(catalog)[optimization_type_id]
    params = parameter_by_id(catalog)
    controls: list[dict[str, Any]] = []
    for parameter_id in opt["control_parameter_ids"]:
        definition = params[str(parameter_id)]
        feature_name = feature_for_parameter_id(str(parameter_id)) or str(parameter_id)
        live = (live_values or {}).get(feature_name) or (live_values or {}).get(str(parameter_id))
        value, source, source_ts, quality = _resolve_context_value(
            history,
            feature_name,
            latest_index,
            now_utc,
            live=live,
            default=(float(definition["approved_min"]) + float(definition["approved_max"])) / 2.0,
        )
        observed_min, observed_max = _observed_range(history, feature_name)
        controls.append(
            {
                "parameter_id": str(parameter_id),
                "value": _json_number(value),
                "source": source,
                "source_timestamp": _iso_z(source_ts),
                "freshness_seconds": max(0, int((now_utc - source_ts).total_seconds())),
                "quality": quality,
                "observed_min": _json_number(observed_min),
                "observed_max": _json_number(observed_max),
                "approved_min": definition.get("approved_min"),
                "approved_max": definition.get("approved_max"),
            }
        )
    return controls


def _context_input_groups(
    optimization_type_id: str,
    history: pd.DataFrame,
    latest_index: datetime,
    now_utc: datetime,
    *,
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    opt = optimization_by_id(catalog)[optimization_type_id]
    params = parameter_by_id(catalog)
    groups: list[dict[str, Any]] = []
    for group in opt["input_groups"]:
        values: list[dict[str, Any]] = []
        for parameter_id in group["parameter_ids"]:
            definition = params.get(str(parameter_id))
            if definition is None:
                continue
            feature_name = feature_for_parameter_id(str(parameter_id)) or str(parameter_id)
            value = _latest_numeric(history, feature_name, latest_index)
            source_ts = latest_index
            values.append(
                {
                    "parameter_id": str(parameter_id),
                    "value": _json_number(value),
                    "source": "historical" if value is not None else "unavailable",
                    "source_timestamp": _iso_z(source_ts),
                    "freshness_seconds": max(0, int((now_utc - source_ts).total_seconds())),
                    "quality": "good" if value is not None else "missing",
                }
            )
        groups.append({"id": group["id"], "label": group["label"], "values": values})
    return groups


def _resolve_context_value(
    history: pd.DataFrame,
    feature_name: str,
    latest_index: datetime,
    now_utc: datetime,
    *,
    live: dict[str, Any] | None,
    default: float,
) -> tuple[float, str, datetime, str]:
    if live:
        value = _finite_or_none(live.get("value"))
        if value is not None:
            ts = _aware_utc(live.get("timestamp") or now_utc)
            return value, "live", ts, str(live.get("quality") or "good")
    value = _latest_numeric(history, feature_name, latest_index)
    if value is not None:
        return value, "historical", latest_index, "historical_fallback"
    return default, "configured_default", latest_index, "fallback"


def _latest_numeric(
    history: pd.DataFrame,
    feature_name: str,
    latest_index: datetime,
) -> float | None:
    if feature_name not in history.columns:
        controls = control_parameter_by_feature()
        if feature_name in controls:
            return float(controls[feature_name]["default_value"])
        return None
    series = pd.to_numeric(history[feature_name], errors="coerce").dropna()
    if series.empty:
        return None
    if isinstance(series.index, pd.DatetimeIndex):
        bounded = series[series.index <= latest_index]
        if not bounded.empty:
            series = bounded
    return _finite_or_none(series.iloc[-1])


def _observed_range(history: pd.DataFrame, feature_name: str) -> tuple[float | None, float | None]:
    if feature_name not in history.columns:
        return None, None
    series = pd.to_numeric(history[feature_name], errors="coerce").dropna()
    if series.empty:
        return None, None
    return _finite_or_none(series.min()), _finite_or_none(series.max())


def _path_version(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8", errors="ignore"))
    try:
        stat = path.stat()
    except OSError:
        digest.update(b":missing")
    else:
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return f"dataset-{digest.hexdigest()[:16]}"


def _aware_utc(value: Any | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_number(value: Any) -> float | None:
    numeric = _finite_or_none(value)
    return float(numeric) if numeric is not None else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return _iso_z(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _iso_z(value: datetime) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")
