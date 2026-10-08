"""Versioned Si models: background retrain, review, approval, rollback."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.bmo.si_forecast_context import LiveSiliconForecastSource
from utils.bmo.si_forecast.ledger import DEFAULT_STORAGE_DIR

_RETRAIN_LOCK = threading.Lock()


def _bundle_ready(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if int(manifest.get("format_version", 1)) == 1:
        return (path / "forest.npz").is_file()
    horizons = manifest.get("horizons") or []
    return bool(horizons) and all(
        (path / str(row.get("artifact", ""))).is_file() for row in horizons
    )


def models_dir(storage_dir: str | Path = DEFAULT_STORAGE_DIR) -> Path:
    return Path(storage_dir) / "models"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def active_info(storage_dir: str | Path) -> dict[str, Any]:
    path = models_dir(storage_dir) / "active.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def active_bundle(storage_dir: str | Path, default_bundle: str | Path) -> Path:
    default = Path(default_bundle)
    info = active_info(storage_dir)
    version = str(info.get("version") or "")
    if version == default.name:
        return default
    candidate = models_dir(storage_dir) / version
    if version and _bundle_ready(candidate):
        return candidate
    return default


def versions(
    storage_dir: str | Path, default_bundle: str | Path | None = None
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    root = models_dir(storage_dir)
    if root.is_dir():
        for folder in sorted(
            (path for path in root.iterdir() if _bundle_ready(path)),
            reverse=True,
        ):
            report_path = folder / "report.json"
            report = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.is_file()
                else {}
            )
            entries.append(
                {"version": folder.name, "path": folder, "report": report, "bundled": False}
            )
    if default_bundle is not None:
        default = Path(default_bundle)
        report_path = default / "report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else {}
        )
        entries.append(
            {"version": default.name, "path": default, "report": report, "bundled": True}
        )
    return entries


def activate(
    storage_dir: str | Path,
    version: str,
    *,
    default_bundle: str | Path,
    accepted_by: str,
    now: datetime,
    note: str = "",
) -> None:
    if not str(accepted_by).strip():
        raise ValueError("Accepting a silicon model requires the approver's name.")
    default = Path(default_bundle)
    folder = default if version == default.name else models_dir(storage_dir) / version
    if not _bundle_ready(folder):
        raise FileNotFoundError(f"No silicon model version {version}")
    previous = active_bundle(storage_dir, default).name
    record = {
        "version": version,
        "accepted_by": accepted_by.strip(),
        "accepted_at": pd.Timestamp(now).isoformat(),
        "previous": previous,
        "note": str(note),
    }
    _write_json(models_dir(storage_dir) / "active.json", record)
    log = models_dir(storage_dir) / "activations.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")


def retrain_status(storage_dir: str | Path) -> dict[str, Any]:
    path = models_dir(storage_dir) / "retrain.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def start_retrain(
    *,
    storage_dir: str | Path,
    base_bundle: str | Path,
    static_dataset_path: str | Path,
    data_end_origin: pd.Timestamp,
    requested_by: str,
    plant_timezone: str = "Asia/Kolkata",
    settings: Any = None,
) -> bool:
    """Create a candidate in a daemon thread; never activate it automatically."""

    status_path = models_dir(storage_dir) / "retrain.json"
    if not _RETRAIN_LOCK.acquire(blocking=False):
        return False

    def run() -> None:
        from utils.bmo.si_forecast.training import (
            TrainingSettings,
            train,
            write_bundle,
        )

        started = datetime.now()
        state: dict[str, Any] = {
            "stage": "running",
            "message": "starting",
            "requested_by": requested_by,
            "started_at": started.isoformat(timespec="seconds"),
            "data_end": str(data_end_origin),
        }

        def say(message: str) -> None:
            state["message"] = message
            _write_json(status_path, state)

        try:
            cfg = settings or TrainingSettings()
            say("fetching the 51 online channels, raw casts and charge events")
            source = LiveSiliconForecastSource(
                bundle_dir=base_bundle,
                static_dataset_path=static_dataset_path,
                plant_timezone=plant_timezone,
                lab_history_days=7,
            )
            feeds = source.fetch_training_sources(
                data_end_origin=data_end_origin,
                history_days=cfg.history_days + cfg.holdout_days,
            )
            result = train(
                feeds,
                base_bundle=base_bundle,
                data_end_origin=data_end_origin,
                settings=cfg,
                progress=say,
            )
            write_bundle(
                result,
                models_dir(storage_dir) / result.version,
                base_bundle=base_bundle,
            )
            state.update(
                stage="complete",
                version=result.version,
                message="ready for review",
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:  # noqa: BLE001 - expose retrain failure in UI
            state.update(
                stage="failed",
                message=str(exc),
                trace=traceback.format_exc(limit=8),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        finally:
            _write_json(status_path, state)
            _RETRAIN_LOCK.release()

    _write_json(
        status_path,
        {
            "stage": "running",
            "message": "queued",
            "requested_by": requested_by,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "data_end": str(data_end_origin),
        },
    )
    threading.Thread(target=run, name="bf2-si-retrain", daemon=True).start()
    return True


__all__ = [
    "activate",
    "active_bundle",
    "active_info",
    "models_dir",
    "retrain_status",
    "start_retrain",
    "versions",
]
