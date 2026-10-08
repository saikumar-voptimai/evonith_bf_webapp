"""Versioned charged-coke models: retrain in the background, accept, roll back.

    storage/bmo_charged_coke/models/
        <version>/        one candidate: model.npz, manifest, policy, report, holdout
        active.json       which version the app uses (absent: the bundled model)
        activations.jsonl every accept or roll-back: who, when, which version
        retrain.json      the running or last retrain: stage, message, version

A retrain only ever writes a candidate. Nothing changes in the app until an
operator reviews its report and accepts it; any earlier version can be made
active again the same way.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from utils.bmo.charged_coke.ledger import DEFAULT_STORAGE_DIR

_RETRAIN_LOCK = threading.Lock()


def models_dir(storage_dir: str | Path = DEFAULT_STORAGE_DIR) -> Path:
    return Path(storage_dir) / "models"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def active_bundle(storage_dir: str | Path, default_bundle: str | Path) -> Path:
    """The accepted model's directory, or the bundled model if none accepted."""

    pointer = models_dir(storage_dir) / "active.json"
    if pointer.is_file():
        version = json.loads(pointer.read_text(encoding="utf-8")).get("version")
        candidate = models_dir(storage_dir) / str(version)
        if version and (candidate / "model.npz").is_file():
            return candidate
    return Path(default_bundle)


def active_info(storage_dir: str | Path) -> dict[str, Any]:
    pointer = models_dir(storage_dir) / "active.json"
    return json.loads(pointer.read_text(encoding="utf-8")) if pointer.is_file() else {}


def versions(storage_dir: str | Path) -> list[dict[str, Any]]:
    """Every candidate on disk with its report, newest first."""

    out = []
    root = models_dir(storage_dir)
    if not root.is_dir():
        return out
    for folder in sorted((p for p in root.iterdir() if (p / "model.npz").is_file()), reverse=True):
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        out.append({"version": folder.name, "path": folder, "report": report})
    return out


def activate(storage_dir: str | Path, version: str, *, accepted_by: str, now: datetime, note: str = "") -> None:
    """Make ``version`` the active model and record who did it and when."""

    if not str(accepted_by).strip():
        raise ValueError("Accepting a model needs the name of whoever accepts it.")
    folder = models_dir(storage_dir) / version
    if not (folder / "model.npz").is_file():
        raise FileNotFoundError(f"No candidate {version}")
    previous = active_info(storage_dir).get("version")
    record = {"version": version, "accepted_by": accepted_by.strip(), "accepted_at": pd.Timestamp(now).isoformat(),
              "previous": previous, "note": note}
    _write_json(models_dir(storage_dir) / "active.json", record)
    with open(models_dir(storage_dir) / "activations.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def retrain_status(storage_dir: str | Path) -> dict[str, Any]:
    path = models_dir(storage_dir) / "retrain.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def start_retrain(
    *,
    storage_dir: str | Path,
    dataset_path: str | Path,
    base_bundle: str | Path,
    end: pd.Timestamp,
    requested_by: str,
    settings: Any = None,
) -> bool:
    """Start a background retrain to ``end``; False if one is already running."""

    status_path = models_dir(storage_dir) / "retrain.json"
    if not _RETRAIN_LOCK.acquire(blocking=False):
        return False

    def run() -> None:
        from utils.bmo.charged_coke.physics import load_physics_config
        from utils.bmo.charged_coke.policy import PolicyConfig
        from utils.bmo.charged_coke.training import TrainingSettings, train, write_bundle

        started = datetime.now()
        state = {"stage": "running", "message": "starting", "requested_by": requested_by,
                 "started_at": started.isoformat(timespec="seconds"), "data_end": str(end)}

        def say(text: str) -> None:
            state["message"] = text
            _write_json(status_path, state)

        try:
            say("reading the dataset")
            source = pd.read_csv(dataset_path, parse_dates=["time"]).set_index("time")
            result = train(source, load_physics_config(base_bundle), PolicyConfig.load(base_bundle), end=end,
                           settings=settings or TrainingSettings(), progress=say)
            write_bundle(result, models_dir(storage_dir) / result.version, base_bundle=base_bundle)
            state.update(stage="complete", version=result.version, message="ready for review",
                         finished_at=datetime.now().isoformat(timespec="seconds"))
        except Exception as exc:  # noqa: BLE001 - the operator sees the reason
            state.update(stage="failed", message=str(exc), trace=traceback.format_exc(limit=5),
                         finished_at=datetime.now().isoformat(timespec="seconds"))
        finally:
            _write_json(status_path, state)
            _RETRAIN_LOCK.release()

    _write_json(status_path, {"stage": "running", "message": "queued", "requested_by": requested_by,
                              "started_at": datetime.now().isoformat(timespec="seconds"), "data_end": str(end)})
    threading.Thread(target=run, name="charged-coke-retrain", daemon=True).start()
    return True


__all__ = ["activate", "active_bundle", "active_info", "models_dir", "retrain_status", "start_retrain", "versions"]
