"""The day's calibration of the scheduled and API optimiser runs (SV-1.7, #287).

``calibration.nightly`` (02:30 IST) calibrates the two-zone balance on the stable shifts dated in the 14 days before
today (``data.burden_engine.calibration.live_calibration`` with ``as_of`` = today: the C shift that ends at 06:00 is
not complete at 02:30 and is not among them), saves it as ``src/storage/burden_engine/calibration/<date>.json`` and
builds the STRATA-Flow network Jacobian at it (``network_jacobian``: 17 network solves, written to the disk cache of
``utils.burden_engine.strata_network`` that every engine on the same calibration reads). The optimiser runs of the day
(``services.optimiser.default_case(calibration="day")``) use that calibration: the Step 2 and Step 1 plans of a day
compare on one calibration, and none of them spends the Jacobian's solves. Page 11 keeps its live calibration (the
shifts balanced so far), which ``calibration="live"`` reproduces.

The last ``KEEP`` days are kept. Pure file and model code: no Streamlit.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data.burden_engine.calibration import CalibrationInfo, live_calibration
from services import coke_energy as ce
from utils.coke_energy.pipeline import PipelineSettings

SRC = Path(__file__).resolve().parents[1]
STORE_DIR = SRC / "storage" / "burden_engine" / "calibration"
KEEP = 30
PLANT_TZ = "Asia/Kolkata"


class PlantDataUnavailable(RuntimeError):
    """The Coke Energy pipeline could not run (InfluxDB, Neon or the hourly dataset unavailable)."""


def today() -> date:
    return pd.Timestamp.now(tz=PLANT_TZ).date()


def folder(directory: Path | str | None = None) -> Path:
    """``directory``, else the store folder (``STORE_DIR``, read at the call: tests point it elsewhere)."""
    return Path(directory) if directory is not None else STORE_DIR


def path_for(day: date, directory: Path | str | None = None) -> Path:
    return folder(directory) / f"{day.isoformat()}.json"


def _jsonable(value: Any) -> Any:
    from services.optimiser import jsonable

    return jsonable(value)


def save(
    info: CalibrationInfo,
    day: date,
    *,
    made_at: datetime,
    job_id: str | None = None,
    network: Mapping[str, Any] | None = None,
    directory: Path | str | None = None,
) -> Path:
    """Write the day's calibration (atomic replace) and keep the last ``KEEP`` days."""
    path = path_for(day, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(
        day=day.isoformat(),
        made_at=made_at.isoformat(),
        job_id=job_id,
        info=dict(
            cal=info.cal,
            regime=info.regime,
            source=info.source,
            window=list(info.window),
            n_shifts=info.n_shifts,
            message=info.message,
            shifts=list(info.shifts),
        ),
        network=dict(network or {}),
    )
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_jsonable(doc), indent=1), encoding="utf-8")
    os.replace(tmp, path)
    prune(directory)
    return path


def read(day: date, directory: Path | str | None = None) -> dict[str, Any] | None:
    """The stored document of ``day`` (None when there is none or it does not parse)."""
    try:
        return json.loads(path_for(day, directory).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load(day: date, directory: Path | str | None = None) -> CalibrationInfo | None:
    """The day's calibration as page 11's ``CalibrationInfo`` (None when it was not made)."""
    doc = read(day, directory)
    if doc is None:
        return None
    i = doc["info"]
    return CalibrationInfo(
        cal=i["cal"],
        regime=i.get("regime"),
        source=i.get("source", "live"),
        window=tuple(i.get("window") or ("", "")),
        n_shifts=int(i.get("n_shifts", 0)),
        message=i.get("message", ""),
        shifts=list(i.get("shifts") or []),
    )


def prune(directory: Path | str | None = None, keep: int = KEEP) -> list[Path]:
    files = sorted(folder(directory).glob("????-??-??.json"))
    removed = files[:-keep] if keep > 0 else files
    for f in removed:
        f.unlink(missing_ok=True)
    return removed


def compute(
    day: date,
    cfg: Mapping[str, Any],
    *,
    nonce: int = 0,
    page_data_for: Callable[[ce.DataKeys, PipelineSettings], Any] | None = None,
    ce_config: Mapping[str, Any] | None = None,
) -> CalibrationInfo:
    """Calibration on the stable shifts dated in [day - 14 d, day) of the current pipeline result.

    Raises ``PlantDataUnavailable`` when the pipeline cannot run (the caller retries; page 11 would fall back to the
    shipped calibration)."""
    page_data_for = page_data_for or ce.page_data
    ce_cfg = ce.config() if ce_config is None else ce_config
    try:
        data = page_data_for(ce.data_keys(ce_cfg, nonce), PipelineSettings.from_config(ce_cfg))
    except Exception as exc:  # noqa: BLE001 - every source failure means the same: try again later
        raise PlantDataUnavailable(f"{type(exc).__name__}: {exc}") from exc
    return live_calibration(data.result, cfg, as_of=pd.Timestamp(day))


def network_jacobian(cal_json: str) -> dict[str, Any]:
    """Runs in a pool process: the STRATA-ω model of an engine on this calibration (exact solve at the calibration
    state and its 17-solve Jacobian, read from or written to the disk cache). Returns the network statistics."""
    from utils.burden_engine.engine import Engine

    t0 = time.perf_counter()
    eng = Engine(cal=json.loads(cal_json))
    model = eng.omega_model()
    stats = dict(model.network_stats())
    stats["seconds_total"] = round(time.perf_counter() - t0, 1)
    return stats


def summary(info: CalibrationInfo) -> dict[str, Any]:
    """The headline of a calibration: window, shifts, omega, losses, coke."""
    cal = info.cal or {}

    def num(key: str) -> float | None:
        try:
            return round(float(cal[key]), 4)
        except (KeyError, TypeError, ValueError):
            return None

    return dict(
        source=info.source,
        window=list(info.window),
        n_shifts=info.n_shifts,
        omega=num("omega"),
        q_lz_gj_h=num("Q_LZ_GJh"),
        q_uz_gj_h=num("Q_UZ_GJh"),
        r_h=num("r_H"),
        coke_kg_thm=num("K"),
        message=info.message,
    )


__all__ = [
    "KEEP",
    "STORE_DIR",
    "PlantDataUnavailable",
    "compute",
    "load",
    "network_jacobian",
    "path_for",
    "prune",
    "read",
    "save",
    "summary",
    "today",
]
