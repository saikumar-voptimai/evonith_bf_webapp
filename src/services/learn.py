"""STRATA-Learn shadow run of the worker (``learn.shadow`` at 03:00, SV-1.7, #287).

``shadow(directory)`` runs ``utils.burden_engine.learn.run.shadow_run`` on the evidence store
``<directory>/evidence.csv`` (created from the shipped seed on the first run: 191 days to 7 Oct 2026) against the
deployed asset (STRATA 1.5) and the fact register, and writes the run's outputs next to it: ``strata_learn_state.json``,
``strata_learn_history.csv``, ``register_proposal.json`` and ``strata_learn_report.md`` (overwritten), plus
``runs/<date>.json`` with the day's summary (kept). Nothing in the asset, the register or the pages is changed.

The store receives no new days here: the evidence rows (q channels from STRATA-Flow runs of each day, the day's
calibrated omega against its 14-day anchor; GAS-20) are built offline, and ``append_csv`` adds them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from utils.burden_engine.facts import REGISTER_PATH
from utils.burden_engine.learn.evidence import EvidenceStore
from utils.burden_engine.learn.run import report_md, shadow_run
from utils.burden_engine.omega_model import MODEL_PATH

SRC = Path(__file__).resolve().parents[1]
STORE_DIR = SRC / "storage" / "burden_engine" / "learn"
IST = ZoneInfo("Asia/Kolkata")


def _plain(o: Any) -> Any:
    return o.item() if hasattr(o, "item") else str(o)


def shadow(
    directory: Path | str | None = None,
    *,
    now: datetime | None = None,
    append_csv: Path | str | None = None,
    spec: Mapping[str, Any] | None = None,
    register: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One shadow run on the store in ``directory`` (default ``STORE_DIR``); returns the summary that
    ``runs/<date>.json`` holds (IST date of ``now``)."""
    now = now or datetime.now(UTC)
    out = Path(directory) if directory is not None else STORE_DIR
    out.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(out / "evidence.csv", seed=True)
    created = not (out / "evidence.csv").exists()
    appended = store.append(pd.read_csv(append_csv, parse_dates=["date"])) if append_csv else 0
    if created or appended:
        store.save()
    if spec is None:
        spec = json.loads(Path(MODEL_PATH).read_text(encoding="utf-8"))
    if register is None:
        register = json.loads(Path(REGISTER_PATH).read_text(encoding="utf-8"))
    res = shadow_run(store, dict(spec), dict(register))
    res["history"].to_csv(out / "strata_learn_history.csv", index=False, float_format="%.6g")
    state = {k: v for k, v in res.items() if k != "history"}
    (out / "strata_learn_state.json").write_text(json.dumps(state, indent=1, default=_plain), encoding="utf-8")
    (out / "register_proposal.json").write_text(json.dumps(res["proposal"], indent=1, default=str), encoding="utf-8")
    (out / "strata_learn_report.md").write_text(report_md(res), encoding="utf-8")
    gate, dr, prop = res["gate"], res["drift"], res["proposal"]
    dates = pd.to_datetime(store.df["date"])
    summary = dict(
        at=now.isoformat(),
        evidence_days=int(len(store.df)),
        evidence_from=str(dates.min().date()) if len(dates) else None,
        evidence_to=str(dates.max().date()) if len(dates) else None,
        appended_days=int(appended),
        deployed=res["deployed"],
        gate=dict(status=gate.get("status"), promote=bool(gate.get("promote")), reason=gate.get("reason")),
        drift=dict(alarm=bool(dr.get("alarm")), basis=dr.get("basis"), first_alarm=dr.get("first_alarm_date")),
        proposal=dict(automatic=int(prop.get("n_auto", 0)), approval=int(prop.get("n_approval", 0))),
        skill={k: {m: round(float(v), 4) for m, v in d.items()} for k, d in res["skill"].items()},
    )
    runs = out / "runs"
    runs.mkdir(exist_ok=True)
    day = now.astimezone(IST).date().isoformat()
    (runs / f"{day}.json").write_text(json.dumps(summary, indent=1, default=_plain), encoding="utf-8")
    return summary


__all__ = ["STORE_DIR", "shadow"]
