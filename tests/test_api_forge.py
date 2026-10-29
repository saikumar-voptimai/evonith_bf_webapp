"""FORGE routes (SV-1.11, #291): read the stored view, role user, RFC 7807 problems, units in the schemas.

The view is written by hand (40 days of shifts and the Rist frame of ``frontend/src/lib/charts/fixtures/
rist_shift.json``); the statistics, frames and calibration themselves are checked against page 10 in
``tests/test_services_forge.py``. Here the routes must return what ``services.forge`` computes from the stored file.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services import forge
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie

ROOT = Path(__file__).resolve().parents[1]
RIST_FIXTURE = ROOT / "frontend" / "src" / "lib" / "charts" / "fixtures" / "rist_shift.json"
START = date(2026, 9, 1)
DAYS = 40


def shift_rows() -> list[dict[str, Any]]:
    rows = []
    for i in range(DAYS * 3):
        day = START + timedelta(days=i // 3)
        letter = "ABC"[i % 3]
        burnt = 300 + 6 * math.sin(0.7 * i)
        rows.append(
            {
                "skey": f"{day}_{letter}",
                "sdate": day.isoformat(),
                "shift": letter,
                "t_mid": (pd.Timestamp(day) + pd.Timedelta(hours={"A": 10, "B": 18, "C": 26}[letter])).isoformat(),
                "stable": i % 7 != 0,
                "provisional": False,
                "K_energy": None if i < 30 else burnt - 1 + 4 * math.cos(1.3 * i),
                "K_burnt": burnt,
                "K_obal": None if i < 30 else burnt + 2,
                "K_charged": burnt + 4,
                "omega_fixed": None if i < 30 else 0.79,
                "Q_LZ_GJh_fixed": None if i < 30 else 45.0,
                "Q_UZ_GJh_fixed": None if i < 30 else 25.0,
                "r_H_fixed": None if i < 30 else 0.7,
                "omega_shift": 0.79 + 0.01 * math.sin(i),
                "Q_LZ_GJh_shift": 45 + math.cos(i),
                "Q_UZ_GJh_shift": 25 + math.sin(2 * i),
                "r_H_shift": 0.7 + 0.02 * math.cos(3 * i),
            }
        )
    return rows


def view() -> dict[str, Any]:
    rows = shift_rows()
    frame = {
        k: v for k, v in json.loads(RIST_FIXTURE.read_text(encoding="utf-8")).items() if k not in ("source", "expected")
    }
    frame["skey"] = rows[-2]["skey"]
    return {
        "schema": forge.SCHEMA,
        "model": forge.MODEL,
        "built_at": "2026-10-11T00:50:00+00:00",
        "job_id": "job-7",
        "settings": {
            "window_days": 14,
            "min_shifts": 10,
            "history_start": "2026-08-01",
            "predict_from": "2026-08-01",
            "stats_days": 30,
            "rist_days": 30,
        },  # fmt: skip
        "sources": {"hourly_end": "2026-10-11 05:00:00", "online_end": None, "hm_slag_end": None, "dpr_end": None},
        "provisional": [],
        "errors": 0,
        "shifts": rows,
        "rist_requested": 1,
        "rist": [frame],
    }


@pytest.fixture()
def store_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(forge, "STORE_DIR", tmp_path)
    return tmp_path


@pytest.fixture()
def client(store_dir):
    app = build()
    with TestClient(app, base_url="https://testserver") as c:
        c.headers["Cookie"] = session_cookie(app, "op")  # role user
        yield c


def test_without_a_stored_view_the_routes_answer_a_problem(client) -> None:
    r = client.get("/api/forge/prediction")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["title"] == "FORGE view not built" and "twin.forge" in r.json()["detail"]


def test_the_routes_need_a_session(store_dir) -> None:
    forge.save(view(), store_dir)
    with TestClient(build(), base_url="https://testserver") as anonymous:
        assert anonymous.get("/api/forge/prediction").status_code == 401
    assert set(PASSWORDS) >= {"op"}


def test_the_prediction_is_the_services_answer_for_the_default_period(client, store_dir) -> None:
    forge.save(view(), store_dir)
    r = client.get("/api/forge/prediction")
    assert r.status_code == 200, r.text
    body = r.json()
    ref = forge.prediction(forge.load(store_dir))
    assert body["start"] == str(START + timedelta(days=10)) and body["end"] == str(START + timedelta(days=DAYS - 1))
    assert body["start"] == str(ref["start"]) and body["stats"] == pytest.approx(ref["stats"])
    assert body["record"]["n"] == ref["record"]["n"] and len(body["shifts"]) == len(ref["shifts"]) > 0
    assert len(body["rolling"]) == len(ref["rolling"]) and body["rolling"][-1]["n"] == ref["stats"]["n"]
    first = body["shifts"][0]
    assert first["t_mid"].endswith("Z") and first["error"] == pytest.approx(first["k_forge"] - first["k_burnt"])
    assert body["view"]["built_at"].startswith("2026-10-11T00:50") and body["view"]["job_id"] == "job-7"
    assert body["view"]["sources"]["hourly_end"].startswith("2026-10-10T23:30")  # 05:00 IST
    unstable = client.get("/api/forge/prediction", params={"unstable": True}).json()
    assert unstable["stats"]["n"] > body["stats"]["n"] and unstable["with_unstable"]


def test_a_period_is_checked(client, store_dir) -> None:
    forge.save(view(), store_dir)
    r = client.get("/api/forge/prediction", params={"start": "2026-10-05", "end": "2026-10-01"})
    assert r.status_code == 422 and "after" in r.json()["detail"]
    r = client.get("/api/forge/prediction", params={"start": "2026-06-01"})  # to the last day: 132 days
    assert r.status_code == 422 and "92 days" in r.json()["detail"]
    r = client.get("/api/forge/prediction", params={"start": "2026-09-20", "end": "2026-09-24"})
    assert r.status_code == 200 and {s["day"] for s in r.json()["shifts"]} <= {f"2026-09-{d}" for d in range(20, 25)}


def test_the_rist_frames_are_the_stored_lines(client, store_dir) -> None:
    v = view()
    forge.save(v, store_dir)
    body = client.get("/api/forge/rist").json()
    assert body["requested"] == 1 and len(body["frames"]) == 1
    frame, stored = body["frames"][0], v["rist"][0]
    for key in ("skey", "t_r", "y_w", "actual", "model", "ideal"):
        assert frame[key] == stored[key]
    assert frame["p_actual"] == stored["p_actual"] and frame["p_model"] == stored["p_model"]
    assert frame["k_forge"] == pytest.approx(v["shifts"][-2]["K_energy"]) and frame["day"] == body["last_day"]
    assert client.get("/api/forge/rist", params={"end": "2026-09-05"}).json()["frames"] == []


def test_the_calibration_lists_the_shifts_of_the_window(client, store_dir) -> None:
    forge.save(view(), store_dir)
    body = client.get("/api/forge/calibration", params={"day": "2026-10-01"}).json()
    ref = forge.calibration(forge.load(store_dir), date(2026, 10, 1))
    assert body["window_start"] == "2026-09-17" and body["window_end"] == "2026-09-30" and body["complete"]
    assert body["n_shifts"] == ref["n_shifts"] == len(body["shifts"]) and body["omega_cal"] == pytest.approx(
        ref["omega_cal"]
    )
    default = client.get("/api/forge/calibration").json()
    assert default["day"] == str(START + timedelta(days=DAYS))


UNITS = ("kg/tHM", "%", "count", "dimensionless", "days", "GJ/h", "O/Fe", "O/C", "°C")


def _numeric(prop: dict[str, Any]) -> bool:
    types = {prop.get("type")} | {p.get("type") for p in prop.get("anyOf", [])}
    return bool(types & {"number", "integer"})


@pytest.mark.parametrize(
    "schema",
    ["ForgeViewInfo", "ForgeStats", "ForgeShift", "ForgeRollingPoint", "ForgePrediction", "RistLine", "RistIdeal",
     "ForgeRistFrame", "ForgeRist", "CalibrationShift", "ForgeCalibration"],
)  # fmt: skip
def test_every_numeric_field_states_its_unit(schema) -> None:
    spec = build().openapi()["components"]["schemas"][schema]
    missing = []
    for name, prop in spec.get("properties", {}).items():
        if not _numeric(prop):
            continue
        unit = (prop.get("description") or "").rsplit(",", 1)[-1].strip()
        if not (prop.get("description") and "," in prop["description"] and unit.split(" ")[0] in UNITS):
            missing.append(f"{name}: {prop.get('description')!r}")
    assert not missing, missing
