"""Live model view of the Blend & Fuel Optimiser (user decision 12 Nov 2026): the twin's hourly windows, the HMT
forecast with its live score against persistence, page 11's model check on the FORGE view, and the route."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import TwinStore
from services import forge, live_model
from services import store as store_service
from tests.test_api_auth import PASSWORDS, build
from tests.test_api_live import session_cookie

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)  # 14:30 IST


@pytest.fixture()
def twin(tmp_path):
    engine = build_engine(f"sqlite:///{tmp_path / 'twin.db'}")
    TwinBase.metadata.create_all(engine)
    store = TwinStore(session_factory(engine), engine)
    store_service.use(store)
    yield store
    store_service.reset()
    store.dispose()


def _windows(store: TwinStore, hours: int = 60) -> None:
    for h in range(hours, 0, -1):
        end = NOW - timedelta(hours=h)
        store.snapshots.upsert(
            {
                "window_start": end - timedelta(hours=8),
                "window_end": end,
                "evaluated_at": end,
                "status": "complete",
                "k_trace": 300.0 + h % 5,
                "k_forge": 302.0 + h % 3,
                "k_strata": 301.0,
                "k_charged": 320.0,
                "quantities": {
                    "hmt_c": 1480.0,
                    "hmt_now_c": 1478.0 + h % 4,
                    "hmt_forecast_c": 1482.0 + h % 3,
                    "hmt_forecast_horizon_h": 6.0,
                    "tau_raceway_h": 6.0,
                    "eta_co_pct": 46.0,
                    "eta_co_forge_pct": 46.5,
                    "top_gas_c": 150.0,
                    "top_gas_forge_c": 155.0,
                },
                "flags": {"hmt_forecast_gate": "pending"},
            }
        )


def _taps() -> pd.Series:
    idx = pd.date_range("2026-10-28 06:00", "2026-11-12 14:00", freq="2h")  # naive IST, as the status service
    return pd.Series([1480.0 + (i % 5) for i in range(len(idx))], index=idx)


def _shifts() -> pd.DataFrame:
    keys = [f"2026-11-1{d}_{s}" for d, s in zip("0001112", "ABCABCA", strict=True)]
    return pd.DataFrame(
        {
            "sdate": pd.to_datetime(["2026-11-10"] * 3 + ["2026-11-11"] * 3 + ["2026-11-12"]),
            "shift": list("ABCABCA"),
            "t_mid": pd.to_datetime(
                ["2026-11-10 10:00", "2026-11-10 18:00", "2026-11-11 02:00", "2026-11-11 10:00", "2026-11-11 18:00",
                 "2026-11-12 02:00", "2026-11-12 10:00"]
            ),  # fmt: skip
            "stable": [True, True, True, True, True, False, True],
            "K_burnt": [300.0, 305.0, 298.0, 301.0, 303.0, 299.0, 302.0],
            "K_energy": [302.0, 303.0, 300.0, 299.0, 305.0, 330.0, 300.0],
            "K_obal": [301.0, 304.0, 299.0, 300.0, 304.0, 298.0, 301.0],
            "K_charged": [320.0] * 7,
        },
        index=keys,
    )


def test_the_view_composes_windows_forecast_score_and_the_shift_check(twin) -> None:
    _windows(twin)
    v = live_model.LiveModelView.model_validate(
        live_model.view(
            twin, now=NOW, taps=lambda a, b: _taps(), shifts=_shifts(), built_at=NOW, gate="pending",
            cfg={"model_check": {}},
        )  # fmt: skip
    )
    assert v.hours == 48 and len(v.hourly) == 48 and v.latest is not None
    assert v.latest.end == NOW - timedelta(hours=1) and v.latest.k_burnt == 301.0 and v.latest.k_forge == 303.0
    assert v.latest.eta_co_pct == 46.0 and v.latest.top_gas_forge_c == 155.0
    # the taps shown are those of the hours; the forecast of the newest window lies in the future, unmeasured
    assert v.taps and all(t.at >= NOW - timedelta(hours=48) for t in v.taps)
    assert v.forecast.gate == "pending"
    assert v.forecast.latest is not None and v.forecast.latest.origin == v.latest.end
    assert v.forecast.latest.at == v.latest.end + timedelta(hours=6) and v.forecast.latest.measured is None
    # every stored forecast of the hours shown, the older ones scored against the tap at their horizon
    assert len(v.forecast.points) == 48 and v.forecast.points[0].measured is not None
    s = v.forecast.score
    assert s.days == 14 and s.n >= 40 and s.verdict == "better" and s.ratio is not None and s.ratio < 1
    assert s.rmse_model_c is not None and s.rmse_persistence_c is not None and s.rmse_model_c < s.rmse_persistence_c
    assert "better than persistence" in s.text
    # page 11's model check: stable shifts only (the unstable 330 is not scored), on track, its thresholds
    c = v.shifts
    assert c is not None and c.days == 14 and c.state == "ok" and c.energy.n == 6 and c.oxygen.n == 6
    assert c.energy.bias is not None and abs(c.energy.bias) < 8 and c.latest_shift == "2026-11-12_A"
    assert c.age_days == 0 and c.thresholds["bias_kgthm"] == 8.0 and len(c.rows) == 7
    assert [r.stable for r in c.rows].count(False) == 1 and c.built_at == NOW
    assert v.notes == []


def test_the_check_watches_a_biased_model_and_says_when_shifts_are_too_few() -> None:
    from utils.burden_engine.model_check import Thresholds

    shifts = _shifts()
    shifts["K_energy"] = shifts["K_burnt"] + 15.0
    today = pd.Timestamp("2026-11-12")
    c = live_model.shift_check(shifts, 14, thresholds=Thresholds(), today=today)
    assert c["state"] == "watch" and "bias +15.0" in c["reason"]
    few = live_model.shift_check(shifts.iloc[:3], 14, thresholds=Thresholds(), today=today)
    assert few["state"] == "insufficient" and few["energy"]["n"] == 3 and few["age_days"] == 2
    nothing = live_model.shift_check(None, 7, thresholds=Thresholds(), today=today)
    assert nothing["state"] == "insufficient" and nothing["rows"] == [] and nothing["latest_shift"] is None


def test_without_taps_or_forge_view_the_view_stands_with_notes(twin, monkeypatch) -> None:
    _windows(twin, hours=10)

    def no_taps(a, b):
        raise ConnectionError("Neon")

    def no_view():
        raise forge.ViewUnavailable("The FORGE view has not been built yet: the twin.forge job writes it hourly.")

    monkeypatch.setattr(forge, "load", no_view)
    v = live_model.LiveModelView.model_validate(
        live_model.view(twin, now=NOW, taps=no_taps, gate="pending", cfg={"model_check": {}})
    )
    assert len(v.hourly) == 10 and v.taps == [] and v.shifts is None
    assert v.forecast.score.verdict == "insufficient" and v.forecast.score.n == 0
    assert all(p.measured is None for p in v.forecast.points)
    assert any("tap temperatures" in n for n in v.notes) and any("FORGE view" in n for n in v.notes)
    empty = live_model.LiveModelView.model_validate(
        live_model.view(twin, now=NOW - timedelta(days=30), taps=lambda a, b: _taps(), shifts=None, gate="pending",
                        cfg={"model_check": {}})  # fmt: skip
    )
    assert empty.hourly == [] and empty.latest is None and empty.forecast.latest is None
    assert any("No twin window" in n for n in empty.notes)


def test_the_route_serves_the_view_to_users(twin, monkeypatch) -> None:
    _windows(twin, hours=20)
    monkeypatch.setattr(live_model, "_taps", lambda a, b: _taps())
    monkeypatch.setattr(forge, "load", lambda: forge.StoredView(meta={"built_at": NOW.isoformat()}, shifts=_shifts()))
    app = build()
    with TestClient(app, base_url="https://testserver") as client:
        cookies = {u: session_cookie(app, u) for u in PASSWORDS}
        assert client.get("/api/optimiser/live-model").status_code == 401
        r = client.get("/api/optimiser/live-model?hours=12&days=7", headers={"Cookie": cookies["op"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["hours"] == 12 and len(body["hourly"]) == 12 and body["shifts"]["days"] == 7
        assert body["forecast"]["gate"] in ("pending", "passed", "failed") and body["latest"]["k_burnt"] is not None
        assert client.get("/api/optimiser/live-model?hours=4", headers={"Cookie": cookies["op"]}).status_code == 422
        schema = client.get("/api/openapi.json").json()["components"]["schemas"]["LiveModelWindow"]
        assert all(schema["properties"][k].get("description") for k in ("k_burnt", "hmt_forecast_c", "top_gas_forge_c"))
