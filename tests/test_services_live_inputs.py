"""services.live_inputs (SV-7.3, #326): one upstream read per group and TTL for any number of readers, the as-of time
and stale flag per group, the snapshot id, the shared provider behind page 9's ore rows and page 11's yard, and the
snapshot a composed case carries.

Neon is a counting fake of the provider's stock and chemistry snapshots; InfluxDB the counting fake of
``tests/test_services_status.py``; the operator inputs the YAML file (no twin store). No plant source is reached.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

from data.furnace_status import repository as fs_repository
from services import live_inputs as li
from services import operator_inputs, status
from tests.test_services_status import NOW, CountingFetcher, plant_frames
from utils.bmo.types import OreChemistry

T_STOCK = "2026-10-27T03:00:00+00:00"  # 08:30 IST, the shift A entry
T_STOCK_OLDER = "2026-10-26T19:00:00+00:00"
T_SAMPLE = "2026-10-26T10:15:00+00:00"
T_SAMPLE_OLDER = "2026-10-25T06:00:00+00:00"
CASE_CAL = os.path.join(os.path.dirname(__file__), "fixtures", "cal_2z_jul_oct_2026_case_report_15g.json")


class FakeProvider:
    """Stand-in for ``EvonithBmoContextProvider``'s stock and chemistry snapshots (a new instance per read, as the
    service makes one); counts the reads on the class."""

    calls: dict[str, int] = {"stock": 0, "chemistry": 0}
    delay_s = 0.0
    fail_stock = False
    stock_time = T_STOCK

    def __init__(self) -> None:
        self._last_stock_diagnostics: dict[str, Any] = {}
        self._last_chemistry_diagnostics: dict[str, Any] = {}
        self.settings = {"data_sources": {"chemistry_time_range_days": 30}}

    def get_stock_snapshot(self) -> tuple[dict[str, float], list[str]]:
        type(self).calls["stock"] += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        warnings: list[str] = []
        if self.fail_stock:
            warnings.append("Stock query failed for offline_feed.raw_material_stock: ConnectionError: no neon")
            rows = [
                self._row("sinter_sp_02", "SINTER (SP-02)", "sinter_3", 3800.0, "sinter_fallback", ""),
                self._row("nmdc_donimalai", "NMDC DONIMALAI", "ore_1", 0.0, "offline_db_missing_as_zero", ""),
            ]
        else:
            rows = [
                self._row("sinter_sp_02", "SINTER (SP-02)", "sinter_3", 3800.0, "sinter_fallback", ""),
                self._row("nmdc_donimalai", "NMDC DONIMALAI", "ore_1", 1250.0, "offline_db", self.stock_time),
                self._row("lloyds_clo", "LLOYDS CLO", "ore_2", 640.0, "offline_db", T_STOCK_OLDER),
                self._row("geomin_clo", "GEOMIN CLO", "ore_3", 0.0, "offline_db_missing_as_zero", ""),
            ]
        self._last_stock_diagnostics = {
            "table": "offline_feed.raw_material_stock",
            "time_range": "last 1 week",
            "returned_rows": 0 if self.fail_stock else 2,
            "fallback_count": 1,
            "zero_count": 1,
            "material_rows": rows,
            "warnings": list(warnings),
        }
        return {r["ore_id"]: r["stock_mt"] for r in rows}, warnings

    @staticmethod
    def _row(ore_id: str, name: str, code: str, stock: float, source: str, at: str) -> dict[str, Any]:
        return {
            "ore_id": ore_id,
            "display_name": name,
            "material_code": code,
            "stock_mt": stock,
            "source": source,
            "timestamp": at,
        }

    def get_chemistry_snapshot(
        self, mode: str = "latest", window_days: int | None = None
    ) -> tuple[dict[str, OreChemistry], list[str]]:
        type(self).calls["chemistry"] += 1
        chem = {
            "sinter_sp_02": OreChemistry(fe_t_pct=54.8, moisture_pct=0.0, feo_pct=8.1, sio2_pct=5.6, al2o3_pct=2.9),
            "nmdc_donimalai": OreChemistry(fe_t_pct=63.4, moisture_pct=4.0, sio2_pct=3.4, al2o3_pct=1.9),
            "lloyds_clo": OreChemistry(fe_t_pct=64.1, moisture_pct=3.5, sio2_pct=2.8, al2o3_pct=1.6),
        }
        samples = {"sinter_sp_02": T_SAMPLE_OLDER, "nmdc_donimalai": T_SAMPLE, "lloyds_clo": T_SAMPLE_OLDER}
        rows = []
        for ore_id, c in chem.items():
            rows.append(
                {
                    "ore_id": ore_id,
                    "display_name": ore_id.upper(),
                    "material_code": {"sinter_sp_02": "sinter_3", "nmdc_donimalai": "ore_1", "lloyds_clo": "ore_2"}[
                        ore_id
                    ],
                    "source": "offline_db_latest_used" if mode == "latest" else "offline_db_avg_non_zero",
                    "table": f"offline_feed.{'sinter' if ore_id.startswith('sinter') else 'ore'}_chemistry",
                    "rows_used": 1 if mode == "latest" else 6,
                    "sample_timestamp": samples[ore_id],
                    "latest_used_time": T_STOCK,
                    **{f: getattr(c, f) for f in ("fe_t_pct", "moisture_pct", "feo_pct", "sio2_pct", "al2o3_pct")},
                }
            )
        rows.append(
            {
                "ore_id": "geomin_clo",
                "display_name": "GEOMIN CLO",
                "material_code": "ore_3",
                "source": "fallback",
                "table": "offline_feed.ore_chemistry",
                "rows_used": 0,
                "sample_timestamp": "",
                "latest_used_time": "",
                "fe_t_pct": 0.0,
            }
        )
        self._last_chemistry_diagnostics = {"mode": mode, "window_days": window_days, "material_rows": rows}
        return chem, []


def reset_caches() -> None:
    li.clear()
    status.clear_cache()
    operator_inputs.invalidate()


def install_fakes(monkeypatch) -> None:
    """Neon as ``FakeProvider`` (counters reset), both clocks at ``NOW``; shared with ``tests/test_api_inputs.py``."""
    reset_caches()
    FakeProvider.calls = {"stock": 0, "chemistry": 0}
    FakeProvider.delay_s = 0.0
    FakeProvider.fail_stock = False
    FakeProvider.stock_time = T_STOCK
    monkeypatch.setattr(li, "provider_factory", FakeProvider)
    monkeypatch.setattr(li, "clock", lambda: NOW)
    monkeypatch.setattr(status, "clock", lambda: NOW)


def counting_influx(monkeypatch) -> CountingFetcher:
    """InfluxDB as the counting fake of the status tests."""
    fake = CountingFetcher(plant_frames())
    monkeypatch.setattr(fs_repository, "TimeSeriesDataFetcher", fake)
    return fake


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    install_fakes(monkeypatch)
    yield
    reset_caches()


@pytest.fixture()
def fetcher(monkeypatch) -> CountingFetcher:
    return counting_influx(monkeypatch)


@pytest.fixture()
def inputs_reads(monkeypatch) -> list[str]:
    """Counts the operator-input reads (``services.operator_inputs.load_current``)."""
    calls: list[str] = []
    original = operator_inputs.load_current

    def counting(*args: Any, **kwargs: Any):
        calls.append(str(kwargs.get("yaml_path")))
        return original(*args, **kwargs)

    monkeypatch.setattr(operator_inputs, "load_current", counting)
    return calls


# ---------------------------------------------------------------------------------------------- one read per TTL
def test_two_readers_within_the_ttl_cause_one_upstream_read(fetcher, inputs_reads) -> None:
    a, b = li.stock(), li.stock()
    assert a == b and FakeProvider.calls["stock"] == 1
    c, d = li.rm_composition(), li.rm_composition()
    assert c == d and FakeProvider.calls["chemistry"] == 1
    e, f = li.operator_inputs_current(), li.operator_inputs_current()
    assert e == f and len(inputs_reads) == 1
    g, h = li.process_conditions(), li.process_conditions()
    assert g == h and fetcher.snapshot_queries == 1
    # A snapshot of the four groups right after costs no read.
    view = li.snapshot()
    assert FakeProvider.calls == {"stock": 1, "chemistry": 1} and len(inputs_reads) == 1
    assert fetcher.snapshot_queries == 1
    assert (view.stock, view.rm_composition, view.process_conditions, view.operator_inputs) == (a, c, g, e)


def test_concurrent_misses_wait_for_the_one_read() -> None:
    FakeProvider.delay_s = 0.15
    out: list[li.StockGroup] = []
    threads = [threading.Thread(target=lambda: out.append(li.stock())) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert FakeProvider.calls["stock"] == 1 and len(out) == 4 and all(g == out[0] for g in out)


def test_the_ttls_come_from_the_yaml_and_expire_a_read(monkeypatch) -> None:
    assert li._stock_cached.ttl_seconds == li.SPECS["stock"].ttl_s == 300.0
    assert li._chemistry_cached.ttl_seconds == li.SPECS["rm_composition"].ttl_s == 600.0
    assert li._operator_inputs_cached.ttl_seconds == li.SPECS["operator_inputs"].ttl_s == 10.0
    assert li.SPECS["process_conditions"].ttl_s == status.PERIOD_S == 10.0
    assert li.read_config() == li.SPECS
    monkeypatch.setattr(li._stock_cached, "ttl_seconds", 0.0)  # an expired entry is read again
    li.stock()
    li.stock()
    assert FakeProvider.calls["stock"] == 2


def test_a_chemistry_mode_or_window_of_its_own_is_one_more_read() -> None:
    li.rm_composition()
    li.rm_composition("latest", 30)  # page 9's defaults, spelled out
    assert FakeProvider.calls["chemistry"] == 1
    avg = li.rm_composition("avg", 14)
    assert FakeProvider.calls["chemistry"] == 2
    assert (avg.chemistry_mode, avg.window_days) == ("avg", 14)
    assert li.chemistry_key() == ("latest", 30) and li.chemistry_key("other", None) == ("latest", 30)


# ---------------------------------------------------------------------------------------- as-of time and staleness
def test_stock_is_as_of_the_newest_entry_and_stale_past_its_limit(monkeypatch) -> None:
    group = li.stock()
    assert group.as_of.isoformat() == T_STOCK and group.read_at == NOW
    assert group.age_min == pytest.approx((NOW - group.as_of).total_seconds() / 60) and not group.stale
    assert group.limits.ttl_s == 300.0 and group.limits.stale_after_min == 720.0
    assert group.total_t == 1250.0 + 640.0  # the materials with an entry
    by_id = {m.id: m for m in group.materials}
    assert by_id["nmdc_donimalai"].at.isoformat() == T_STOCK and by_id["sinter_sp_02"].at is None
    assert by_id["geomin_clo"].source == "offline_db_missing_as_zero" and group.source.endswith("raw_material_stock")

    li.clear()
    monkeypatch.setattr(li, "clock", lambda: NOW + timedelta(hours=13))
    late = li.stock()
    assert late.stale and "limit 12 h" in late.stale_reason
    assert late.age_min == pytest.approx(13 * 60 + (NOW - late.as_of).total_seconds() / 60)


def test_a_failed_stock_read_is_stale_with_its_error() -> None:
    FakeProvider.fail_stock = True
    group = li.stock()
    assert group.stale and group.error.startswith("Stock query failed") and group.stale_reason.startswith("The last")
    assert group.as_of is None and group.total_t == 0.0 and group.warnings == [group.error]


def test_composition_is_as_of_the_newest_sample_used() -> None:
    group = li.rm_composition()
    assert group.as_of.isoformat() == T_SAMPLE and not group.stale
    assert (group.chemistry_mode, group.window_days) == ("latest", 30)
    by_id = {m.id: m for m in group.materials}
    assert by_id["nmdc_donimalai"].fe_t_pct == 63.4 and by_id["nmdc_donimalai"].sample_at.isoformat() == T_SAMPLE
    assert by_id["sinter_sp_02"].feo_pct == 8.1 and by_id["sinter_sp_02"].table == "offline_feed.sinter_chemistry"
    assert by_id["geomin_clo"].source == "fallback" and by_id["geomin_clo"].sample_at is None
    assert by_id["nmdc_donimalai"].loi_pct == 0.0  # not in the lab row: the chemistry record's value
    assert group.source == "Neon offline_feed.ore_chemistry, offline_feed.sinter_chemistry"
    assert group.limits.stale_after_min == 7 * 24 * 60


def test_process_conditions_are_the_status_snapshots_readings(fetcher) -> None:
    group = li.process_conditions()
    view = status.current_snapshot_view()
    assert [p.key for p in group.parameters] == list(li.PROCESS_KEYS)
    page = {p.key: p for p in view.parameters}
    for p in group.parameters:
        assert (p.value, p.unit, p.at, p.state, p.setpoint) == (
            page[p.key].value,
            page[p.key].unit,
            page[p.key].at,
            page[p.key].state,
            page[p.key].setpoint,
        )
    assert group.as_of == view.freshness.data_at and group.read_at == view.freshness.computed_at == NOW
    assert group.telemetry == view.freshness.telemetry and not group.stale
    assert group.limits == li.GroupLimits(ttl_s=10.0, stale_after_min=15.0)
    assert group.warnings == ["Measurement not read: miscellaneous"]  # the fake's failing measurement
    assert group.blast_moisture is not None and group.blast_moisture.value_g_nm3 > 0  # the shipped Open-Meteo file
    assert fetcher.snapshot_queries == 1


def test_operator_inputs_are_the_version_in_force(inputs_reads) -> None:
    group = li.operator_inputs_current()
    assert group.source.startswith("YAML file") and group.version is None and not group.stale
    assert group.limits.stale_after_min is None and group.inputs.ore_editor is not None
    assert group.as_of is not None and group.reason == operator_inputs.NO_STORE
    assert inputs_reads == [str(operator_inputs.YAML_PATH)]


def test_the_yaml_standing_in_for_a_failed_store_is_stale(monkeypatch) -> None:
    failed = operator_inputs.OperatorInputs(
        inputs={}, source="yaml", reason="The twin store is not reachable (OperationalError).", yaml_path="x.yml"
    )
    monkeypatch.setattr(operator_inputs, "load_current", lambda **kw: failed)
    group = li.operator_inputs_current()
    assert group.stale and "versions saved since are not in force" in group.stale_reason
    assert group.warnings == [group.stale_reason] and group.error is None


# ------------------------------------------------------------------------------------------------------- snapshot
def test_the_snapshot_id_hashes_the_four_as_of_times(fetcher) -> None:
    view = li.snapshot()
    expected = li.snapshot_id([view.stock.as_of, view.rm_composition.as_of, view.process_conditions.as_of,
                               view.operator_inputs.as_of])  # fmt: skip
    assert view.snapshot_id == expected and len(view.snapshot_id) == 16 and view.read_at == NOW
    assert view.stale == [] and list(view.ref().as_of) == list(li.GROUP_KEYS)
    assert li.snapshot().snapshot_id == view.snapshot_id  # the same data, the same id
    assert view.yard_key == "|".join([NOW.isoformat()] * 3)

    li.clear()
    FakeProvider.stock_time = "2026-10-27T11:00:00+00:00"  # a newer stock entry: another id
    later = li.snapshot()
    assert later.snapshot_id != view.snapshot_id and later.ref().as_of["stock"].isoformat() == FakeProvider.stock_time
    assert li.snapshot_id([None, None, None, None]) == li.snapshot_id([None] * 4) != view.snapshot_id


def test_snapshot_ref_never_fails_a_run(monkeypatch) -> None:
    monkeypatch.setattr(li, "snapshot", lambda **kw: (_ for _ in ()).throw(RuntimeError("no provider")))
    assert li.snapshot_ref() == (None, "")


# -------------------------------------------------------------------------------------------- the shared provider
def test_the_shared_provider_composes_page_9s_ore_rows_and_page_11s_yard_from_the_cached_reads() -> None:
    """``provider().build_ore_inputs`` is the provider's own composition over the memoised snapshots; the panel's
    groups and ``load_yard`` use the same reads, so nothing is read twice."""
    p = li.provider()
    ores, diagnostics = p.build_ore_inputs(mode="latest", window_days=30)
    by_id = {o.ore_id: o for o in ores}
    assert by_id["nmdc_donimalai"].stock_mt == 1250.0 and by_id["nmdc_donimalai"].chemistry.fe_t_pct == 63.4
    assert by_id["nmdc_donimalai"].metadata["stock_timestamp"] == T_STOCK
    assert by_id["nmdc_donimalai"].metadata["chemistry_sample_timestamp"] == T_SAMPLE
    assert by_id["geomin_clo"].metadata["chemistry_source"] == "fallback" and by_id["geomin_clo"].stock_mt == 0.0
    assert p.get_data_diagnostics()["stock"]["table"] == "offline_feed.raw_material_stock"
    assert diagnostics["warnings"] == []
    assert FakeProvider.calls == {"stock": 1, "chemistry": 1}

    stock, composition = li.stock(), li.rm_composition()
    assert {m.id: m.stock_t for m in stock.materials}["nmdc_donimalai"] == by_id["nmdc_donimalai"].stock_mt
    assert {m.id: m.fe_t_pct for m in composition.materials}["lloyds_clo"] == by_id["lloyds_clo"].chemistry.fe_t_pct

    yard = li.load_yard({"yard": {"chemistry_mode": "latest", "chemistry_window_days": 30}, "prices": {}})
    rows = {m["id"]: m for m in yard["materials"]}
    assert rows["nmdc_donimalai"]["stock_mt"] == 1250.0 and rows["nmdc_donimalai"]["chem"]["Fe"] == 63.4
    assert rows["nmdc_donimalai"]["carried"] and not rows["geomin_clo"]["carried"]
    assert yard["operator_inputs"]["source"] == "yaml"
    assert FakeProvider.calls == {"stock": 1, "chemistry": 1}  # the yard read nothing more

    p.get_chemistry_snapshot(mode="avg")  # the provider's own default window
    assert FakeProvider.calls["chemistry"] == 2 and li.rm_composition("avg", 30) is not None
    assert FakeProvider.calls["chemistry"] == 2


# --------------------------------------------------------------------------------------- a run carries the snapshot
def _ref(snapshot_id: str) -> li.LiveInputsRef:
    return li.LiveInputsRef(
        snapshot_id=snapshot_id,
        read_at=NOW,
        as_of={k: NOW for k in li.GROUP_KEYS},
        stale=[],
        chemistry_mode="avg",
        chemistry_window_days=30,
    )


def test_the_optimiser_case_carries_the_snapshot_and_recomposes_the_yard_when_a_read_refreshed(monkeypatch) -> None:
    from services import optimiser as svc
    from services.optimiser_results import case_view

    svc.clear()
    cfg = dict(svc.read_page_config(0))
    cfg.pop("prices", None)
    cfg["calibration"] = dict(cfg.get("calibration") or {}, fallback_path=CASE_CAL)
    live_yard = svc.reference_yard(cfg)  # a yard that passes for live: every material carried
    loads: list[int] = []
    monkeypatch.setattr(svc, "load_yard", lambda c: loads.append(1) or live_yard)
    monkeypatch.setattr(svc, "read_charged_blend", lambda materials: {})
    monkeypatch.setattr(li, "snapshot_ref", lambda **kw: (_ref("a1b2c3d4e5f60718"), "reads-1"))
    kw = dict(omega_mode="fixed", calibration_basis="reference", yard_basis="live", cfg=cfg, production_t_d=2270.0)

    case = svc.default_case("step2", **kw)
    assert case.yard_source == "live" and case.live_inputs["snapshot_id"] == "a1b2c3d4e5f60718"
    assert case.summary()["live_inputs"]["chemistry_mode"] == "avg"
    assert case_view(svc.jsonable(case.summary())).live_inputs.snapshot_id == "a1b2c3d4e5f60718"
    svc.default_case("step2", **kw)
    assert loads == [1]  # the same reads: the yard is reused

    monkeypatch.setattr(li, "snapshot_ref", lambda **kw: (_ref("ffffffffffffffff"), "reads-2"))
    again = svc.default_case("step2", **kw)
    assert loads == [1, 1] and again.live_inputs["snapshot_id"] == "ffffffffffffffff"

    reference = svc.default_case("step2", **{**kw, "yard_basis": "reference"})
    assert reference.live_inputs is None and case_view(svc.jsonable(reference.summary())).live_inputs is None
    svc.clear()


def test_the_blend_mix_case_carries_the_snapshot_of_its_sources() -> None:
    """``CaseSources.live_inputs`` reaches the composed case and the run result (``bmo.run``'s JSON)."""
    import json

    from services import bmo_views
    from tests.test_services_bmo_case import golden_day

    ref = _ref("0123456789abcdef").model_dump(mode="json")
    with golden_day("2026-10-01") as (sources, inputs, _golden):
        out = bmo_views.run_case(
            {"pellet_inputs_confirmed": True, "coke_model": "data_driven"}, sources=replace(sources, live_inputs=ref)
        )
    result = bmo_views.BmoRunResult.model_validate(json.loads(json.dumps(out, allow_nan=False)))
    assert result.live_inputs is not None and result.live_inputs.snapshot_id == "0123456789abcdef"
    assert result.live_inputs.as_of["stock"] == NOW and result.status in ("ok", "infeasible")
