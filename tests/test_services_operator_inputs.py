"""Operator inputs in Neon (SV-3.1, #298): the document model keeps every value page 9 saves, the YAML file is read
(and written) without a store and seeds version 1, every past version is found by its date, stale edits and values the
optimisers cannot use are refused, an unmigrated store falls back to the YAML file."""

from __future__ import annotations

import shutil
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from data.bmo import ore_editor_preferences as prefs
from data.twin.engine import build_engine, session_factory
from data.twin.models import TwinBase
from data.twin.repositories import Actor, TwinStore, VersionConflict
from services import operator_inputs as svc

T0 = datetime(2026, 10, 20, 4, 30, tzinfo=UTC)  # Tue 20 Oct 2026 10:00 IST
DAY = timedelta(days=1)
SEEDED = T0 - 3 * DAY  # the YAML file's modification time in these tests
SV1 = Actor("sv1", "supervisor", "10.0.0.7")


def sqlite_store(path, *, migrated: bool = True) -> TwinStore:
    engine = build_engine(f"sqlite:///{path}")
    if migrated:
        TwinBase.metadata.create_all(engine)
    return TwinStore(session_factory(engine), engine)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "_modified", lambda _path: SEEDED)
    svc.invalidate()
    twin = sqlite_store(tmp_path / "twin.db")
    yield twin
    svc.invalidate()
    twin.dispose()


def price(current: svc.OperatorInputs, ore: str = "sinter_sp_02") -> float:
    return current.inputs["ore_editor"]["rows"][ore]["price_rs_per_mt"]


def set_price(value: float, ore: str = "sinter_sp_02"):
    def change(doc: dict) -> dict:
        doc.setdefault("ore_editor", {}).setdefault("rows", {}).setdefault(ore, {})["price_rs_per_mt"] = value
        return doc

    return change


# ------------------------------------------------------------------------------------------------- document model
def page_9_document() -> dict:
    """Every section page 9 saves, built by its own savers' builders from editors carrying every persisted column."""
    columns = ["ore_id", "selected", "price_rs_per_mt", "min_share_pct", "max_share_pct"]
    ore = pd.DataFrame([["sinter_sp_02", True, 8784, 58, 70], ["nmdc_rom", False, 10323.0, 0.0, 20.0]], columns=columns)
    flux = pd.DataFrame([{"flux_id": "quartz", **{c: 10.0 + i for i, c in enumerate(prefs.PERSISTED_FLUX_COLUMNS)}}])
    fuel = pd.DataFrame(
        [
            {
                "fuel_id": "coke",
                "enabled": True,
                **{c: 1.0 + i for i, c in enumerate(prefs.PERSISTED_FUEL_ASH_NUMERIC_COLUMNS)},
                **{"rate_basis": "Dry", "mn_basis": "MnO", "ti_basis": "TiO2"},
            }
        ]
    )
    dust = pd.DataFrame(
        [
            {
                "dust_id": "bf_gas_dust",
                "enabled": False,
                **{c: 2.0 + i for i, c in enumerate(prefs.PERSISTED_DUST_NUMERIC_COLUMNS)},
                "rate_basis": "kg_per_charge",
            }
        ]
    )
    doc: dict = {}
    doc = prefs.merge_ore_editor(doc, ore)
    doc = prefs.merge_flux(doc, flux)
    doc = prefs.merge_model_inputs(doc, {c: 1.5 + i for i, c in enumerate(prefs.PERSISTED_MODEL_INPUT_COLUMNS)})
    doc = prefs.merge_fuel_ash(doc, fuel)
    doc = prefs.merge_fuel_prices(doc, pd.DataFrame([{"fuel_id": "pci", "price_rs_per_mt": 21168.0}]))
    doc = prefs.merge_hm_chemistry(doc, {c: 0.5 + i for i, c in enumerate(prefs.PERSISTED_HM_CHEMISTRY_COLUMNS)})
    return prefs.merge_dust(doc, dust)


def test_the_document_model_keeps_every_value_page_9_saves() -> None:
    doc = page_9_document()
    assert set(doc) == set(svc.SECTIONS)
    assert svc.normalise(doc) == doc  # nothing dropped, nothing added
    assert (
        len(svc.normalise(doc)["fuel_ash_editor"]["rows"]["coke"]) == 1 + 17 + 3 + 1
    )  # enabled, numbers, bases, source


def test_the_yaml_document_equals_the_streamlit_reader() -> None:
    """Parity: the service reads the shipped YAML as page 9 did (``load_ore_editor_preferences``), and its normalised
    form (what version 1 stores) has the same values."""
    streamlit_path = prefs.load_ore_editor_preferences(svc.YAML_PATH)
    assert svc.read_yaml() == streamlit_path
    stored = svc.normalise(streamlit_path)
    flat_a, flat_b = pd.json_normalize(stored).iloc[0], pd.json_normalize(streamlit_path).iloc[0]
    assert set(flat_a.index) == set(flat_b.index)
    for key in flat_a.index:
        a, b = flat_a[key], flat_b[key]
        if isinstance(a, float):
            assert abs(a - b) <= 1e-6 * max(1.0, abs(b)), key
        else:
            assert a == b, key


def test_wrong_types_and_unknown_sections_are_refused() -> None:
    with pytest.raises(ValidationError):
        svc.normalise({"model_inputs": {"charge_mass_mt": "heavy"}})
    with pytest.raises(ValueError, match="unknown section"):
        svc.replace_sections({"prices": {}})
    assert svc.normalise({"model_inputs": {"charge_mass_mt": 26, "legacy_key": 1}}) == {
        "model_inputs": {"charge_mass_mt": 26.0}
    }  # unknown keys of a stored document are dropped, integers become floats


def test_put_sections_replace_remove_and_keep() -> None:
    doc = {"ore_editor": {"rows": {}}, "dust_editor": {"rows": {"d": {}}}, "model_inputs": {"charge_mass_mt": 26.0}}
    change = svc.replace_sections({"model_inputs": {"charge_mass_mt": 25.0}, "dust_editor": None})
    assert change(doc) == {"ore_editor": {"rows": {}}, "model_inputs": {"charge_mass_mt": 25.0}}


def test_values_the_optimisers_cannot_use_are_refused() -> None:
    svc.check(svc.read_yaml())  # the shipped inputs pass
    bad = {
        "ore_editor": {"rows": {"a": {"price_rs_per_mt": -1.0, "min_share_pct": 30.0, "max_share_pct": 20.0}}},
        "flux_editor": {"rows": {"q": {"max_qty_mt": 120.0, "min_qty_mt": 150.0}}},
        "model_inputs": {"charge_mass_mt": 0.0, "target_slag_basicity_min": 1.2, "target_slag_basicity_max": 1.0},
        "dust_editor": None,
    }
    with pytest.raises(svc.InvalidInputs) as refused:
        svc.check(bad)
    assert refused.value.problems == [
        "ore a.price_rs_per_mt: -1 is negative",
        "ore a share %: lower bound 30 above upper bound 20",
        "flux q t/d: lower bound 150 above upper bound 120",
        "slag CaO/SiO₂: lower bound 1.2 above upper bound 1",
        "model_inputs.charge_mass_mt: 0 must be positive",
    ]


# ---------------------------------------------------------------------------------------------------- as of a date
def test_a_date_means_the_end_of_that_ist_day() -> None:
    assert svc.as_of(date(2026, 10, 20)) == datetime(2026, 10, 20, 18, 29, 59, 999999, tzinfo=UTC)
    assert svc.as_of(datetime(2026, 10, 20, 10, 0)) == T0  # naive: IST
    assert svc.as_of(T0) == T0


# ------------------------------------------------------------------------------------------------------- no store
def test_without_a_store_the_yaml_file_is_read_and_written(tmp_path) -> None:
    path = tmp_path / "bmo_operator_inputs.yml"
    shutil.copyfile(svc.YAML_PATH, path)
    current = svc.load_current(None, yaml_path=path)
    assert (current.source, current.version, current.reason) == ("yaml", None, svc.NO_STORE)
    assert current.inputs == prefs.load_ore_editor_preferences(path)

    written = svc.write(set_price(9000.0), actor=SV1, store=None, yaml_path=path)
    assert written.changed and str(written) == str(path)
    assert price(svc.load_current(None, yaml_path=path)) == 9000.0
    assert prefs.load_ore_editor_preferences(path)["ore_editor"]["rows"]["sinter_sp_02"]["price_rs_per_mt"] == 9000.0
    assert svc.token(path, store=None).startswith("yaml:")
    with pytest.raises(svc.Unavailable, match="needs the twin store"):
        svc.load_at(date(2026, 10, 20), None)


# ------------------------------------------------------------------------------------------------------- versions
def test_every_past_version_is_found_by_its_date(store) -> None:
    yaml_inputs = svc.normalise(svc.read_yaml())
    empty = svc.load_current(store)
    assert (empty.source, empty.reason, empty.inputs) == ("yaml", svc.EMPTY, svc.read_yaml())
    assert svc.token(store=store).endswith(":empty")

    saves = [  # (price, save time): v2 on 20 Oct, v3 and v4 on 21 Oct, v5 on 24 Oct
        (9000.0, T0),
        (9100.0, T0 + DAY - timedelta(hours=2)),
        (9200.0, T0 + DAY + timedelta(hours=6)),
        (9300.0, T0 + 4 * DAY),
    ]
    for i, (value, when) in enumerate(saves):
        written = svc.write(set_price(value), actor=SV1, origin="api", note=f"save {i}", store=store, at=when,
                            base_version=0 if i == 0 else i + 1)  # fmt: skip
        assert written.changed and written.current.version == i + 2
        assert svc.token(store=store) == f"neon:{i + 2}"
    first = svc.load_at(SEEDED, store)
    assert (first.version, first.origin, first.author, first.inputs) == (1, "seed", "seed", yaml_inputs)

    by_date = {  # IST dates: the version in force at the end of the day
        date(2026, 10, 16): None,  # before the YAML's modification time
        date(2026, 10, 17): (1, 8784.0),
        date(2026, 10, 19): (1, 8784.0),
        date(2026, 10, 20): (2, 9000.0),
        date(2026, 10, 21): (4, 9200.0),  # two saves that day: the later one
        date(2026, 10, 23): (4, 9200.0),
        date(2026, 10, 24): (5, 9300.0),
        date(2026, 12, 31): (5, 9300.0),
    }
    for day, expected in by_date.items():
        found = svc.load_at(day, store)
        assert (None if found is None else (found.version, price(found))) == expected, day
    morning = svc.load_at(datetime(2026, 10, 21, 10, 0), store)  # naive: IST, between v3 (08:00) and v4 (16:00)
    assert (morning.version, price(morning), morning.note) == (3, 9100.0, "save 1")
    assert morning.valid_upto == T0 + DAY + timedelta(hours=6)

    current = svc.load_current(store)
    assert (current.source, current.version, current.author, current.valid_upto) == ("neon", 5, "sv1", None)
    assert current.provenance() == {
        "source": "neon",
        "version": 5,
        "valid_from": (T0 + 4 * DAY).isoformat(),
        "author": "sv1",
        "reason": None,
    }
    actions = [a.action for a in reversed(store.audit.query(object_type="operator_inputs"))]
    assert actions == ["operator_inputs.seed"] + ["operator_inputs.update"] * 4


def test_an_edit_from_a_stale_version_is_refused(store) -> None:
    svc.write(set_price(9000.0), actor=SV1, store=store, at=T0, base_version=0)
    with pytest.raises(VersionConflict):
        svc.write(set_price(9100.0), actor=SV1, store=store, at=T0 + DAY, base_version=0)
    with pytest.raises(VersionConflict, match="version 1, but version 2 is current"):
        svc.write(set_price(9100.0), actor=SV1, store=store, at=T0 + DAY, base_version=1)
    same = svc.write(set_price(9000.0), actor=SV1, store=store, at=T0 + DAY, base_version=2)
    assert not same.changed and same.current.version == 2 and "unchanged" in str(same)
    with pytest.raises(ValidationError):  # a wrong type is refused before anything is stored
        svc.write(lambda d: {**d, "model_inputs": {"charge_mass_mt": "x"}}, actor=SV1, store=store)
    assert svc.load_current(store).version == 2


def test_an_unmigrated_store_falls_back_to_the_yaml_file(tmp_path) -> None:
    svc.invalidate()
    twin = sqlite_store(tmp_path / "bare.db", migrated=False)
    try:
        current = svc.load_current(twin)
        assert current.source == "yaml" and "no operator_inputs table" in current.reason
        assert current.inputs == svc.read_yaml()
        assert svc.token(store=twin).endswith(":unavailable")
        with pytest.raises(svc.Unavailable, match="no operator_inputs table"):
            svc.write(set_price(9000.0), actor=SV1, store=twin)
        with pytest.raises(svc.Unavailable):
            svc.load_at(T0, twin)
    finally:
        svc.invalidate()
        twin.dispose()


# ------------------------------------------------------------------------------------------------- cache keys
def test_only_inputs_that_fit_their_cache_key_are_kept() -> None:
    v2 = {"source": "neon", "version": 2}
    yaml_down = {"source": "yaml", "reason": "The twin store is not reachable (OperationalError)."}
    assert svc.fits("neon:2", v2) and svc.fits("yaml:17:unavailable", yaml_down) and svc.fits("yaml:17", yaml_down)
    assert not svc.fits("neon:2", yaml_down)  # the store failed between the token and the read
    assert not svc.fits("neon:2", {"source": "neon", "version": 3})  # a version saved in between
    assert not svc.fits("yaml:17:empty", v2)  # the first version saved in between
    assert svc.fits("", yaml_down) and svc.fits("neon:2", None) and svc.fits("neon:2", {"source": "given"})
    assert svc.require_fit("neon:2", v2, "kept") == "kept"
    with pytest.raises(svc.TokenMismatch) as mismatch:
        svc.require_fit("neon:2", yaml_down, "used once")
    assert mismatch.value.value == "used once"


def test_the_yaml_file_standing_in_for_a_failed_store_is_announced() -> None:
    down = svc.fallback_warning({"source": "yaml", "reason": "The twin store is not reachable (OperationalError)."})
    assert down.startswith("The twin store is not reachable (OperationalError).") and svc.YAML_PATH.name in down
    assert "versions saved since are not in force" in down
    assert svc.fallback_warning({"source": "yaml", "reason": svc._why(Exception("no such table"))}) is not None
    for in_force in ({"source": "yaml", "reason": svc.NO_STORE}, {"source": "yaml", "reason": svc.EMPTY},
                     {"source": "neon", "version": 2}, {"source": "given"}, None):  # fmt: skip
        assert svc.fallback_warning(in_force) is None
