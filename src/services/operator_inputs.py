"""Operator inputs of the Blend Mix Optimiser and the Burden & Fuel Optimiser, versioned in Neon (SV-3.1, #298).

What the operators set is one document of the shape of ``src/config/bmo_operator_inputs.yml``: ferrous availability,
prices and share bounds (``ore_editor``), flux prices, stock and quantity bounds (``flux_editor``), slag targets and
charging limits (``model_inputs``), fuel prices and ash chemistry (``fuel_ash_editor``), the held hot-metal chemistry
(``hot_metal_chemistry``) and dust inputs (``dust_editor``). Every save is a new version in ``twin.operator_inputs``
(SCD-2: valid in [valid_from, valid_upto), author, origin, one ``twin.audit_log`` row in the same transaction).

* ``load_current``: the current version. The YAML file when the twin store is not configured, not migrated, not
  reachable or still empty; ``source`` says which and ``reason`` why.
* ``load_at``: the version valid at a time or a date (the end of that IST day); the history needs the store.
* ``write``: the next version from a change of the current document; the first save records the YAML as version 1
  (origin ``seed``, valid from the file's modification time). Without a configured store the YAML file is written
  (development, the test suite). The API never writes the file.
* ``token``: a cheap key of the current version for the readers' caches (re-read after 10 s, 60 s after a failure).

Readers: page 9 (``ui.bmo.operator_inputs``), page 11, the API and the worker through
``data.burden_engine.inputs.load_yard`` (``EvonithBmoContextProvider.get_operator_inputs``), ``GET`` and ``PUT
/api/operator-inputs``. The document model below is the API schema (units in the descriptions) and the validation of
every write; it covers every field page 9 persists (``data.bmo.ore_editor_preferences``, tested). ``check`` refuses
what the optimisers cannot use (``problems``) in the sections an API write sets.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import types
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError

from data.twin.models import OperatorInputs as OperatorInputsRow
from data.twin.repositories import Actor, Seed, TwinStore, VersionConflict
from services import store as store_service

log = logging.getLogger(__name__)

YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "bmo_operator_inputs.yml"
IST = ZoneInfo("Asia/Kolkata")
SECTIONS = ("ore_editor", "flux_editor", "model_inputs", "fuel_ash_editor", "hot_metal_chemistry", "dust_editor")
TOKEN_TTL_S = 10.0  # how long a reader trusts the current version number
TOKEN_RETRY_S = 60.0  # after a failure: no new connection attempt before this

NO_STORE = "No twin store is configured (BF_TWIN_DB_URL / DATABASE_URL): the inputs come from the YAML file."
EMPTY = "The twin store holds no version yet: the YAML file is in force and becomes version 1 at the first save."


class Unavailable(RuntimeError):
    """The twin store is configured but cannot serve the request (not migrated, not reachable), or is not configured
    for a request that needs it (the history)."""


# ------------------------------------------------------------------------------------------------- document model
class _Section(BaseModel):
    # One JSON schema for requests and responses (no -Input / -Output pair in the TypeScript client).
    model_config = ConfigDict(extra="forbid", json_schema_mode_override="validation")


class OreInputRow(_Section):
    price_rs_per_mt: float | None = Field(None, description="Price per wet tonne, Rs/t")
    min_share_pct: float | None = Field(None, description="Lower bound of the wet share of the ferrous charge, %")
    max_share_pct: float | None = Field(None, description="Upper bound of the wet share of the ferrous charge, %")


class OreEditorInputs(_Section):
    selected_ore_ids: list[str] = Field(
        default_factory=list,
        description="Ferrous materials offered to the optimisers (availability), ids of bmo_ore_mapping.yml; "
        "empty: every material",
    )
    rows: dict[str, OreInputRow] = Field(default_factory=dict, description="Price and share bounds by ore id")


class FluxInputRow(_Section):
    price_rs_per_mt: float | None = Field(None, description="Price per wet tonne, Rs/t")
    stock_mt: float | None = Field(None, description="Stock in the yard, t")
    min_qty_mt: float | None = Field(None, description="Lower bound of the flux charged when it is optimised, t/d")
    max_qty_mt: float | None = Field(
        None, description="Upper bound of the flux charged when it is optimised (none: the stock), t/d"
    )


class FluxEditorInputs(_Section):
    rows: dict[str, FluxInputRow] = Field(default_factory=dict, description="Price, stock and bounds by flux id")


class SlagChargingInputs(_Section):
    target_slag_basicity_min: float | None = Field(None, description="Lower limit of the slag CaO/SiO₂, dimensionless")
    target_slag_basicity_max: float | None = Field(None, description="Upper limit of the slag CaO/SiO₂, dimensionless")
    target_slag_t_basicity_min: float | None = Field(
        None, description="Lower limit of the slag (CaO + MgO)/SiO₂, dimensionless"
    )
    target_slag_t_basicity_max: float | None = Field(
        None, description="Upper limit of the slag (CaO + MgO)/SiO₂, dimensionless"
    )
    target_slag_rate_kg_per_thm: float | None = Field(None, description="Upper limit of the slag rate, kg/tHM")
    target_slag_al2o3_max_pct: float | None = Field(None, description="Upper limit of Al₂O₃ in the slag, %")
    target_slag_mgo_min_pct: float | None = Field(None, description="Lower limit of MgO in the slag, %")
    target_slag_mgo_al2o3_ratio_min: float | None = Field(
        None, description="Lower limit of MgO per Al₂O₃ charged, dimensionless"
    )
    max_charges_per_hour: float | None = Field(None, description="Charging capacity of the skips, charges/h")
    charge_mass_mt: float | None = Field(None, description="Mass of one charge, t")


class FuelInputRow(_Section):
    enabled: bool | None = Field(None, description="The fuel's ash enters the slag balance")
    rate_kg_per_thm: float | None = Field(None, description="Fuel rate, kg/tHM")
    price_rs_per_mt: float | None = Field(None, description="Price per wet tonne, Rs/t")
    moisture_pct: float | None = Field(None, description="Moisture (TM for coke and nut coke, IM for PCI), %")
    vm_pct: float | None = Field(None, description="Volatile matter of the ash analysis, %")
    ash_pct: float | None = Field(None, description="Ash of the fuel, %")
    sio2_pct: float | None = Field(None, description="SiO₂ in the ash, %")
    al2o3_pct: float | None = Field(None, description="Al₂O₃ in the ash, %")
    cao_pct: float | None = Field(None, description="CaO in the ash, %")
    mgo_pct: float | None = Field(None, description="MgO in the ash, %")
    fe2o3_pct: float | None = Field(None, description="Fe₂O₃ in the ash, %")
    mno_pct: float | None = Field(None, description="Mn or MnO in the ash (mn_basis), %")
    tio2_pct: float | None = Field(None, description="Ti or TiO₂ in the ash (ti_basis), %")
    alkali_pct: float | None = Field(None, description="Alkalis in the ash, %")
    na2o_pct: float | None = Field(None, description="Na₂O in the ash, %")
    k2o_pct: float | None = Field(None, description="K₂O in the ash, %")
    s_pct: float | None = Field(None, description="Sulphur in the fuel, %")
    p_pct: float | None = Field(None, description="Phosphorus in the fuel, %")
    rate_basis: str | None = Field(None, description="Basis of the fuel rate")
    mn_basis: str | None = Field(None, description="mn or mno: how mno_pct is given")
    ti_basis: str | None = Field(None, description="ti or tio2: how tio2_pct is given")
    chemistry_source: str | None = Field(None, description="manual: saved from the page")


class FuelEditorInputs(_Section):
    rows: dict[str, FuelInputRow] = Field(
        default_factory=dict, description="Price, rate and ash chemistry by fuel id (coke, nut_coke, pci)"
    )


class HotMetalChemistryInputs(_Section):
    carbon_pct: float | None = Field(None, description="Carbon in the hot metal held for the slag balance, %")
    silicon_pct: float | None = Field(None, description="Silicon in the hot metal held for the slag balance, %")
    sulphur_pct: float | None = Field(None, description="Sulphur in the hot metal held for the slag balance, %")
    other_pct: float | None = Field(None, description="Other elements in the hot metal held for the slag balance, %")


class DustInputRow(_Section):
    enabled: bool | None = Field(None, description="The dust is deducted from the burden balance")
    wet_qty_mt: float | None = Field(None, description="Wet dust when rate_basis is mt_per_day, t/d")
    quantity_kg_per_charge: float | None = Field(
        None, description="Wet dust when rate_basis is kg_per_charge, kg/charge"
    )
    moisture_pct: float | None = Field(None, description="Moisture of the dust, %")
    sio2_pct: float | None = Field(None, description="SiO₂ in the dry dust, %")
    al2o3_pct: float | None = Field(None, description="Al₂O₃ in the dry dust, %")
    cao_pct: float | None = Field(None, description="CaO in the dry dust, %")
    mgo_pct: float | None = Field(None, description="MgO in the dry dust, %")
    fe_pct: float | None = Field(None, description="Fe in the dry dust, %")
    mn_pct: float | None = Field(None, description="Mn in the dry dust, %")
    p_pct: float | None = Field(None, description="Phosphorus in the dry dust, %")
    s_pct: float | None = Field(None, description="Sulphur in the dry dust, %")
    ti_pct: float | None = Field(None, description="Ti in the dry dust, %")
    zn_pct: float | None = Field(None, description="Zn in the dry dust, %")
    na2o_pct: float | None = Field(None, description="Na₂O in the dry dust, %")
    k2o_pct: float | None = Field(None, description="K₂O in the dry dust, %")
    caf2_pct: float | None = Field(None, description="CaF₂ in the dry dust, %")
    rate_basis: str | None = Field(None, description="mt_per_day or kg_per_charge")
    source: str | None = Field(None, description="manual: saved from the page")


class DustEditorInputs(_Section):
    rows: dict[str, DustInputRow] = Field(default_factory=dict, description="Quantity and chemistry by dust id")


class OperatorInputsDocument(_Section):
    """The operator inputs; a section left out is not set (the configured defaults apply)."""

    ore_editor: OreEditorInputs | None = Field(None, description="Ferrous materials: availability, prices, bounds")
    flux_editor: FluxEditorInputs | None = Field(None, description="Fluxes: prices, stock and quantity bounds")
    model_inputs: SlagChargingInputs | None = Field(None, description="Slag targets and charging limits")
    fuel_ash_editor: FuelEditorInputs | None = Field(None, description="Fuel prices, rates and ash chemistry")
    hot_metal_chemistry: HotMetalChemistryInputs | None = Field(
        None, description="Hot-metal chemistry held for the slag balance"
    )
    dust_editor: DustEditorInputs | None = Field(None, description="BF gas dust deducted from the burden balance")


def _model_of(annotation: Any) -> Any:
    """``X`` of ``X | None``."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        options = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(options) == 1:
            return options[0]
    return annotation


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def prune(doc: Any, model: type[BaseModel] = OperatorInputsDocument) -> dict[str, Any]:
    """``doc`` with only the keys of the document model (unknown keys and null values dropped, null rows empty)."""
    if not isinstance(doc, Mapping):
        return {}
    out: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        value = doc.get(name)
        if value is None:
            continue
        annotation = _model_of(info.annotation)
        if _is_model(annotation):
            value = prune(value, annotation)
        elif typing.get_origin(annotation) is dict and isinstance(value, Mapping):
            item = typing.get_args(annotation)[1]
            if _is_model(item):
                value = {str(k): prune(v or {}, item) for k, v in value.items()}
        out[name] = value
    return out


def normalise(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The document as stored: known keys only, numbers as floats, nothing added (``ValueError`` on a wrong type)."""
    model = OperatorInputsDocument.model_validate(prune(doc))
    return model.model_dump(exclude_unset=True, exclude_none=True)


def replace_sections(sections: Mapping[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A change that replaces the given sections, removes those given as None and keeps the others
    (``PUT /api/operator-inputs``)."""
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown section(s): {', '.join(sorted(unknown))}")

    def change(doc: dict[str, Any]) -> dict[str, Any]:
        out = {**doc, **{k: v for k, v in sections.items() if v is not None}}
        return {k: v for k, v in out.items() if not (k in sections and sections[k] is None)}

    return change


class InvalidInputs(ValueError):
    """Operator inputs the optimisers cannot use (``problems``)."""

    def __init__(self, found: list[str]) -> None:
        super().__init__("; ".join(found))
        self.problems = found


def check(sections: Mapping[str, Any]) -> None:
    """Raise ``InvalidInputs`` when the given sections (None values ignored) have ``problems``."""
    found = problems(normalise({k: v for k, v in sections.items() if v is not None}))
    if found:
        raise InvalidInputs(found)


def problems(doc: Mapping[str, Any]) -> list[str]:
    """What makes the inputs unusable for the optimisers: negative prices, stock or quantities, shares outside
    0-100 %, a lower bound above its upper bound, no positive charging capacity."""
    out: list[str] = []

    def bounds(where: str, lo: Any, hi: Any, top: float | None = None) -> None:
        for name, value in (("lower", lo), ("upper", hi)):
            if value is not None and (value < 0 or (top is not None and value > top)):
                limit = f"0-{top:g}" if top is not None else ">= 0"
                out.append(f"{where}: {name} bound {value:g} outside {limit}")
        if lo is not None and hi is not None and lo > hi:
            out.append(f"{where}: lower bound {lo:g} above upper bound {hi:g}")

    def non_negative(where: str, row: Mapping[str, Any], *keys: str) -> None:
        for key in keys:
            value = row.get(key)
            if value is not None and value < 0:
                out.append(f"{where}.{key}: {value:g} is negative")

    for ore_id, row in ((doc.get("ore_editor") or {}).get("rows") or {}).items():
        non_negative(f"ore {ore_id}", row, "price_rs_per_mt")
        bounds(f"ore {ore_id} share %", row.get("min_share_pct"), row.get("max_share_pct"), 100.0)
    for flux_id, row in ((doc.get("flux_editor") or {}).get("rows") or {}).items():
        non_negative(f"flux {flux_id}", row, "price_rs_per_mt", "stock_mt")
        bounds(f"flux {flux_id} t/d", row.get("min_qty_mt"), row.get("max_qty_mt"))
    for fuel_id, row in ((doc.get("fuel_ash_editor") or {}).get("rows") or {}).items():
        non_negative(f"fuel {fuel_id}", row, "price_rs_per_mt", "rate_kg_per_thm")
    mi = doc.get("model_inputs") or {}
    bounds("slag CaO/SiO₂", mi.get("target_slag_basicity_min"), mi.get("target_slag_basicity_max"))
    for key in ("max_charges_per_hour", "charge_mass_mt"):
        if mi.get(key) is not None and mi[key] <= 0:
            out.append(f"model_inputs.{key}: {mi[key]:g} must be positive")
    return out


# ----------------------------------------------------------------------------------------------------- the view
@dataclass(frozen=True)
class OperatorInputs:
    """The operator inputs as a reader gets them, with where they came from."""

    inputs: dict[str, Any]
    #: ``neon`` (a version of ``twin.operator_inputs``) or ``yaml`` (the file).
    source: str
    version: int | None = None
    valid_from: datetime | None = None
    valid_upto: datetime | None = None
    author: str | None = None
    origin: str | None = None
    note: str | None = None
    #: Why the YAML file was used.
    reason: str | None = None
    yaml_path: str | None = None

    def provenance(self) -> dict[str, Any]:
        """Where the inputs came from, as plain values (stored with optimiser runs, shown by the pages)."""
        return {
            "source": self.source,
            "version": self.version,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "author": self.author,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Written:
    """The result of a save: the inputs in force afterwards and whether a version (or the file) was written."""

    current: OperatorInputs
    changed: bool
    location: str = field(default="")

    def __str__(self) -> str:  # "saved to ..." in the page's messages
        return self.location


def yaml_file(path: str | Path | None = None) -> Path:
    if path is None:
        return YAML_PATH
    p = Path(path)
    return p if p.is_absolute() else YAML_PATH.parents[2] / p


def read_yaml(path: str | Path | None = None) -> dict[str, Any]:
    """The YAML document (empty when the file is missing); what ``load_ore_editor_preferences`` reads."""
    p = yaml_file(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def write_yaml(path: str | Path | None, doc: Mapping[str, Any]) -> Path:
    """Write the document to the YAML file (``write_preferences_yaml`` format)."""
    p = yaml_file(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(dict(doc), fh, sort_keys=False)
    return p


def _modified(path: Path) -> datetime:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _from_yaml(path: Path, reason: str) -> OperatorInputs:
    return OperatorInputs(inputs=read_yaml(path), source="yaml", reason=reason, yaml_path=str(path))


def _from_row(row: OperatorInputsRow) -> OperatorInputs:
    return OperatorInputs(
        inputs=dict(row.inputs or {}),
        source="neon",
        version=row.version,
        valid_from=_aware(row.valid_from),
        valid_upto=_aware(row.valid_upto),
        author=row.author,
        origin=row.origin,
        note=row.note,
    )


def _why(exc: Exception) -> str:
    if isinstance(exc, ProgrammingError) or "no such table" in str(exc):  # PostgreSQL / SQLite
        return "The twin store has no operator_inputs table (alembic -n twin upgrade head, twin_0005)."
    return f"The twin store is not reachable ({type(exc).__name__})."


_PROCESS = object()


def _store(store: Any) -> TwinStore | None:
    return store_service.twin_store() if store is _PROCESS else store


# ------------------------------------------------------------------------------------------------------ reading
def load_current(store: Any = _PROCESS, *, yaml_path: str | Path | None = None) -> OperatorInputs:
    """The inputs in force: the current version in the twin store, else the YAML file (``reason`` says why).
    ``store``: default the process's store (``services.store``); None reads the file."""
    path = yaml_file(yaml_path)
    twin = _store(store)
    if twin is None:
        return _from_yaml(path, NO_STORE)
    try:
        row = twin.operator_inputs.current()
    except SQLAlchemyError as exc:
        log.warning("operator inputs from %s: %s", path.name, str(exc).splitlines()[0][:300])
        return _from_yaml(path, _why(exc) + " The inputs come from the YAML file.")
    return _from_yaml(path, EMPTY) if row is None else _from_row(row)


def as_of(when: datetime | date) -> datetime:
    """The instant a lookup by ``when`` means, in UTC: a date is the end of that day in IST (the last version saved
    on or before it), a naive time is IST, an aware time is itself."""
    if not isinstance(when, datetime):
        next_day = datetime.combine(when + timedelta(days=1), datetime.min.time(), IST)
        return (next_day - timedelta(microseconds=1)).astimezone(UTC)
    return (when.replace(tzinfo=IST) if when.tzinfo is None else when).astimezone(UTC)


def load_at(when: datetime | date, store: Any = _PROCESS) -> OperatorInputs | None:
    """The version valid at ``when`` (``as_of``: a date is the end of that IST day); None before the first version.
    Raises ``Unavailable`` without a usable store."""
    twin = _store(store)
    if twin is None:
        raise Unavailable(
            "The history of the operator inputs needs the twin store: set BF_TWIN_DB_URL or DATABASE_URL."
        )
    try:
        row = twin.operator_inputs.at(as_of(when))
    except SQLAlchemyError as exc:
        raise Unavailable(_why(exc)) from exc
    return None if row is None else _from_row(row)


# --------------------------------------------------------------------------------------------------------- token
_TOKENS: dict[tuple[str, int], tuple[float, str]] = {}
_TOKENS_LOCK = threading.Lock()


def _read_token(path: Path, twin: TwinStore | None) -> tuple[str, bool]:
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        stamp = 0
    if twin is None:
        return f"yaml:{stamp}", True
    try:
        version = twin.operator_inputs.current_version()
    except SQLAlchemyError:
        return f"yaml:{stamp}:unavailable", False
    return (f"neon:{version}", True) if version is not None else (f"yaml:{stamp}:empty", True)


def token(yaml_path: str | Path | None = None, *, store: Any = _PROCESS) -> str:
    """Key of the inputs in force (``neon:<version>`` or ``yaml:<mtime>[:why]``), for the readers' caches: one indexed
    query, trusted for 10 s (60 s after a failure, so an unreachable store costs one attempt a minute)."""
    path = yaml_file(yaml_path)
    twin = _store(store)
    key = (str(path), id(twin))
    now = time.monotonic()
    with _TOKENS_LOCK:
        hit = _TOKENS.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value, ok = _read_token(path, twin)
    with _TOKENS_LOCK:
        _TOKENS[key] = (now + (TOKEN_TTL_S if ok else TOKEN_RETRY_S), value)
    return value


def invalidate() -> None:
    """Forget the cached tokens (after a save in this process)."""
    with _TOKENS_LOCK:
        _TOKENS.clear()


# --------------------------------------------------------------------------------------------------------- write
def write(
    change: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    actor: Actor,
    origin: str = "streamlit",
    note: str | None = None,
    base_version: int | None = None,
    store: Any = _PROCESS,
    yaml_path: str | Path | None = None,
    at: datetime | None = None,
) -> Written:
    """Save ``change(current inputs)`` as the next version (validated; ``ValueError`` on a wrong type). The first save
    records the YAML as version 1. ``base_version``: the version the edit started from (``VersionConflict`` when
    another is current). Without a configured store the YAML file is written instead. ``Unavailable`` when the store
    cannot be written."""
    path = yaml_file(yaml_path)
    twin = _store(store)

    def apply(doc: dict[str, Any]) -> dict[str, Any]:
        return normalise(change(doc))

    if twin is None:
        doc = apply(read_yaml(path))
        written = write_yaml(path, doc)
        invalidate()
        current = replace(_from_yaml(path, NO_STORE), inputs=doc)
        return Written(current, True, str(written))

    def seed() -> Seed | None:
        doc = read_yaml(path)
        if not doc:
            return None
        return Seed(normalise(doc), _modified(path), note=f"Recorded from {path.name} at the first save")

    try:
        row, changed = twin.operator_inputs.write(
            apply, actor, origin=origin, note=note, base_version=base_version, seed=seed, at=at
        )
    except VersionConflict:
        raise
    except SQLAlchemyError as exc:
        raise Unavailable(_why(exc)) from exc
    invalidate()
    location = f"twin.operator_inputs version {row.version}" + ("" if changed else " (unchanged)")
    return Written(_from_row(row), changed, location)


def main(argv: list[str] | None = None) -> int:
    """``python -m services.operator_inputs [--seed]``: where the inputs come from; ``--seed`` records the YAML file
    as version 1 when the twin store holds no version yet (otherwise the first save does it)."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m services.operator_inputs", description=main.__doc__)
    parser.add_argument("--seed", action="store_true", help="record the YAML as version 1 if the store is empty")
    args = parser.parse_args(argv)
    if args.seed:
        if _store(_PROCESS) is None:
            print(NO_STORE)
            return 1
        user = os.environ.get("USERNAME") or os.environ.get("USER") or "operator_inputs.cli"
        result = write(lambda doc: doc, actor=Actor(name=user, role="admin"), origin="api")
        print(f"{result.location}: {'version 1 recorded from the YAML' if result.changed else 'nothing to do'}")
    current = load_current()
    print(
        f"source {current.source}, version {current.version}, valid from {current.valid_from}: {current.reason or ''}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
