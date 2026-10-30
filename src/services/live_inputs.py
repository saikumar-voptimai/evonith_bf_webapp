"""Live inputs of the optimisers through one TTL cache with an as-of time per group (SV-7.3, #326).

Four groups, each read once per TTL (``config/live_inputs.yml``) and served with the time of its data (``as_of``, UTC:
the newest stock entry, the newest lab sample used, the newest reading, the start of the version in force), not the
time of the read, and a ``stale`` flag against the group's stale limit:

* ``stock`` and ``rm_composition``: the stock and chemistry snapshots of page 9's context provider
  (``EvonithBmoContextProvider.get_stock_snapshot`` / ``get_chemistry_snapshot``, Neon), memoised here
  (``stock_read``, ``chemistry_read``). ``provider()`` is that provider with the two snapshots taken from the memo, so
  its ``build_ore_inputs`` (page 9's ore rows with ``stock_mt``, the yard of page 11 through ``load_yard``) composes the
  same values the panel shows without a second read; the composition itself stays the provider's.
* ``process_conditions``: the furnace-status snapshot (``services.status.current_snapshot_view``, one InfluxDB query
  per 10-s slot, InfluxDB recorded in ``source_health.HEALTH`` there): the blast, oxygen, steam, PCI, fuel-rate,
  top-gas, RAFT and production readings the optimisers use, with the plan day's Open-Meteo blast moisture.
* ``operator_inputs``: the version in force (``services.operator_inputs.load_current``: ``twin.operator_inputs``, the
  YAML file as fallback), keyed on ``operator_inputs.token()`` so a saved version is picked up within its TTL.

``snapshot()`` composes the four into one ``LiveInputsView`` with a ``snapshot_id`` (hash of the four as-of times) and
``read_at``; ``LiveInputsView.ref()`` is what a run stores (``Case.live_inputs`` of the Burden & Fuel Optimiser,
``CaseSources.live_inputs`` of the Blend Mix Optimiser), so a run's inputs can be matched with the panel's values.

The memoised reads are process-wide (``utils.memo``); a lock per group makes concurrent misses share one read. The
TTLs are read from the YAML at import (a change needs a restart); ``clear()`` empties every memo (tests, a refresh).
"""

from __future__ import annotations

import copy
import functools
import hashlib
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from services import operator_inputs
from services.operator_inputs import OperatorInputsDocument
from utils.memo import memo

log = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CONFIG_PATH = CONFIG_DIR / "live_inputs.yml"
BMO_SETTINGS_PATH = CONFIG_DIR / "setting_bmo.yml"
IST = "Asia/Kolkata"

GROUP_KEYS: tuple[str, ...] = ("stock", "rm_composition", "process_conditions", "operator_inputs")
GroupKey = Literal["stock", "rm_composition", "process_conditions", "operator_inputs"]

#: Catalogue keys of ``config/furnace_status.yml`` the optimisers take their process conditions from, display order.
PROCESS_KEYS: tuple[str, ...] = (
    "hot_blast_volume",
    "hbt",
    "blast_pressure",
    "top_pressure",
    "body_dp_total",
    "o2_injection",
    "oxygen_flow",
    "steam_injection",
    "pci_rate",
    "coke_rate",
    "nut_coke_rate",
    "fuel_rate",
    "co_utilization",
    "top_gas_h2",
    "raft",
    "production_rate",
    "permeability",
)
MOISTURE_TTL_S = 300.0  # the Open-Meteo value is daily; its file reads are not repeated per request


def clock() -> datetime:
    """Current time (UTC); tests replace it."""
    return datetime.now(UTC)


# ------------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class GroupSpec:
    """One group of ``config/live_inputs.yml``."""

    key: str
    label: str
    ttl_s: float
    stale_after_min: float | None  # None: never stale by age
    source: str


def read_config(path: Path = CONFIG_PATH) -> dict[str, GroupSpec]:
    """The groups of the YAML (every group of ``GROUP_KEYS`` must be present with a ``ttl_s``)."""
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    groups = dict((doc.get("live_inputs") or {}).get("groups") or {})
    out: dict[str, GroupSpec] = {}
    for key in GROUP_KEYS:
        g = groups.get(key)
        if not isinstance(g, Mapping) or "ttl_s" not in g:
            raise ValueError(f"{path.name}: group {key!r} needs ttl_s")
        stale = g.get("stale_after_min")
        out[key] = GroupSpec(
            key=key,
            label=str(g.get("label") or key),
            ttl_s=float(g["ttl_s"]),
            stale_after_min=None if stale is None else float(stale),
            source=str(g.get("source") or ""),
        )
    return out


SPECS: dict[str, GroupSpec] = read_config()


def spec(key: str) -> GroupSpec:
    return SPECS[key]


def _bmo_mtime() -> int:
    return BMO_SETTINGS_PATH.stat().st_mtime_ns if BMO_SETTINGS_PATH.exists() else 0


@memo(max_entries=2)
def _bmo_defaults(mtime_ns: int) -> dict[str, Any]:
    doc = yaml.safe_load(BMO_SETTINGS_PATH.read_text(encoding="utf-8")) if BMO_SETTINGS_PATH.exists() else {}
    bmo = dict((doc or {}).get("bmo") or {})
    data_sources = dict(bmo.get("data_sources") or {})
    return dict(
        chemistry_mode=str(bmo.get("chemistry_mode", "latest")),
        chemistry_window_days=int(bmo.get("chemistry_window_days", 30)),
        stock_table=str(data_sources.get("stock_table", "offline_feed.raw_material_stock")),
    )


def bmo_defaults() -> dict[str, Any]:
    """Page 9's chemistry mode and window (``setting_bmo.yml``): the panel's defaults for the composition group."""
    return _bmo_defaults(_bmo_mtime())


def chemistry_key(mode: str | None = None, window_days: int | None = None) -> tuple[str, int]:
    """The canonical (mode, window_days) of a chemistry read: ``avg`` or ``latest``, page 9's defaults for None."""
    defaults = bmo_defaults()
    m = str(mode or defaults["chemistry_mode"])
    return ("avg" if m == "avg" else "latest"), int(window_days or defaults["chemistry_window_days"])


# ------------------------------------------------------------------------------------------------------- reads
def provider_factory() -> Any:
    """A fresh ``EvonithBmoContextProvider`` for an upstream read; tests replace it with a counting fake."""
    from data.bmo.context_provider import EvonithBmoContextProvider

    return EvonithBmoContextProvider()


@dataclass(frozen=True)
class StockRead:
    """One stock read: wet stock by ore id (t), the provider's warnings and diagnostics (its ``material_rows`` carry
    the entry time and source per material), when it was read (UTC)."""

    stock_t: dict[str, float]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    read_at: datetime


def _read_stock() -> StockRead:
    p = provider_factory()
    stock_map, warnings = p.get_stock_snapshot()
    diagnostics = copy.deepcopy(dict(getattr(p, "_last_stock_diagnostics", None) or {}))
    return StockRead(
        stock_t={str(k): float(v) for k, v in dict(stock_map).items()},
        warnings=tuple(str(w) for w in warnings),
        diagnostics=diagnostics,
        read_at=clock(),
    )


@dataclass(frozen=True)
class ChemistryRead:
    """One chemistry read at (mode, window_days): ``OreChemistry`` by ore id, warnings, the provider's diagnostics
    (``material_rows``: source, table, rows used, sample time per material), when it was read (UTC)."""

    mode: str
    window_days: int
    chemistry: dict[str, Any]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
    read_at: datetime


def _read_chemistry(mode: str, window_days: int) -> ChemistryRead:
    p = provider_factory()
    chemistry, warnings = p.get_chemistry_snapshot(mode=mode, window_days=window_days)
    diagnostics = copy.deepcopy(dict(getattr(p, "_last_chemistry_diagnostics", None) or {}))
    return ChemistryRead(
        mode=mode,
        window_days=int(window_days),
        chemistry={str(k): v for k, v in dict(chemistry).items()},
        warnings=tuple(str(w) for w in warnings),
        diagnostics=diagnostics,
        read_at=clock(),
    )


@dataclass(frozen=True)
class OperatorInputsRead:
    """The operator inputs in force (``services.operator_inputs.OperatorInputs``), their as-of time (the version's
    start, or the YAML file's modification time) and when they were read (UTC)."""

    record: operator_inputs.OperatorInputs
    as_of: datetime | None
    read_at: datetime


def _mtime(path: str) -> datetime | None:
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime, UTC)
    except OSError:
        return None


def _read_operator_inputs(path: str, token: str) -> OperatorInputsRead:
    record = operator_inputs.load_current(yaml_path=path)
    as_of = record.valid_from if record.source == "neon" else _mtime(record.yaml_path or path)
    read = OperatorInputsRead(record=record, as_of=as_of, read_at=clock())
    # A version saved, or the store failing, between the token and the read: used once, not kept under this key.
    return operator_inputs.require_fit(token, record.provenance(), read)


_stock_cached = memo(ttl_seconds=SPECS["stock"].ttl_s, copy=True)(_read_stock)
_chemistry_cached = memo(ttl_seconds=SPECS["rm_composition"].ttl_s, copy=True)(_read_chemistry)
_operator_inputs_cached = memo(ttl_seconds=SPECS["operator_inputs"].ttl_s, copy=True)(_read_operator_inputs)
_locks = {key: threading.Lock() for key in ("stock", "chemistry", "operator_inputs")}


def stock_read() -> StockRead:
    """The stock snapshot, read once per ``stock.ttl_s`` for every reader of the process (concurrent misses wait for
    the one read); the caller gets its own copy."""
    with _locks["stock"]:
        return _stock_cached()


def chemistry_read(mode: str | None = None, window_days: int | None = None) -> ChemistryRead:
    """The chemistry snapshot at (mode, window_days) (page 9's defaults for None), read once per
    ``rm_composition.ttl_s`` per key."""
    m, days = chemistry_key(mode, window_days)
    with _locks["chemistry"]:
        return _chemistry_cached(m, days)


def operator_inputs_read(yaml_path: str | Path | None = None) -> OperatorInputsRead:
    """The operator inputs in force, read once per ``operator_inputs.ttl_s`` per version (``operator_inputs.token``)."""
    path = str(operator_inputs.yaml_file(yaml_path))
    token = operator_inputs.token(yaml_path)
    with _locks["operator_inputs"]:
        try:
            return _operator_inputs_cached(path, token)
        except operator_inputs.TokenMismatch as mismatch:
            return mismatch.value


@functools.cache
def _shared_provider_class() -> type:
    from data.bmo.context_provider import EvonithBmoContextProvider

    class SharedReadsProvider(EvonithBmoContextProvider):
        """Page 9's context provider whose stock and chemistry snapshots (and the operator inputs) are the process's
        memoised reads. ``build_ore_inputs`` and every other method are inherited unchanged: they compose from these
        snapshots exactly as the provider does, so page 9's runs, page 11's yard and the live-inputs panel show one
        set of values per TTL. The diagnostics the provider keeps after a read (``get_data_diagnostics``) are restored
        from the memo."""

        def get_stock_snapshot(self) -> tuple[dict[str, float], list[str]]:
            read = stock_read()
            self._last_stock_diagnostics = dict(read.diagnostics)
            return dict(read.stock_t), list(read.warnings)

        def get_chemistry_snapshot(self, mode: str = "latest", window_days: int | None = None) -> tuple[dict, list]:
            cfg = self.settings.get("data_sources", {})
            days = int(window_days or cfg.get("chemistry_time_range_days", 30))  # the provider's own default
            read = chemistry_read(mode, days)
            self._last_chemistry_diagnostics = dict(read.diagnostics)
            return dict(read.chemistry), list(read.warnings)

        def get_operator_inputs(self, at: datetime | None = None) -> Any:
            if at is not None:
                return super().get_operator_inputs(at)
            return operator_inputs_read().record

    return SharedReadsProvider


def provider(**kwargs: Any) -> Any:
    """``EvonithBmoContextProvider`` on the shared reads (``SharedReadsProvider``); ``kwargs`` as the provider's."""
    return _shared_provider_class()(**kwargs)


def load_yard(cfg: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """``data.burden_engine.inputs.load_yard`` on the shared reads: page 11's yard from the same stock, chemistry and
    operator inputs the panel shows (``provider`` may be given, e.g. a stand-in in tests)."""
    from data.burden_engine.inputs import load_yard as _load_yard

    kwargs.setdefault("provider", provider())
    return _load_yard(cfg, **kwargs)


# ------------------------------------------------------------------------------------------------------- views
class GroupLimits(BaseModel):
    ttl_s: float = Field(..., description="How long one upstream read serves every reader of the process, s")
    stale_after_min: float | None = Field(
        None, description="Age of the data from which the group is stale; null: never stale by age, min"
    )


class GroupView(BaseModel):
    """One live-input group: where it comes from, the time of its data, whether it is stale."""

    key: GroupKey
    label: str
    source: str = Field(..., description="Where the values come from (table, version or file)")
    as_of: datetime | None = Field(
        None,
        description=(
            "Time of the data (UTC): the newest stock entry, the newest lab sample used, the newest reading, the "
            "start of the version in force; null without data"
        ),
    )
    read_at: datetime = Field(..., description="When the source was read (UTC); readers within ttl_s share this read")
    age_min: float | None = Field(None, description="Age of the data at the response, min")
    stale: bool = Field(..., description="Older than stale_after_min, without data, or the last read failed")
    stale_reason: str | None = Field(None, description="Why the group is stale")
    limits: GroupLimits = Field(..., description="The group's TTL and stale limit (config/live_inputs.yml)")
    warnings: list[str] = Field(default_factory=list, description="Warnings of the read")
    error: str | None = Field(None, description="The failure of the last read; the values are then a fallback")


class StockMaterial(BaseModel):
    id: str = Field(..., description="Ore id of bmo_ore_mapping.yml")
    name: str
    code: str = Field(..., description="Material code of the stock table")
    stock_t: float = Field(..., description="Wet stock in the yard, t")
    source: str = Field(
        ...,
        description=(
            "offline_db: the latest entry; offline_db_null_as_zero / offline_db_missing_as_zero: no entry, counted as "
            "no stock; sinter_fallback: planning stock"
        ),
    )
    at: datetime | None = Field(None, description="Time of the stock entry (UTC); null without an entry")


class StockGroup(GroupView):
    table: str = Field(..., description="Neon table of the stock entries")
    materials: list[StockMaterial] = Field(default_factory=list, description="Every configured ferrous material")
    total_t: float = Field(..., description="Stock of the materials with an entry, t")


class RmMaterial(BaseModel):
    id: str = Field(..., description="Ore id of bmo_ore_mapping.yml")
    name: str
    code: str = Field(..., description="Material code of the chemistry table")
    source: str = Field(
        ...,
        description=(
            "offline_db_latest_used: the latest sample before the latest charge; offline_db_latest_no_usage / "
            "_no_prior_sample: the latest sample; offline_db_avg_non_zero: the window's average; fallback: configured "
            "values"
        ),
    )
    table: str = Field(..., description="Neon chemistry table")
    rows_used: int = Field(..., description="Lab rows behind the values, count")
    sample_at: datetime | None = Field(None, description="Time of the sample used, or the newest averaged (UTC)")
    latest_used_at: datetime | None = Field(None, description="Latest charge of the material (UTC)")
    fe_t_pct: float = Field(..., description="Total Fe, dry, %")
    moisture_pct: float = Field(..., description="Moisture, %")
    feo_pct: float = Field(..., description="FeO, dry, %")
    sio2_pct: float = Field(..., description="SiO₂, dry, %")
    al2o3_pct: float = Field(..., description="Al₂O₃, dry, %")
    cao_pct: float = Field(..., description="CaO, dry, %")
    mgo_pct: float = Field(..., description="MgO, dry, %")
    mno_pct: float = Field(..., description="MnO, dry, %")
    tio2_pct: float = Field(..., description="TiO₂, dry, %")
    p_pct: float = Field(..., description="P, dry, %")
    na2o_pct: float = Field(..., description="Na₂O, dry, %")
    k2o_pct: float = Field(..., description="K₂O, dry, %")
    loi_pct: float = Field(..., description="Loss on ignition, dry, %")


class RmGroup(GroupView):
    chemistry_mode: Literal["latest", "avg"] = Field(
        ..., description="latest: the sample used before the latest charge; avg: the window's non-zero average"
    )
    window_days: int = Field(..., description="Chemistry window, d")
    materials: list[RmMaterial] = Field(default_factory=list, description="Every configured ferrous material")


class ProcessValue(BaseModel):
    key: str = Field(..., description="Catalogue key of config/furnace_status.yml")
    label: str
    value: float | None = Field(None, description="Latest reading of the status window, unrounded, in unit")
    unit: str = Field(..., description="Unit of value and setpoint; empty when not confirmed")
    decimals: int = Field(..., description="Decimals the V-Board shows, count")
    at: datetime | None = Field(None, description="Time of the reading (UTC)")
    state: str = Field(..., description="ok, low, high, stale, no_data, fetch_failed or no_source")
    setpoint: float | None = Field(None, description="Latest setpoint where the parameter has one, in unit")


class BlastMoisture(BaseModel):
    value_g_nm3: float = Field(..., description="Blast moisture of the plan day, g/Nm³")
    source: str = Field(..., description="Open-Meteo daily RH, its day-of-year mean, or the monthly table")
    day: date = Field(..., description="Plan day (IST)")


class ProcessGroup(GroupView):
    telemetry: Literal["live", "partial", "offline"] = Field(..., description="Telemetry of the status snapshot")
    parameters: list[ProcessValue] = Field(default_factory=list, description="PROCESS_KEYS, display order")
    blast_moisture: BlastMoisture | None = Field(None, description="The plan day's blast moisture; null if unread")


class OperatorInputsGroup(GroupView):
    version: int | None = Field(None, description="Version of twin.operator_inputs; null for the YAML file, count")
    author: str | None = Field(None, description="User who saved the version")
    origin: str | None = Field(None, description="seed, streamlit or api")
    reason: str | None = Field(None, description="Why the YAML file is in force")
    inputs: OperatorInputsDocument = Field(..., description="The document in force (units in the field descriptions)")


class LiveInputsRef(BaseModel):
    """What a run stores about the live inputs it was built from."""

    snapshot_id: str = Field(..., description="Hash of the four as-of times (16 hex digits)")
    read_at: datetime = Field(..., description="When the snapshot was composed (UTC)")
    as_of: dict[str, datetime | None] = Field(..., description="As-of time per group (UTC)")
    stale: list[str] = Field(default_factory=list, description="Groups stale at the snapshot")
    chemistry_mode: str = Field(..., description="Chemistry mode of the composition group")
    chemistry_window_days: int = Field(..., description="Chemistry window of the composition group, d")


class LiveInputsView(BaseModel):
    """The four live-input groups read once per TTL, with one snapshot id."""

    snapshot_id: str = Field(..., description="Hash of the four as-of times (16 hex digits): equal data, equal id")
    read_at: datetime = Field(..., description="When the snapshot was composed (UTC)")
    stale: list[str] = Field(default_factory=list, description="Keys of the groups that are stale")
    stock: StockGroup
    rm_composition: RmGroup
    process_conditions: ProcessGroup
    operator_inputs: OperatorInputsGroup

    def groups(self) -> list[GroupView]:
        return [getattr(self, key) for key in GROUP_KEYS]

    def ref(self) -> LiveInputsRef:
        return LiveInputsRef(
            snapshot_id=self.snapshot_id,
            read_at=self.read_at,
            as_of={g.key: g.as_of for g in self.groups()},
            stale=list(self.stale),
            chemistry_mode=self.rm_composition.chemistry_mode,
            chemistry_window_days=self.rm_composition.window_days,
        )

    @property
    def yard_key(self) -> str:
        """Identity of the reads a yard is composed from (stock, chemistry, operator inputs): a cache key."""
        return "|".join(getattr(self, k).read_at.isoformat() for k in ("stock", "rm_composition", "operator_inputs"))


# --------------------------------------------------------------------------------------------------- composing
def _utc(value: Any) -> datetime | None:
    """A time as UTC (a naive time is IST, as the plant's); None for nothing or an unreadable value."""
    if value is None or value == "":
        return None
    import pandas as pd

    try:
        if pd.isna(value):
            return None
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    return ts.tz_convert("UTC").to_pydatetime()


def _latest(times: Iterable[Any]) -> datetime | None:
    found = [t for t in (_utc(v) for v in times) if t is not None]
    return max(found) if found else None


def _age_min(as_of: datetime | None, now: datetime) -> float | None:
    return None if as_of is None else (now - as_of).total_seconds() / 60.0


def _freshness(
    key: str, as_of: datetime | None, now: datetime, *, error: str | None = None, reason: str | None = None
) -> dict[str, Any]:
    """The shared fields of a group view: limits, age and the stale verdict (a failed read, no data, or the age
    beyond the limit; ``reason`` adds a stale reason of the source itself)."""
    s = spec(key)
    age = _age_min(as_of, now)
    why: str | None = None
    if error:
        why = f"The last read failed: {error}"
    elif reason:
        why = reason
    elif as_of is None:
        why = "No data"
    elif s.stale_after_min is not None and age is not None and age >= s.stale_after_min:
        why = f"Data {age / 60:.1f} h old, limit {s.stale_after_min / 60:g} h"
    return dict(
        key=key,
        label=s.label,
        as_of=as_of,
        age_min=age,
        stale=why is not None,
        stale_reason=why,
        limits=GroupLimits(ttl_s=s.ttl_s, stale_after_min=s.stale_after_min),
        error=error,
    )


def _first(warnings: Iterable[str], prefix: str) -> str | None:
    return next((w for w in warnings if w.startswith(prefix)), None)


def stock(now: datetime | None = None) -> StockGroup:
    """The stock group: the latest entry per configured material (``stock_read``); as of the newest entry."""
    now = now or clock()
    read = stock_read()
    rows = [r for r in read.diagnostics.get("material_rows") or [] if isinstance(r, Mapping)]
    materials = [
        StockMaterial(
            id=str(r.get("ore_id", "")),
            name=str(r.get("display_name") or r.get("ore_id", "")),
            code=str(r.get("material_code") or ""),
            stock_t=float(r.get("stock_mt") or 0.0),
            source=str(r.get("source") or ""),
            at=_utc(r.get("timestamp")),
        )
        for r in rows
    ]
    table = str(read.diagnostics.get("table") or bmo_defaults()["stock_table"])
    error = _first(read.warnings, "Stock query failed")
    return StockGroup(
        **_freshness("stock", _latest(m.at for m in materials), now, error=error),
        source=f"Neon {table}",
        read_at=read.read_at,
        warnings=list(read.warnings),
        table=table,
        materials=materials,
        total_t=sum(m.stock_t for m in materials if m.source == "offline_db"),
    )


_CHEM_FIELDS = (
    "fe_t_pct",
    "moisture_pct",
    "feo_pct",
    "sio2_pct",
    "al2o3_pct",
    "cao_pct",
    "mgo_pct",
    "mno_pct",
    "tio2_pct",
    "p_pct",
    "na2o_pct",
    "k2o_pct",
    "loi_pct",
)


def rm_composition(mode: str | None = None, window_days: int | None = None, now: datetime | None = None) -> RmGroup:
    """The composition group: page 9's chemistry snapshot per configured material (``chemistry_read``); as of the
    newest lab sample used."""
    now = now or clock()
    read = chemistry_read(mode, window_days)
    rows = [r for r in read.diagnostics.get("material_rows") or [] if isinstance(r, Mapping)]
    materials = []
    for r in rows:
        chem = read.chemistry.get(str(r.get("ore_id", "")))
        values = {f: float(r.get(f, getattr(chem, f, 0.0)) or 0.0) for f in _CHEM_FIELDS}
        materials.append(
            RmMaterial(
                id=str(r.get("ore_id", "")),
                name=str(r.get("display_name") or r.get("ore_id", "")),
                code=str(r.get("material_code") or ""),
                source=str(r.get("source") or ""),
                table=str(r.get("table") or ""),
                rows_used=int(r.get("rows_used") or 0),
                sample_at=_utc(r.get("sample_timestamp")),
                latest_used_at=_utc(r.get("latest_used_time")),
                **values,
            )
        )
    error = _first(read.warnings, "Chemistry query failed")
    tables = sorted({m.table for m in materials if m.table})
    return RmGroup(
        **_freshness("rm_composition", _latest(m.sample_at for m in materials), now, error=error),
        source="Neon " + (", ".join(tables) if tables else "chemistry tables"),
        read_at=read.read_at,
        warnings=list(read.warnings),
        chemistry_mode=read.mode,  # type: ignore[arg-type]
        window_days=read.window_days,
        materials=materials,
    )


def _read_blast_moisture() -> BlastMoisture | None:
    try:
        from utils.coke_energy.weather import plan_blast_moisture

        g, label, day = plan_blast_moisture()
    except Exception as exc:  # noqa: BLE001 - the group is served without the moisture
        log.warning("live inputs: blast moisture not read (%s: %s)", type(exc).__name__, exc)
        return None
    return BlastMoisture(value_g_nm3=float(g), source=str(label), day=day.date())


_blast_moisture_cached = memo(ttl_seconds=MOISTURE_TTL_S)(_read_blast_moisture)


def process_conditions(now: datetime | None = None) -> ProcessGroup:
    """The process group: the furnace-status snapshot's readings of ``PROCESS_KEYS`` (one InfluxDB query per 10-s
    slot, ``services.status``) and the plan day's blast moisture; as of the newest reading of the snapshot."""
    from services import source_health
    from services import status as status_service

    now = now or clock()
    view = status_service.current_snapshot_view()
    by_key = {p.key: p for p in view.parameters}
    parameters = [
        ProcessValue(
            key=p.key,
            label=p.label,
            value=p.value,
            unit=p.unit,
            decimals=p.decimals,
            at=p.at,
            state=p.state,
            setpoint=p.setpoint,
        )
        for p in (by_key.get(k) for k in PROCESS_KEYS)
        if p is not None
    ]
    fresh = view.freshness
    error = None
    if fresh.failed_sources and fresh.telemetry == "offline":
        error = source_health.HEALTH.reason("influxdb") or (
            "InfluxDB measurements not read: " + ", ".join(fresh.failed_sources)
        )
    reason = "No reading within the status window" if fresh.stale and error is None else None
    warnings = [f"Measurement not read: {m}" for m in fresh.failed_sources] if error is None else []
    return ProcessGroup(
        **_freshness("process_conditions", fresh.data_at, now, error=error, reason=reason),
        source=f"InfluxDB process_params ({fresh.source} snapshot)",
        read_at=fresh.computed_at,
        warnings=warnings,
        telemetry=fresh.telemetry,
        parameters=parameters,
        blast_moisture=_blast_moisture_cached(),
    )


def operator_inputs_current(yaml_path: str | Path | None = None, now: datetime | None = None) -> OperatorInputsGroup:
    """The operator-inputs group: the version in force (``operator_inputs_read``); as of the version's start (the
    YAML file's modification time when the file is in force); stale while the file stands in for a failed store."""
    now = now or clock()
    read = operator_inputs_read(yaml_path)
    rec = read.record
    fallback = operator_inputs.fallback_warning(rec.provenance())
    if rec.source == "neon":
        source = f"twin.operator_inputs version {rec.version}"
    else:
        source = f"YAML file {Path(rec.yaml_path).name if rec.yaml_path else 'bmo_operator_inputs.yml'}"
    return OperatorInputsGroup(
        **_freshness("operator_inputs", read.as_of, now, reason=fallback),
        source=source,
        read_at=read.read_at,
        warnings=[fallback] if fallback else [],
        version=rec.version,
        author=rec.author,
        origin=rec.origin,
        reason=rec.reason,
        inputs=OperatorInputsDocument.model_validate(operator_inputs.prune(rec.inputs)),
    )


def snapshot_id(as_of: Iterable[datetime | None]) -> str:
    """Hash of the as-of times (GROUP_KEYS order): the first 16 hex digits of SHA-256 over their ISO forms."""
    text = "|".join("" if t is None else _utc(t).isoformat() for t in as_of)  # type: ignore[union-attr]
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def snapshot(
    *,
    chemistry_mode: str | None = None,
    chemistry_window_days: int | None = None,
    yaml_path: str | Path | None = None,
    now: datetime | None = None,
) -> LiveInputsView:
    """The four groups as one view (each from its own TTL cache), with the snapshot id of their as-of times."""
    now = now or clock()
    groups = (
        stock(now),
        rm_composition(chemistry_mode, chemistry_window_days, now),
        process_conditions(now),
        operator_inputs_current(yaml_path, now),
    )
    return LiveInputsView(
        snapshot_id=snapshot_id(g.as_of for g in groups),
        read_at=now,
        stale=[g.key for g in groups if g.stale],
        stock=groups[0],
        rm_composition=groups[1],
        process_conditions=groups[2],
        operator_inputs=groups[3],
    )


def snapshot_ref(**kwargs: Any) -> tuple[LiveInputsRef | None, str]:
    """``(snapshot().ref(), yard_key)`` for a run; ``(None, "")`` with a warning when the snapshot cannot be composed
    (the run goes on without a live-inputs reference)."""
    try:
        view = snapshot(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a run must not fail on the panel's bookkeeping
        log.warning("live inputs: snapshot not composed (%s: %s)", type(exc).__name__, exc)
        return None, ""
    return view.ref(), view.yard_key


def clear() -> None:
    """Empty every memo (tests, a manual refresh)."""
    for fn in (_stock_cached, _chemistry_cached, _operator_inputs_cached, _blast_moisture_cached, _bmo_defaults):
        fn.clear()


__all__ = [
    "GROUP_KEYS",
    "PROCESS_KEYS",
    "SPECS",
    "BlastMoisture",
    "ChemistryRead",
    "GroupLimits",
    "GroupSpec",
    "GroupView",
    "LiveInputsRef",
    "LiveInputsView",
    "OperatorInputsGroup",
    "OperatorInputsRead",
    "ProcessGroup",
    "ProcessValue",
    "RmGroup",
    "RmMaterial",
    "StockGroup",
    "StockMaterial",
    "StockRead",
    "bmo_defaults",
    "chemistry_key",
    "chemistry_read",
    "clear",
    "clock",
    "load_yard",
    "operator_inputs_current",
    "operator_inputs_read",
    "process_conditions",
    "provider",
    "provider_factory",
    "read_config",
    "rm_composition",
    "snapshot",
    "snapshot_id",
    "snapshot_ref",
    "spec",
    "stock",
    "stock_read",
]
