"""BF-02 process fact register: the versioned rules the burden engine rests on.

The register (``src/config/facts_register.json``) is the machine copy of *BF-02 process fact register v1.1*. Every
coefficient the engine uses that is not part of the two-zone balance itself (burden-resistance rules, the packed-bed
exponent and base point, the enrichment-tag factor, the plausibility bands and the default limits) is read from here,
so a monthly refit or a test result changes a versioned file, not code.

Retired facts (status ``Retired``, e.g. COKE-17 the webapp hand coefficients and DATA-10 the regressed RAFT) can be
displayed but never used: :meth:`FactRegister.param` refuses them.

Overlay (SV-4.2, #305): administrators change a fact's value, band or status in the admin space; an approved version
of ``twin.facts_version`` is an overlay on this file (``apply_overlay``). ``load_register()`` returns the file with the
latest approved overlay made against the file's version (``services.facts.approved_overlay``, re-read every 60 s), else
the file alone; ``load_register(path)`` and ``load_register(overlay=False)`` read the file only.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REGISTER_PATH = Path(__file__).resolve().parents[2] / "config" / "facts_register.json"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "schemas" / "facts_register.schema.json"

RETIRED = "Retired"

#: Fields of a fact an overlay changes (the admin editor): value, band ([lo, hi]: ``params`` lo / hi of a plausibility
#: band, else ``range``) and status.
OVERLAY_FIELDS = ("value", "band", "status")

log = logging.getLogger(__name__)


class RetiredFactError(KeyError):
    """A retired fact was asked for a value the engine would use."""


class OverlayError(ValueError):
    """An overlay that does not apply to the register: unknown fact or field, or a value of the wrong type."""


@dataclass(frozen=True)
class Fact:
    """One register row."""

    id: str
    group: str
    statement: str
    value: float | None
    status: str
    grade: str
    unit: str = ""
    se: float | None = None
    range: tuple[float, float] | None = None
    basis: str = ""
    literature: str = ""
    validity_window: str = ""
    next_test: str = ""
    version: str = ""
    date: str = ""
    note: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def retired(self) -> bool:
        return self.status == RETIRED

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Fact":
        rng = d.get("range")
        return cls(
            id=d["id"], group=d["group"], statement=d["statement"], value=d.get("value"), status=d["status"],
            grade=d.get("grade", ""), unit=d.get("unit", ""), se=d.get("se"), range=tuple(rng) if rng else None,
            basis=d.get("basis", ""), literature=d.get("literature", ""), validity_window=d.get("validity_window", ""),
            next_test=d.get("next_test", ""), version=d.get("version", ""), date=d.get("date", ""),
            note=d.get("note", ""), params=dict(d.get("params") or {}),
        )


class FactRegister:
    """Read-only view of the register with guarded access for the engine."""

    def __init__(self, data: dict[str, Any], overlay_version: int | None = None):
        self.name: str = data["name"]
        self.version: str = data["version"]
        #: Approved overlay version applied to the file (``twin.facts_version``), None for the file alone.
        self.overlay_version = overlay_version
        self.date: str = data["date"]
        self.statuses: list[str] = list(data["statuses"])
        self.revision_log: list[dict[str, str]] = list(data.get("revision_log", []))
        self._facts = {f["id"]: Fact.from_dict(f) for f in data["facts"]}
        self._check()

    @property
    def label(self) -> str:
        """``1.8``, or ``1.8 + overlay 3`` with an approved overlay."""
        return self.version if self.overlay_version is None else f"{self.version} + overlay {self.overlay_version}"

    # ---------------------------------------------------------------------------------------------- access
    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def __getitem__(self, fact_id: str) -> Fact:
        return self._facts[fact_id]

    def facts(self, group: str | None = None, include_retired: bool = True) -> list[Fact]:
        return [f for f in self._facts.values()
                if (group is None or f.group == group) and (include_retired or not f.retired)]

    def active(self, fact_id: str) -> Fact:
        """The fact, refusing a retired one."""
        f = self._facts[fact_id]
        if f.retired:
            raise RetiredFactError(f"{fact_id} is retired ({f.note or f.basis}); the engine must not use it")
        return f

    def value(self, fact_id: str) -> float:
        v = self.active(fact_id).value
        if v is None:
            raise ValueError(f"{fact_id} has no numeric value")
        return float(v)

    def param(self, fact_id: str, key: str, default: Any = None) -> Any:
        p = self.active(fact_id).params
        if key not in p and default is None:
            raise KeyError(f"{fact_id} has no parameter {key!r}")
        return p.get(key, default)

    def band(self, fact_id: str) -> tuple[float, float]:
        p = self.active(fact_id).params
        return float(p["lo"]), float(p["hi"])

    # ---------------------------------------------------------------------------------------------- engine view
    def thumb_rules(self) -> dict[str, Any]:
        """Burden-resistance rules and hydraulics constants in the form the engine uses."""
        base = dict(self.param("FLOW-01", "base"))
        rules = {self.param(fid, "engine_key"): self.value(fid)
                 for fid in ("FLOW-02", "FLOW-03", "FLOW-04", "FLOW-05", "FLOW-06", "FLOW-07")}
        return dict(rules, n_flow=self.value("FLOW-01"), base=base, p_atm=float(self.param("FLOW-01", "p_atm")),
                    enrich_tag_per_o2_pct=self.value("DATA-02"),
                    valid_max=dict(pellet=float(self.param("FLOW-02", "valid_max")),
                                   gomti=float(self.param("FLOW-03", "valid_max"))))

    def default_limits(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for fid in ("ASM-11", "ASM-12", "ASM-13"):
            out.update({k: float(v) for k, v in self.active(fid).params.items()})
        return out

    # ---------------------------------------------------------------------------------------------- integrity
    def _check(self) -> None:
        for f in self._facts.values():
            if f.status not in self.statuses:
                raise ValueError(f"{f.id}: unknown status {f.status!r}")
            if not f.id.startswith(f.group + "-"):
                raise ValueError(f"{f.id}: id does not match group {f.group}")
        if not self.revision_log or self.revision_log[0].get("version") != self.version:
            raise ValueError("revision log must start with the register's current version")

    def table(self, ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Rows for display (id, statement, value, unit, +/-, status, grade, next test, version)."""
        rows = []
        for f in (self._facts[i] for i in ids) if ids is not None else self._facts.values():
            rows.append(dict(id=f.id, statement=f.statement, value=f.value, unit=f.unit,
                             uncertainty=(f"± {f.se:g}" if f.se is not None else
                                          (f"{f.range[0]:g} to {f.range[1]:g}" if f.range else "")),
                             status=f.status, grade=f.grade, next_test=f.next_test, version=f.version))
        return rows


    def model_table(self) -> list[dict[str, Any]]:
        """MOD and VAL rows with the Fixed-omega and STRATA-omega values side by side (``utils.burden_engine.model_facts``)."""
        rows = []
        for f in self._facts.values():
            if f.group not in ("MOD", "VAL"):
                continue
            p = f.params
            row = dict(id=f.id, statement=f.statement, unit=f.unit, fixed_omega=p.get("fixed_omega"),
                       strata_omega=p.get("strata_omega"), sigma_strata=p.get("sigma_strata", f.se), status=f.status)
            if f.group == "VAL":
                row.update(realised=p.get("realised"), n=p.get("n"), rmse_fixed=p.get("rmse_fixed"),
                           rmse_strata=p.get("rmse_strata"), skill=p.get("skill"))
            rows.append(row)
        return rows


def load_register_data(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or REGISTER_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# -------------------------------------------------------------------------------------------------- overlay
def fact_band(fact: Mapping[str, Any]) -> list[float] | None:
    """[lo, hi] of a fact: ``params`` lo / hi of a plausibility band (BAND-01..03), else ``range``, else None."""
    p = fact.get("params") or {}
    if "lo" in p and "hi" in p:
        return [float(p["lo"]), float(p["hi"])]
    rng = fact.get("range")
    return [float(rng[0]), float(rng[1])] if rng else None


def band_kind(fact: Mapping[str, Any]) -> str:
    """``plausibility`` when the band is ``params`` lo / hi (the engine's bands), else ``range``."""
    p = fact.get("params") or {}
    return "plausibility" if "lo" in p and "hi" in p else "range"


def _number(fid: str, name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise OverlayError(f"{fid}: {name} must be a finite number, got {v!r}")
    return float(v)


def apply_overlay(data: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of the register document ``data`` with ``overlay`` applied: ``{fact id: {value?, band?, status?}}``.

    ``value`` a number or None; ``band`` [lo, hi] with lo <= hi (None removes a ``range``; a plausibility band cannot
    be removed); ``status`` text (checked against the statuses by ``FactRegister``). ``OverlayError`` on an unknown
    fact or field or a wrong type."""
    out = copy.deepcopy(dict(data))
    facts = {f["id"]: f for f in out["facts"]}
    for fid, patch in overlay.items():
        if fid not in facts:
            raise OverlayError(f"{fid}: no such fact in register v{out.get('version')}")
        if not isinstance(patch, Mapping) or not patch:
            raise OverlayError(f"{fid}: a change sets at least one of {', '.join(OVERLAY_FIELDS)}")
        unknown = sorted(set(patch) - set(OVERLAY_FIELDS))
        if unknown:
            raise OverlayError(f"{fid}: {', '.join(unknown)} cannot be changed (only {', '.join(OVERLAY_FIELDS)})")
        f = facts[fid]
        if "value" in patch:
            f["value"] = None if patch["value"] is None else _number(fid, "value", patch["value"])
        if "band" in patch:
            band = patch["band"]
            if band is None:
                if band_kind(f) == "plausibility":
                    raise OverlayError(f"{fid}: a plausibility band cannot be removed")
                f.pop("range", None)
            else:
                if not isinstance(band, (list, tuple)) or len(band) != 2:
                    raise OverlayError(f"{fid}: band must be [lo, hi], got {band!r}")
                lo, hi = _number(fid, "band lo", band[0]), _number(fid, "band hi", band[1])
                if lo > hi:
                    raise OverlayError(f"{fid}: band lo {lo:g} lies above hi {hi:g}")
                if band_kind(f) == "plausibility":
                    f["params"] = {**f["params"], "lo": lo, "hi": hi}
                else:
                    f["range"] = [lo, hi]
        if "status" in patch:
            status = patch["status"]
            if not isinstance(status, str) or not status:
                raise OverlayError(f"{fid}: status must be text, got {status!r}")
            f["status"] = status
    return out


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def register_problems(data: Mapping[str, Any]) -> list[str]:
    """Why ``data`` is not a usable register: JSON-schema errors (``config/schemas/facts_register.schema.json``),
    then the register's own checks (statuses, ids, revision log), then the engine's reads (``engine_problems``).
    Empty when it is usable."""
    from jsonschema import Draft202012Validator

    out = []
    for e in sorted(Draft202012Validator(_schema()).iter_errors(data), key=lambda e: list(map(str, e.path))):
        path = list(e.path)
        where = "/".join(str(p) for p in path)
        if len(path) >= 2 and path[0] == "facts" and isinstance(path[1], int):
            fid = (data["facts"][path[1]] or {}).get("id", path[1])
            where = "/".join([str(fid)] + [str(p) for p in path[2:]])
        out.append(f"{where or 'register'}: {e.message}")
    if out:
        return out
    try:
        reg = FactRegister(dict(data))
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    return engine_problems(reg)


def engine_problems(reg: FactRegister) -> list[str]:
    """The guarded reads the engine makes when it is built (hydraulics rules, default limits, plausibility bands):
    a retired fact among them, or a missing parameter, would stop every optimiser run."""
    out = []
    for name, read in (("hydraulics rules", reg.thumb_rules), ("default limits", reg.default_limits),
                       ("fuel-rate band", lambda: reg.band("BAND-01")), ("HMT band", lambda: reg.band("BAND-02")),
                       ("Si band", lambda: reg.band("BAND-03"))):
        try:
            read()
        except RetiredFactError as exc:
            out.append(f"{exc.args[0]}: the engine reads it for its {name}; retiring it needs a code change first")
        except (KeyError, TypeError, ValueError) as exc:
            out.append(f"the engine's {name} cannot be read: {exc}")
    return out


@dataclass(frozen=True)
class ApprovedOverlay:
    """The overlay in force: an approved version of ``twin.facts_version`` made against the file ``base_version``."""

    version: int
    base_version: str
    overlay: Mapping[str, Any]
    content_hash: str


OverlaySource = Callable[[], "ApprovedOverlay | None"]
_DEFAULT_SOURCE: Any = object()
_source: Any = _DEFAULT_SOURCE


def set_overlay_source(source: Any = _DEFAULT_SOURCE) -> None:
    """Where ``load_register()`` finds the approved overlay: a callable returning an ``ApprovedOverlay`` or None,
    None (the file only) or, without an argument, the default (``services.facts.approved_overlay``, the twin store
    of the process)."""
    global _source
    _source = source


def _approved() -> ApprovedOverlay | None:
    source = _source
    if source is _DEFAULT_SOURCE:
        try:
            from services.facts import approved_overlay as source
        except ImportError as exc:  # the engine without the service layer (vendored copies, scripts)
            log.debug("fact register overlay unavailable: %s", exc)
            return None
    if source is None:
        return None
    try:
        return source()
    except Exception as exc:  # noqa: BLE001 - a store that fails leaves the file in force
        log.warning("fact register overlay not read (%s): the repository file is in force", exc)
        return None


@lru_cache(maxsize=4)
def _cached(path: str) -> FactRegister:
    return FactRegister(load_register_data(path))


@lru_cache(maxsize=8)
def _overlaid(path: str, version: int, content_hash: str, overlay_json: str) -> FactRegister:
    return FactRegister(apply_overlay(load_register_data(path), json.loads(overlay_json)), overlay_version=version)


def load_register(path: str | Path | None = None, *, overlay: bool = True) -> FactRegister:
    """The register the engine uses: the shipped file with the latest approved overlay made against its version
    (``twin.facts_version``), else the file alone. ``path`` (or ``overlay=False``) reads that file only. Cached per
    file and overlay version."""
    if path is not None or not overlay:
        return _cached(str(Path(path or REGISTER_PATH).resolve()))
    file = str(Path(REGISTER_PATH).resolve())
    base = _cached(file)
    approved = _approved()
    if approved is None:
        return base
    if approved.base_version != base.version:
        log.warning("fact register overlay %s is for v%s, the file is v%s: the file is in force",
                    approved.version, approved.base_version, base.version)
        return base
    try:
        return _overlaid(file, approved.version, approved.content_hash,
                         json.dumps(approved.overlay, sort_keys=True))
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("fact register overlay %s does not apply (%s): the file is in force", approved.version, exc)
        return base


# Facts every engine result rests on; the page shows these with their status.
ENGINE_FACTS = ("ASM-01", "ASM-02", "ASM-03", "ASM-04", "ASM-05", "ASM-06", "ASM-08", "ASM-09",
                "FLOW-01", "FLOW-02", "FLOW-03", "FLOW-04", "FLOW-05", "FLOW-06", "DATA-02", "DATA-03",
                "BAND-01", "BAND-02", "BAND-03", "GAS-10", "ASM-18", "GAS-13", "GAS-14", "GAS-15", "GAS-18",
                "GAS-19", "GAS-20", "GAS-21", "GAS-22", "GAS-23", "GAS-25", "DATA-11", "ASM-19", "ASM-20")
