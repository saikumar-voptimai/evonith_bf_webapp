"""BF-02 process fact register: the versioned rules the burden engine rests on.

The register (``src/config/facts_register.json``) is the machine copy of *BF-02 process fact register v1.1*. Every
coefficient the engine uses that is not part of the two-zone balance itself (burden-resistance rules, the packed-bed
exponent and base point, the enrichment-tag factor, the plausibility bands and the default limits) is read from here,
so a monthly refit or a test result changes a versioned file, not code.

Retired facts (status ``Retired``, e.g. COKE-17 the webapp hand coefficients and DATA-10 the regressed RAFT) can be
displayed but never used: :meth:`FactRegister.param` refuses them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

REGISTER_PATH = Path(__file__).resolve().parents[2] / "config" / "facts_register.json"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "schemas" / "facts_register.schema.json"

RETIRED = "Retired"


class RetiredFactError(KeyError):
    """A retired fact was asked for a value the engine would use."""


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

    def __init__(self, data: dict[str, Any]):
        self.name: str = data["name"]
        self.version: str = data["version"]
        self.date: str = data["date"]
        self.statuses: list[str] = list(data["statuses"])
        self.revision_log: list[dict[str, str]] = list(data.get("revision_log", []))
        self._facts = {f["id"]: Fact.from_dict(f) for f in data["facts"]}
        self._check()

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


def load_register_data(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or REGISTER_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=4)
def _cached(path: str) -> FactRegister:
    return FactRegister(load_register_data(path))


def load_register(path: str | Path | None = None) -> FactRegister:
    """The register at ``path`` (default: the shipped one), cached per path."""
    return _cached(str(Path(path or REGISTER_PATH).resolve()))


# Facts every engine result rests on; the page shows these with their status.
ENGINE_FACTS = ("ASM-01", "ASM-02", "ASM-03", "ASM-04", "ASM-05", "ASM-06", "ASM-08", "ASM-09",
                "FLOW-01", "FLOW-02", "FLOW-03", "FLOW-04", "FLOW-05", "FLOW-06", "DATA-02", "DATA-03",
                "BAND-01", "BAND-02", "BAND-03", "GAS-10")
