"""The plant fact register as FurnaceMind may show it (user decision 13 Nov 2026).

The register (``config/facts_register.json`` with the approved edits, ``utils.burden_engine.facts.load_register``)
holds the plant's facts: bands, responses, assumptions, data facts. FurnaceMind shows a fact's statement, value,
unit, range, status, grade, literature and the test that would settle it, never how a value was worked out (basis,
note, parameters, standard error) and never the model and validation groups; a fact whose shown text names a method
or a model (``skill_guard.method_words_in``) is left out. Settings in ``config/furnacemind_skills.yml`` (``facts``,
``skills.trial_board``).

* ``plant_facts()`` - the shown facts, in register order;
* ``open_facts()`` - the facts not settled on BF-02 (grade C / D or an open status) that name a test;
* ``trials()`` - the open facts grouped by the trial or question that would settle them, with the reason;
* ``lookup_plant_facts(...)`` - the ``lookup_plant_facts`` tool (JSON).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from agents.furnacemind.skill_guard import method_words_in

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "furnacemind_skills.yml"
_TEXT_FIELDS = ("statement", "unit", "literature", "next_test", "validity_window")


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    """``config/furnacemind_skills.yml`` (read once per process)."""
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _register() -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """The register's header, fact rows, trial list and trial rules: the approved edits over the file, else the file
    alone."""
    try:
        from utils.burden_engine.facts import load_register

        reg = load_register()
        header = {"name": reg.name, "version": reg.version, "date": reg.date, "label": reg.label}
        rows = [
            {
                "id": f.id,
                "group": f.group,
                "statement": f.statement,
                "value": f.value,
                "unit": f.unit,
                "range": list(f.range) if f.range else None,
                "status": f.status,
                "grade": f.grade,
                "literature": f.literature,
                "next_test": f.next_test,
                "validity_window": f.validity_window,
                "date": f.date,
            }
            for f in reg.facts()
        ]
        return header, rows, list(reg.trials), dict(reg.trial_rules)
    except Exception as exc:  # noqa: BLE001 - the register loader needs the engine package; the file is enough here
        log.warning("fact register loader unavailable (%s): reading the file", exc)
        from utils.burden_engine.facts import REGISTER_PATH

        data = json.loads(Path(REGISTER_PATH).read_text(encoding="utf-8"))
        header = {"name": data.get("name"), "version": data.get("version"), "date": data.get("date")}
        return header, list(data.get("facts", [])), list(data.get("trials") or []), dict(data.get("trial_rules") or {})


def _register_rows() -> tuple[dict[str, str], list[dict[str, Any]]]:
    header, rows, _, _ = _register()
    return header, rows


def shown(fact: Mapping[str, Any], *, cfg: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The part of one register row FurnaceMind may show, or None when the row is hidden: a hidden group, a retired
    fact, or a statement that names a method or a model. Other text fields that name one are left empty."""
    rules = (cfg or config()).get("facts", {})
    if fact.get("group") in set(rules.get("hidden_groups", [])) or fact.get("status") == "Retired":
        return None
    if method_words_in(str(fact.get("statement") or "")):
        return None
    out: dict[str, Any] = {}
    labels = rules.get("status_labels") or {}
    for key in rules.get("fields", []):
        value = fact.get(key)
        if key == "status" and value in labels:
            value = labels[value]
        if key in _TEXT_FIELDS and isinstance(value, str) and method_words_in(value):
            value = ""
        if value in (None, "", []):
            continue
        out[key] = value
    return out


def plant_facts(rows: Iterable[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every fact FurnaceMind may show, in register order (``rows`` for tests; default the register)."""
    if rows is None:
        _, rows = _register_rows()
    return [s for s in (shown(r) for r in rows) if s]


def register_header() -> dict[str, Any]:
    header, _ = _register_rows()
    return {k: v for k, v in header.items() if v}


def is_open(fact: Mapping[str, Any], *, cfg: Mapping[str, Any] | None = None) -> bool:
    """Not settled on BF-02: its grade is a published value or a judgement, or its status is open; and it names the
    test that would settle it."""
    rules = (cfg or config())["skills"]["trial_board"]
    if not fact.get("next_test"):
        return False
    return fact.get("grade") in set(rules["open_grades"]) or fact.get("status") in set(rules["open_statuses"])


def open_facts(facts: Iterable[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    return [dict(f) for f in (plant_facts() if facts is None else facts) if is_open(f)]


def trial_codes(next_test: str) -> list[str]:
    """The trial and question codes of a ``next_test`` (``"Q6, T5"`` -> ``["Q6", "T5"]``); a described test without a
    code is kept as its text (``"Nut-coke step trial (+15 kg/tHM for 7 days at constant blend)"``)."""
    from utils.burden_engine.facts import next_test_codes

    return next_test_codes(next_test)


def reason(fact: Mapping[str, Any], *, cfg: Mapping[str, Any] | None = None) -> str:
    """Why a test of the fact is suggested, in plain words from its grade and status."""
    rules = (cfg or config())["facts"]
    parts = []
    status = rules.get("statuses", {}).get(fact.get("status"))
    grade = rules.get("grades", {}).get(fact.get("grade"))
    if status:
        parts.append(f"status {fact.get('status')}: {status}")
    if grade:
        parts.append(f"grade {fact.get('grade')}: {grade}")
    return "; ".join(parts)


def _clean(text: Any) -> str:
    """``text`` when it names no method or model, else empty."""
    value = str(text or "").strip()
    return "" if not value or method_words_in(value) else value


def _benefit(trial: Mapping[str, Any]) -> str:
    """A trial's benefit in plant words: the plain wording when the register gives one, else the range in Rs/tHM with
    what it is for, else the list's own words when they name no method or model."""
    plant = trial.get("plant") or {}
    if plant.get("benefit"):
        return str(plant["benefit"])
    benefit = trial.get("benefit") or {}
    rng = benefit.get("rs_thm")
    if rng and len(rng) == 2:
        lo, hi = sorted(abs(float(v)) for v in rng)
        span = f"{lo:,.0f}" if lo == hi else f"{lo:,.0f}–{hi:,.0f}"
        if max(rng) <= 0:
            head = f"Saves Rs {span}/tHM"
        elif min(rng) >= 0:
            head = f"Costs Rs {span}/tHM"
        else:
            head = f"Rs {rng[0]:,.0f} to {rng[1]:,.0f}/tHM"
        basis = _clean(benefit.get("basis"))
        return f"{head}, {basis}" if basis else head
    return _clean(benefit.get("text"))


def shown_trial(trial: Mapping[str, Any], facts: Mapping[str, Mapping[str, Any]], named: list[str]) -> dict | None:
    """The part of a register trial FurnaceMind may show: code, rank, title, change, design, benefit, notes and status
    in plant words (the register's ``plant`` wording where the list's words name a model or method), and the shown
    facts it would settle (the list's own, then those naming it as their next test). None when its title or change
    names a method or model."""
    title, change = _clean(trial.get("title")), _clean(trial.get("change"))
    if not title or not change:
        return None
    plant = trial.get("plant") or {}
    ids = list(dict.fromkeys([*trial.get("facts", []), *named]))
    settles = [
        {**facts[fid], **({"why": reason(facts[fid])} if is_open(facts[fid]) else {})} for fid in ids if fid in facts
    ]
    out = {
        "id": trial["id"],
        "rank": trial.get("rank"),
        "title": title,
        "change": change,
        "design": str(plant.get("design") or _clean(trial.get("design"))),
        "benefit": _benefit(trial),
        "notes": str(plant.get("notes") or _clean(trial.get("notes"))),
        "status": trial.get("status", "Proposed"),
        "benefit_rs_thm": (trial.get("benefit") or {}).get("rs_thm"),
        "facts": settles,
    }
    return {k: v for k, v in out.items() if v not in ("", None)}


def trial_rules() -> dict[str, str]:
    """The register's rules for every trial, each only when it names no method or model."""
    _, _, _, rules = _register()
    return {k: v for k, v in ((k, _clean(v)) for k, v in rules.items()) if v}


def trials(facts: Iterable[Mapping[str, Any]] | None = None) -> OrderedDict[str, dict[str, Any]]:
    """The register's trial list in rank order (``listed`` True, with title, change, design, benefit and the facts
    each settles), then the other tests the open facts name (questions Q1-Q12, trials not on the list, tests
    described in words; ``listed`` False, most facts first). Every fact carries ``why`` when it is open."""
    from utils.burden_engine.facts import trial_links

    shown_facts = plant_facts() if facts is None else [dict(f) for f in facts]
    by_id = {f["id"]: f for f in shown_facts}
    _, rows, listed, _ = _register()
    named, _ = trial_links({"facts": rows, "trials": listed})
    out: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for trial in sorted(listed, key=lambda t: t.get("rank", 999)):
        view = shown_trial(trial, by_id, named.get(trial["id"], []))
        if view:
            out[trial["id"]] = {**view, "listed": True}
    groups: dict[str, list[dict[str, Any]]] = {}
    for f in open_facts(shown_facts):
        for code in trial_codes(str(f.get("next_test") or "")):
            if code not in out:
                groups.setdefault(code, []).append({**f, "why": reason(f)})
    for code, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out[code] = {"id": code, "title": "", "listed": False, "facts": items}
    return out


def _matches(fact: Mapping[str, Any], words: list[str]) -> bool:
    text = " ".join(str(fact.get(k) or "") for k in ("id", "group", "statement", "unit", "literature", "next_test"))
    low = text.lower()
    return all(w in low for w in words)


def lookup_plant_facts(
    *,
    query: str | None = None,
    ids: list[str] | None = None,
    open_only: bool = False,
    trial: str | None = None,
    limit: int = 40,
) -> str:
    """The ``lookup_plant_facts`` tool: shown facts filtered by ``ids``, words of ``query`` (all must appear),
    ``open_only`` and ``trial`` (a test code such as ``T5``), as JSON with the register's label and the meaning of
    grades and statuses; an open fact carries ``why``."""
    facts = plant_facts()
    if ids:
        wanted = {i.strip().upper() for i in ids if i and i.strip()}
        facts = [f for f in facts if str(f.get("id", "")).upper() in wanted]
    if query and query.strip():
        words = [w for w in query.lower().split() if w]
        facts = [f for f in facts if _matches(f, words)]
    if open_only or trial:
        facts = [f for f in facts if is_open(f)]
    board = trials()
    listed = [
        {**{k: v for k, v in t.items() if k != "facts"}, "facts": [f["id"] for f in t["facts"]]}
        for t in board.values()
        if t.get("listed")
    ]
    if trial and trial.strip():
        code = trial.strip().upper()
        entry = next((t for c, t in board.items() if c.upper() == code), None)
        settles = {f["id"] for f in entry["facts"]} if entry else set()
        facts = [
            f
            for f in facts
            if f["id"] in settles or code in [c.upper() for c in trial_codes(str(f.get("next_test") or ""))]
        ]
        listed = [t for t in listed if t["id"].upper() == code]
    elif query and query.strip():
        words = [w for w in query.lower().split() if w]
        listed = [t for t in listed if all(w in f"{t['id']} {t['title']} {t['change']}".lower() for w in words)]
    rules = config()["facts"]
    total = len(facts)
    out = [{**f, **({"why": reason(f)} if is_open(f) else {})} for f in facts[: max(1, int(limit))]]
    return json.dumps(
        {
            "register": register_header(),
            "facts": out,
            "trials": listed,
            "trial_rules": trial_rules(),
            "shown": len(out),
            "matching": total,
            "grades": rules.get("grades", {}),
            "statuses": rules.get("statuses", {}),
            "note": "Plant fact register: statements, values, the trial list and the test that would settle each "
            "fact. How a value was worked out is not part of this view.",
        },
        ensure_ascii=False,
        default=str,
    )
