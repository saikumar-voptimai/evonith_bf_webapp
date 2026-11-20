"""Demo mode (``BF_DEMO=1``; user decision 16 Nov 2026, demo branch): the names of the existing plant, its site, its
suppliers and its mines are replaced wherever the app shows them, by the rules of ``config/demo.yml``.

Pure functions, no web framework: the twin-api applies them to every response (``api.demo.DemoRedaction``), the
FurnaceMind service to the skill questions it stores and ``agents.furnace_tools`` to the tool results the model reads,
so a name reaches neither the screen nor the model.

* ``enabled()`` - demo mode on (``BF_DEMO`` 1 / true / yes / on).
* ``redact_text(text)`` - one string through every rule.
* ``redact(value, ids=False)`` - strings, dict keys and values, lists and tuples, recursively; other values
  unchanged; ``ids`` replaces the names inside ids as well (``id_paths`` views, what the model reads).
* ``ids_on(path)`` - the response of ``path`` gets its ids replaced too (``id_paths`` of ``config/demo.yml``).
* ``redact_json_bytes`` / ``redact_sse_event`` / ``redact_text_bytes`` - a response body of each kind.
* ``person(name)`` - a stored username as shown in the demo.
* ``labels()`` - the product, furnace and site the SPA names (``GET /api/version``), the demo's in demo mode.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "demo.yml"
_TRUE = {"1", "true", "yes", "on"}
#: The product, furnace and site the SPA names outside demo mode (served by ``GET /api/version``, so the SPA build
#: holds no plant name).
PRODUCT = "V-OptimAIse"
PLANT = {"product": PRODUCT, "furnace": "BF-02", "site": "Evonith Steel"}


def enabled() -> bool:
    """Demo mode is on (``BF_DEMO``)."""
    return os.environ.get("BF_DEMO", "").strip().lower() in _TRUE


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    """``config/demo.yml`` (read once per process)."""
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern[str]
    to: str


def _with_case(to: str) -> Callable[[re.Match[str]], str]:
    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        letters = [c for c in found if c.isalpha()]
        if len(letters) > 1 and all(c.isupper() for c in letters):
            return to.upper()
        if letters and all(c.islower() for c in letters):
            return to.lower()
        return to

    return replace


def _pattern(find: str, regex: bool, bounded: bool = True) -> re.Pattern[str]:
    body = find if regex else re.escape(find)
    # Word boundaries where the phrase begins or ends with a letter or digit: `Lloyds` matches "LLOYDS CLO" and not
    # the id `lloyds_clo` (the underscore is a word character, so there is no boundary after "lloyds"). A regex rule
    # is bounded on both sides unless it says `bounded: false` (a path that begins with "/").
    head = r"\b" if (regex and bounded) or (not regex and find[:1].isalnum()) else ""
    tail = r"\b" if (regex and bounded) or (not regex and find[-1:].isalnum()) else ""
    return re.compile(f"{head}(?:{body}){tail}", re.IGNORECASE)


@lru_cache(maxsize=1)
def rules() -> tuple[Rule, ...]:
    return tuple(
        Rule(_pattern(str(r["find"]), bool(r.get("regex")), bool(r.get("bounded", True))), str(r["to"]))
        for r in config().get("rules") or []
    )


def redact_text(text: str) -> str:
    """``text`` with every rule applied, in order."""
    if not text:
        return text
    for rule in rules():
        text = rule.pattern.sub(_with_case(rule.to), text)
    return text


_ID = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")


def redact_ids(text: str) -> str:
    """Ids made of words joined by underscores with each word through the rules (``lloyds_clo`` -> ``source_b_clo``).
    Only for what the model reads: the app's own responses keep their ids, which it sends back on a save."""

    def one(match: re.Match[str]) -> str:
        words = match.group(0).split("_")
        out = redact_text(" ".join(words))
        return out.replace(" ", "_") if out != " ".join(words) else match.group(0)

    return _ID.sub(one, text) if text else text


def redact_for_model(text: str) -> str:
    """What FurnaceMind's model reads in demo mode: names and the ids built from them replaced."""
    return redact_text(redact_ids(text))


def redact(value: Any, ids: bool = False) -> Any:
    """``value`` with every string redacted: dict keys and values, list and tuple items, recursively. ``ids`` replaces
    the names inside ids too (``lloyds_clo`` -> ``source_b_clo``)."""
    one = redact_for_model if ids else redact_text
    if isinstance(value, str):
        return one(value)
    if isinstance(value, dict):
        return {(one(k) if isinstance(k, str) else k): redact(v, ids) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, ids) for v in value]
    if isinstance(value, tuple):
        return tuple(redact(v, ids) for v in value)
    return value


def ids_on(path: str) -> bool:
    """The response of ``path`` is a read-only view that shows ids as text: its ids are replaced as well."""
    return any(path == p or path.startswith(f"{p}/") for p in config().get("id_paths") or [])


def labels() -> dict[str, str]:
    """Product, furnace and site labels: the demo's (``config/demo.yml``) in demo mode, else ``PLANT``."""
    if not enabled():
        return dict(PLANT)
    c = config()
    return {key: str(c.get(key) or PLANT[key]) for key in PLANT}


def person(name: str | None) -> str | None:
    """A stored username as the demo shows it: the system's own names as they are, anybody else as ``person``."""
    if not name:
        return name
    keep = {str(n).lower() for n in config().get("system_users") or []}
    return name if name.lower() in keep else str(config().get("person") or "engineer")


def redact_json_bytes(body: bytes, ids: bool = False) -> bytes:
    """A JSON body with its strings redacted (re-encoded compactly, UTF-8); a body that is not JSON as text."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return redact_text_bytes(body)
    return json.dumps(redact(data, ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def redact_text_bytes(body: bytes) -> bytes:
    try:
        return redact_text(body.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:
        return body


def redact_sse_event(event: bytes) -> bytes:
    """One server-sent event (its lines, without the blank line that ends it): each ``data:`` line redacted as JSON
    when it is JSON, else as text; other lines unchanged."""
    out = []
    for line in event.split(b"\n"):
        if line.startswith(b"data:"):
            payload = line[5:].lstrip(b" ")
            out.append(b"data: " + redact_json_bytes(payload) if payload else line)
        else:
            out.append(line)
    return b"\n".join(out)
