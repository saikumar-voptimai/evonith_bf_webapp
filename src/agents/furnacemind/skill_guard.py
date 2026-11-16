"""What a FurnaceMind quick skill may show and use (user decision 13 Nov 2026: the skills must not reveal how the
V-OptimAIse models work; they analyse the plant's own data and the plant fact register).

* ``METHOD_WORDS`` - the method words a client page never shows, the Python twin of the SPA's
  ``frontend/src/lib/copy/method-words.ts`` (kept equal by ``tests/test_furnacemind_skill_guard.py``), plus the
  model names (``MODEL_NAMES``). ``method_words_in`` finds them in a text.
* ``SKILL_TOOLS`` / ``skill_tools`` - a skill turn offers the model the plant-data tools only: no live-twin tool, no
  optimiser result.
* ``skill_system_prompt`` - the system prompt of a skill turn: the persona without its live-twin routing, the skill's
  context and ``SKILL_TOOL_POLICY``; no tool-routing notes (they name the twin's models).

Used by the agent service (``services.furnacemind``) and page 7 (``agents.furnacemind.ai_cooperate_page``) alike.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MethodWord:
    """A word a client page never shows; ``pattern`` is a regular expression, case-insensitive unless
    ``case_sensitive``."""

    word: str
    pattern: str
    case_sensitive: bool = False

    def search(self, text: str) -> bool:
        return re.search(self.pattern, text, 0 if self.case_sensitive else re.IGNORECASE) is not None


def _w(word: str, pattern: str | None = None, case_sensitive: bool = False) -> MethodWord:
    return MethodWord(word, pattern if pattern is not None else r"\b" + re.escape(word), case_sensitive)


#: The SPA's ``METHOD_WORDS`` in the same order (``method-words.ts``).
METHOD_WORDS: tuple[MethodWord, ...] = (
    _w("calibration", r"\bcalibrat"),
    _w("Jacobian"),
    _w("Kalman"),
    _w("surrogate"),
    _w("CUSUM", r"\bCUSUM\b", True),
    _w("anchor"),
    _w("posterior"),
    _w("AR(1)", r"AR\(1\)", True),
    _w("SLP", r"\bSLP\b", True),
    _w("HiGHS", r"\bHiGHS\b", True),
    _w("differential evolution", r"\bdifferential[ -]evolution"),
    _w("network solve", r"\bnetwork[ -]solve"),
    _w("sigma", r"\bsigma\b"),
    _w("σ", "σ", True),
    _w("STRATA-Flow", "STRATA-Flow", True),
    _w("two-zone", r"\btwo[ -]zone"),
    _w("Rist", r"\bRist\b", True),
    _w("omega", r"\bomega\b"),
    _w("ω", "ω", True),
    _w("shadow price", r"\bshadow[ -]price"),
    _w("z-score", r"\bz[ -]score"),
    _w("Diebold–Mariano", r"\bDiebold"),
    _w("Thompson sampling", r"\bThompson"),
    _w("Bland–Altman", r"\bBland[ –-]Altman"),
    _w("regression", r"\bregress"),
    _w("linearisation", r"\blineari[sz]"),
    _w("random walk", r"\brandom[ -]walk"),
)

#: The names of the V-OptimAIse models (upper case as the product writes them; "trace" in prose is not one).
MODEL_NAMES: tuple[MethodWord, ...] = (
    _w("STRATA", r"\bSTRATA\b", True),
    _w("FORGE", r"\bFORGE\b", True),
    _w("TRACE", r"\bTRACE\b", True),
)


def method_words_in(text: str) -> list[str]:
    """The method words and model names ``text`` contains, in list order (empty when it shows none)."""
    if not text:
        return []
    return [m.word for m in (*METHOD_WORDS, *MODEL_NAMES) if m.search(text)]


#: The tools a skill turn offers: plant data, its charts, the uploaded documents and the plant fact register.
SKILL_TOOLS: frozenset[str] = frozenset(
    {
        "fetch_online_data",
        "fetch_offline_data",
        "fetch_ml_data",
        "merge_furnace_data",
        "concat_datasets",
        "load_static_shift_data",
        "execute_python_plot",
        "render_heatload_plot",
        "show_skill_chart",
        "lookup_plant_facts",
        "search_knowledge_docs",
    }
)


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function") if isinstance(schema, Mapping) else None
    return str((function or {}).get("name") or schema.get("name") or "")


def skill_tools(tools: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The schemas of ``tools`` a skill turn may offer (``SKILL_TOOLS``), in their order."""
    return [dict(t) for t in tools if _tool_name(t) in SKILL_TOOLS]


SKILL_TOOL_POLICY = """\
SKILL TURN RULES (a quick skill is running):
- The skill has already computed its numbers from the plant's own records; they are in the question. Use them as
  given and do not recompute them. Call show_skill_chart first when the question names a chart.
- Use only the plant data, the uploaded documents and the plant fact register (lookup_plant_facts). Call a fetch tool
  only when the question says a number is missing and the note cannot be written without it.
- Never describe, name or guess how V-OptimAIse models, optimisers or digital-twin calculations work: no model names,
  internal parameters, method names or equations. If asked, say this skill works on plant data and the plant fact
  register only.
- Write deviations as "n times the usual spread", never with statistical jargon. Every number carries its unit; times
  are IST.
- Actions follow ACTION / REASON / MAGNITUDE and stay within what the data shows; say what to check when the data
  cannot decide.\
"""


def skill_persona() -> str:
    """FurnaceMind's persona without its live-twin routing line."""
    from agents.furnacemind.prompts import AI_COOPERATE_SYSTEM

    lines = AI_COOPERATE_SYSTEM.splitlines()
    return "\n".join(line for line in lines if "get_twin_snapshot" not in line and "twin" not in line.lower())


def skill_system_prompt(skill_context: str | None = None) -> str:
    """The system prompt of a quick-skill turn: persona, the skill's context, the skill rules."""
    parts = [skill_persona()]
    if skill_context and skill_context.strip():
        parts.append("SKILL CONTEXT (the selected quick skill):\n" + skill_context.strip())
    parts.append(SKILL_TOOL_POLICY)
    return "\n\n".join(parts)
