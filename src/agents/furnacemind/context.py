"""SystemPromptContext — assembles the LLM system prompt from static files + memory.

Construct once per page render (``__init__`` does file I/O).
``build()`` is a cheap pure-string join — call it as often as needed.

Static context loaded (in order, concatenated):
  1. CLAUDE.md          — blast furnace domain knowledge
  2. TOOLS*.md          — tool routing rules
  3. SKILLS*.md         — injected only for the active skill (not all at once)

All files live in ``src/storage/furnacemind/``.
"""

from __future__ import annotations

from pathlib import Path

from agents.furnacemind.prompts import AI_COOPERATE_SYSTEM
from agents.memory.fm_memory import build_persistent_context, load_fm_memory
from utils.logger import get_logger

logger = get_logger(__name__)

_COPILOT_DATA_DIR = Path(__file__).resolve().parents[2] / "storage" / "furnacemind"
_REPO_ROOT = Path(__file__).resolve().parents[4]  # …/evonith_webapp

# Which SKILLS*.md files to inject per active skill. None = free chat (no skill docs).
_SKILL_FILES: dict[str | None, list[str]] = {
    "optimise":      ["SKILLS_BESTSHIFT.md", "SKILLS_OPTIMISE.md"],
    "shift_to_best": ["SKILLS_BESTSHIFT.md"],
    "heatload":      ["SKILLS_HEATLOAD.md"],
    "shift_report":  ["SKILLS_SHIFTREPORT.md"],
    None:            [],
}


def _read_file(path: Path, *, max_chars: int) -> str:
    """Read a text file up to *max_chars*, truncating with a marker if needed."""
    try:
        if not path.exists():
            return ""
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return ""
        return (
            txt
            if len(txt) <= max_chars
            else txt[:max_chars].rstrip() + "\n\n[...truncated...]"
        )
    except Exception:
        return ""


class SystemPromptContext:
    """Loads and caches all context needed to build the LLM system prompt.

    Usage::

        ctx = SystemPromptContext()
        system_prompt = ctx.build(extra=tool_policy)
        # later, to refresh memory after a conversation turn:
        ctx.refresh_memory()
    """

    def __init__(self) -> None:
        self._static = self._load_static()
        self._errors = self._load_errors()
        self.memory = load_fm_memory()
        self._persistent = build_persistent_context(self.memory)

    # ── Public ──────────────────────────────────────────────────────────────

    def build(self, extra: str = "", skill_id: str | None = None) -> str:
        """Assemble the full system prompt string.

        Args:
            extra:    Additional text appended at the end (e.g. TOOL_POLICY).
            skill_id: Active skill key — controls which SKILLS*.md is injected.
                      Pass None for free-chat turns (no skill docs loaded).
        """
        parts = [AI_COOPERATE_SYSTEM]
        if self._static:
            parts.append(
                "STATIC CONTEXT (read this before answering):\n" + self._static
            )
        skill_ctx = self._load_skills(skill_id)
        if skill_ctx:
            parts.append("SKILL CONTEXT (active skill reference data):\n" + skill_ctx)
        if self._persistent:
            parts.append(self._persistent)
        if self._errors:
            parts.append(
                "RECENT TOOL ERRORS (avoid repeating these failure modes):\n"
                + self._errors
            )
        if extra:
            parts.append(extra.strip())
        return "\n\n".join(parts).strip()

    def refresh_memory(self) -> None:
        """Reload memory from disk (call after saving a new conversation turn)."""
        self.memory = load_fm_memory()
        self._persistent = build_persistent_context(self.memory)

    def refresh_session_context(self) -> None:
        """Refresh all per-session data (memory + recent tool errors).

        Call this once per render on the cached SystemPromptContext instance so
        that new conversation turns and new tool errors are picked up without
        re-loading the expensive static files (TOOLS*.md, CLAUDE.md).
        """
        self.refresh_memory()
        self._errors = self._load_errors()

    # ── Private loaders ─────────────────────────────────────────────────────

    def _load_static(self) -> str:
        parts: list[str] = []

        claude_md = _read_file(_REPO_ROOT / "CLAUDE.md", max_chars=24_000)
        if claude_md:
            logger.info("Loaded CLAUDE.md (%d chars)", len(claude_md))
            parts.append("CLAUDE.md (blast furnace domain context):\n" + claude_md)

        for p in sorted(_COPILOT_DATA_DIR.glob("TOOLS*.md"), key=lambda f: f.name):
            txt = _read_file(p, max_chars=12_000)
            if txt:
                logger.info("Loaded %s (%d chars)", p.name, len(txt))
                parts.append(f"{p.name} (available tools + calling rules):\n" + txt)

        return "\n\n---\n\n".join(parts).strip()

    def _load_skills(self, skill_id: str | None) -> str:
        """Load only the SKILLS*.md files relevant to *skill_id*."""
        filenames = _SKILL_FILES.get(skill_id, [])
        parts: list[str] = []
        for name in filenames:
            txt = _read_file(_COPILOT_DATA_DIR / name, max_chars=14_000)
            if txt:
                logger.info("Loaded skill file %s (%d chars) for skill=%s", name, len(txt), skill_id)
                parts.append(f"{name} (skill benchmark data):\n" + txt)
        return "\n\n---\n\n".join(parts)

    def _load_errors(self) -> str:
        try:
            path = Path(__file__).resolve().parents[2] / "agents" / "tool_errors.md"
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8")[-2_500:].strip()
        except Exception:
            return ""
