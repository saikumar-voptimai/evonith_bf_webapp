"""SystemPromptContext — assembles the LLM system prompt from static files + memory.

Construct once per page render (``__init__`` does file I/O).
``build()`` is a cheap pure-string join — call it as often as needed.

Static context loaded (in order, concatenated):
  1. CLAUDE.md          — blast furnace domain knowledge
  2. TOOLS*.md          — tool routing rules
  3. SKILLS*.md         — skill benchmark data (best-shift bands, coefficients)

All files live in ``src/storage/copilot/``.
"""

from __future__ import annotations

from pathlib import Path

from memory.copilot_memory import build_persistent_context, load_copilot_memory
from agents.cooperate.prompts import AI_COOPERATE_SYSTEM
from utils.logger import get_logger

logger = get_logger(__name__)

_COPILOT_DATA_DIR = Path(__file__).resolve().parents[2] / "storage" / "copilot"
_REPO_ROOT        = Path(__file__).resolve().parents[4]   # …/evonith_webapp


def _read_file(path: Path, *, max_chars: int) -> str:
    """Read a text file up to *max_chars*, truncating with a marker if needed."""
    try:
        if not path.exists():
            return ""
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return ""
        return txt if len(txt) <= max_chars else txt[:max_chars].rstrip() + "\n\n[...truncated...]"
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
        self._static    = self._load_static()
        self._errors    = self._load_errors()
        self.memory     = load_copilot_memory()
        self._persistent = build_persistent_context(self.memory)

    # ── Public ──────────────────────────────────────────────────────────────

    def build(self, extra: str = "") -> str:
        """Assemble the full system prompt string."""
        parts = [AI_COOPERATE_SYSTEM]
        if self._static:
            parts.append("STATIC CONTEXT (read this before answering):\n" + self._static)
        if self._persistent:
            parts.append(self._persistent)
        if self._errors:
            parts.append("RECENT TOOL ERRORS (avoid repeating these failure modes):\n" + self._errors)
        if extra:
            parts.append(extra.strip())
        return "\n\n".join(parts).strip()

    def refresh_memory(self) -> None:
        """Reload memory from disk (call after saving a new conversation turn)."""
        self.memory      = load_copilot_memory()
        self._persistent = build_persistent_context(self.memory)

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

        for p in sorted(_COPILOT_DATA_DIR.glob("SKILLS*.md"), key=lambda f: f.name):
            txt = _read_file(p, max_chars=14_000)
            if txt:
                logger.info("Loaded %s (%d chars)", p.name, len(txt))
                parts.append(f"{p.name} (skill benchmark data):\n" + txt)

        return "\n\n---\n\n".join(parts).strip()

    def _load_errors(self) -> str:
        try:
            path = Path(__file__).resolve().parents[2] / "agents" / "tool_errors.md"
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8")[-2_500:].strip()
        except Exception:
            return ""
