"""Build FurnaceMind system prompts from static context and saved memory.

The page creates one ``SystemPromptContext`` and refreshes it for the active
user/conversation before each model call. Static files provide domain and tool
instructions, while the PostgreSQL-backed memory summary adds durable context
from earlier turns in the same conversation.

Static context loaded by this module:
  1. ``CLAUDE.md`` for blast furnace domain knowledge.
  2. ``TOOLS*.md`` for tool routing and usage rules.
  3. ``SKILLS*.md`` for selected and semantically relevant quick skills.
"""

from __future__ import annotations

from pathlib import Path

from agents.furnacemind.prompts import AI_COOPERATE_SYSTEM
from agents.memory.fm_memory import build_persistent_context, load_fm_memory
from furnace_data.runtime_paths import runtime_path
from utils.logger import get_logger

logger = get_logger(__name__)

_COPILOT_DATA_DIR = Path(__file__).resolve().parents[2] / "storage" / "furnacemind"
_REPO_ROOT = Path(__file__).resolve().parents[4]  # …/evonith_webapp

# Which SKILLS*.md files to inject per active skill. None = free chat (no skill docs).
_SKILL_FILES: dict[str | None, list[str]] = {
    "optimise": ["SKILLS_BESTSHIFT.md", "SKILLS_OPTIMISE.md"],
    "shift_to_best": ["SKILLS_BESTSHIFT.md"],
    "heatload": ["SKILLS_HEATLOAD.md"],
    "shift_report": ["SKILLS_SHIFTREPORT.md"],
    None: [],
}


def _read_file(path: Path, *, max_chars: int) -> str:
    """
    Read a prompt-support file and return bounded text for LLM injection.

    Prompt files can be large, missing, or temporarily unreadable during local
    development. This helper keeps the prompt builder resilient by returning an
    empty string for missing/failed files and truncating oversized files with a
    visible marker before they are appended to the system prompt.

    Args:
         - path: Path - File path to read from disk.
         - max_chars: int - Maximum number of characters to keep.

    Returns:
         - return: str - Cleaned file text, truncated text, or an empty string.
    """
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
    """Load and refresh the context needed to build FurnaceMind prompts.

    The class separates static context from per-session context. Static domain
    and tool files are loaded once during initialization, while memory summaries
    and recent tool-error notes can be refreshed for the current user and
    conversation before each chat response.
    """

    def __init__(self) -> None:
        """
        Initialize static prompt context and the default memory snapshot.

        Static files are read once so Streamlit reruns do not repeatedly touch
        the filesystem. Memory starts empty because the page later calls
        ``refresh_session_context`` with the active user and conversation after
        the conversation id has been created or restored.

        Args:
             - None

        Returns:
             - return: None - This function does not return a value.
        """
        self._static = self._load_static()
        self._errors = self._load_errors()
        self.memory = load_fm_memory()
        self._persistent = build_persistent_context(self.memory)

    # ── Public ──────────────────────────────────────────────────────────────

    def build(
        self,
        extra: str = "",
        skill_id: str | None = None,
        skill_context: str | None = None,
    ) -> str:
        """
        Assemble the complete system prompt for the next FurnaceMind model call.

        Args:
             - extra: str - Additional instructions appended at the end.
             - skill_id: str | None - Active quick-skill key for fallback context.
             - skill_context: str | None - Selected and semantically relevant skill context for this turn.

        Returns:
             - return: str - System prompt containing persona, context, memory, and policy.
        """
        parts = [AI_COOPERATE_SYSTEM]
        if self._static:
            parts.append(
                "STATIC CONTEXT (read this before answering):\n" + self._static
            )
        skill_ctx = (
            skill_context.strip()
            if skill_context is not None
            else self._load_skills(skill_id)
        )
        if skill_ctx:
            parts.append(
                "SKILL CONTEXT (selected and semantically relevant skill data):\n"
                + skill_ctx
            )
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

    def refresh_memory(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """
        Reload the persistent conversation summary for the active chat.

        FurnaceMind stores compressed memory summaries per user and
        conversation. This method refreshes that saved summary and rebuilds the
        prompt-ready persistent context block so the next LLM call can include
        durable facts from earlier message windows.

        Args:
             - user_id: str | None - User whose memory should be loaded.
             - conversation_id: str | None - Conversation whose summary should be loaded.

        Returns:
             - return: None - This function does not return a value.
        """
        self.memory = load_fm_memory(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._persistent = build_persistent_context(self.memory)

    def refresh_session_context(
        self,
        *,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """
        Refresh per-session prompt inputs before running the agent.

        The page calls this after the conversation id is known and before the
        model prompt is built. It reloads the PostgreSQL memory summary for the
        active chat and also refreshes recent tool-error notes so the agent can
        avoid repeating known tool failures.

        Args:
             - user_id: str | None - User whose memory should be loaded.
             - conversation_id: str | None - Conversation whose summary should be loaded.

        Returns:
             - return: None - This function does not return a value.
        """
        self.refresh_memory(user_id=user_id, conversation_id=conversation_id)
        self._errors = self._load_errors()

    # ── Private loaders ─────────────────────────────────────────────────────

    def _load_static(self) -> str:
        """
        Load static FurnaceMind domain and tool context from repository files.

        Static context is the shared instruction base for all FurnaceMind chat
        turns. This includes the repository-level furnace knowledge from
        ``CLAUDE.md`` and any tool-routing files stored under the FurnaceMind
        storage directory.

        Args:
             - None

        Returns:
             - return: str - Joined static context sections for prompt injection.
        """
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
        """
        Load skill-specific prompt context for the selected quick skill.

        FurnaceMind should not inject every skill file on every request because
        that would increase prompt size and mix unrelated instructions. This
        method maps the active skill id to the small set of skill files needed
        for that turn and returns only those sections.

        Args:
             - skill_id: str | None - Active quick-skill key from the UI.

        Returns:
             - return: str - Joined skill context sections, or an empty string.
        """
        filenames = _SKILL_FILES.get(skill_id, [])
        parts: list[str] = []
        for name in filenames:
            txt = _read_file(_COPILOT_DATA_DIR / name, max_chars=14_000)
            if txt:
                logger.info(
                    "Loaded skill file %s (%d chars) for skill=%s",
                    name,
                    len(txt),
                    skill_id,
                )
                parts.append(f"{name} (skill benchmark data):\n" + txt)
        return "\n\n---\n\n".join(parts)

    def _load_errors(self) -> str:
        """
        Load recent tool-error notes for prompt-level failure avoidance.

        FurnaceMind records recent tool issues in ``tool_errors.md``. Including
        only the tail of that file gives the agent enough context to avoid
        repeating recent bad calls without allowing old errors to dominate the
        system prompt.

        Args:
             - None

        Returns:
             - return: str - Recent tool-error text, or an empty string.
        """
        try:
            path = runtime_path("logs", "tool_errors.md")
            if not path.exists():
                path = (
                    Path(__file__).resolve().parents[2]
                    / "agents"
                    / "tool_errors.md"
                )
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8")[-2_500:].strip()
        except Exception:
            return ""
