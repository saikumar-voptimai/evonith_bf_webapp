"""Reusable helpers for FurnaceMind skill sidebar and button display.

The Streamlit chat interface owns widget rendering and session state. This
module keeps the pure skill formatting, metadata, filename, and context-preview
helpers separate so UI code can stay focused on layout and interactions.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from furnace_data.assets import package_furnacemind_assets_dir
from furnace_data.runtime_paths import get_repo_root, runtime_path

_SKILL_SLUG_RE = re.compile(r"[^a-z0-9_]+")
_SKILL_FILE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_FURNACEMIND_SOURCE_DIR = package_furnacemind_assets_dir()
_LEGACY_SKILL_STORAGE_DIR = get_repo_root() / "src" / "storage" / "furnacemind"


def _runtime_skill_storage_dir() -> Path:
    """Return the runtime directory for uploaded skill markdown."""
    return runtime_path("uploads", "furnacemind", "skills")


def _skill_context_path(filename: Any) -> Path:
    """Return runtime skill context path, falling back to source storage."""
    safe_name = Path(str(filename or "")).name
    runtime_candidate = _runtime_skill_storage_dir() / safe_name
    if runtime_candidate.exists():
        return runtime_candidate
    package_candidate = _FURNACEMIND_SOURCE_DIR / safe_name
    if package_candidate.exists():
        return package_candidate
    return _LEGACY_SKILL_STORAGE_DIR / safe_name


def skill_metadata(skill: Any) -> dict[str, Any]:
    """Return JSON metadata from a skill row as a plain dictionary.

    SQLAlchemy exposes the ``metadata`` column as ``metadata_json`` in our model.
    Sidebar and registry-adjacent UI helpers read optional fields from this JSON
    block, so malformed or missing metadata is normalized to an empty dictionary
    in one place.
    """
    metadata = getattr(skill, "metadata_json", None)
    return metadata if isinstance(metadata, dict) else {}


def normalize_skill_slug(value: str) -> str:
    """Return a safe runtime slug for skill keys and metadata.

    Skill names can contain spaces, symbols, and mixed case. Streamlit keys,
    registry matching, and Qdrant payloads are easier to reason about when the
    value is normalized to lowercase letters, numbers, and underscores.
    """
    slug = _SKILL_SLUG_RE.sub("_", str(value or "").strip().lower()).strip("_")
    return slug or "skill"


def skill_slug(skill: Any) -> str:
    """Return the runtime slug stored in metadata or derived from the row id.

    The slug is used by the registry and chat prompt context. Existing metadata
    wins so editing a skill name does not silently change its runtime identity.
    """
    metadata = skill_metadata(skill)
    return normalize_skill_slug(
        metadata.get("slug") or metadata.get("key") or getattr(skill, "skill_id", "")
    )


def unique_skill_slug(name: str, skills: list[Any]) -> str:
    """Return a slug that does not collide with existing DB skill rows.

    Newly-created custom skills derive their slug from the entered name. When a
    matching slug already exists, numeric suffixes are added so each row remains
    addressable by a stable key.
    """
    existing = {skill_slug(skill) for skill in skills}
    base = normalize_skill_slug(name)
    slug = base
    suffix = 2
    while slug in existing:
        slug = f"{base}_{suffix}"
        suffix += 1
    return slug


def skill_order(skill: Any, default: int = 100) -> int:
    """Return the internal sort order stored in skill metadata.

    The UI does not expose display-order editing, but the metadata value still
    keeps seeded and custom skills in a predictable order. Invalid values fall
    back to ``default`` instead of breaking the sidebar.
    """
    try:
        return int(skill_metadata(skill).get("order") or default)
    except (TypeError, ValueError):
        return default


def next_skill_order(skills: list[Any]) -> int:
    """Return the next internal sort slot for a newly-created custom skill.

    This is intentionally not a user-facing setting. It appends new skills after
    existing ones while leaving gaps so seed data or admin updates can later
    reorder rows without touching UI code.
    """
    if not skills:
        return 100
    return max(skill_order(skill) for skill in skills) + 10


def skill_symbol(skill: Any) -> str:
    """Return the short symbol displayed before a skill name.

    Newer databases store the symbol in the ``symbol`` column. Metadata fallback
    keeps older rows usable, and the result is capped so a long pasted value does
    not distort the sidebar or quick-skill buttons.
    """
    metadata = skill_metadata(skill)
    symbol = str(getattr(skill, "symbol", None) or metadata.get("symbol") or "").strip()
    return symbol[:8]


def skill_button_text(skill: Any) -> str:
    """Return selectbox/button text using the configured symbol and name."""
    name = str(getattr(skill, "name", "Skill") or "Skill")
    symbol = skill_symbol(skill)
    return f"{symbol} {name}".strip()


def skill_symbol_conflicts(
    symbol: str, skills: list[Any], *, exclude_skill_id: str | None = None
) -> bool:
    """Return whether the proposed symbol is already used by another skill.

    Symbols are meant to be quick visual identifiers, so create/edit validation
    keeps them unique. ``exclude_skill_id`` allows a skill to save its own
    current symbol without being flagged as a conflict.
    """
    normalized = str(symbol or "").strip().casefold()
    if not normalized:
        return False
    for skill in skills:
        if exclude_skill_id and str(getattr(skill, "skill_id", "")) == exclude_skill_id:
            continue
        if skill_symbol(skill).casefold() == normalized:
            return True
    return False


def safe_skill_file_name(slug: str, original_name: str) -> str:
    """Return a safe stored filename for an uploaded ``Skill.md`` file.

    Only the basename and a restricted character set are used. The skill slug is
    prefixed so files from different skills do not overwrite each other when the
    user uploads a generic filename such as ``SKILL.md``.
    """
    original = Path(str(original_name or "skill.md")).name
    stem = _SKILL_FILE_RE.sub("_", Path(original).stem).strip("._") or "skill"
    clean_slug = _SKILL_FILE_RE.sub("_", slug).strip("._") or "skill"
    return f"{clean_slug}_{stem}.md"


def store_skill_markdown(uploaded_file: Any, *, slug: str) -> str:
    """Persist the uploaded markdown context file for a custom skill.

    The database stores only the safe filename in metadata; the actual file lives
    under runtime uploads so the skill registry can later inject it as prompt
    context. Empty uploads are rejected before the row is created.
    """
    filename = safe_skill_file_name(slug, getattr(uploaded_file, "name", "skill.md"))
    path = _runtime_skill_storage_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    content = uploaded_file.getvalue()
    if not content:
        raise ValueError("Skill markdown file is empty.")
    path.write_bytes(content)
    return filename


def html_text(value: Any, default: str = "") -> str:
    """Escape dynamic skill text before rendering custom HTML blocks.

    The details card uses ``unsafe_allow_html`` for compact layout, so every
    database-backed value rendered inside that block must be escaped here first.
    """
    raw = str(value if value is not None else default).strip() or default
    return html.escape(raw, quote=True)


def display_source_type(value: Any) -> str:
    """Return a human-readable label for the skill source type."""
    source = str(value or "custom").strip().replace("_", " ")
    return source.title() if source else "Custom"


def skill_source_type(skill: Any) -> str:
    """Return the persisted source type, defaulting custom rows to ``custom``."""
    return str(getattr(skill, "source_type", "") or "custom").strip() or "custom"


def is_built_in_skill(skill: Any) -> bool:
    """Return True when a row represents a protected built-in skill.

    Built-ins are still database-configurable for symbol/active state, but their
    handler, instruction, and display text stay protected because those map to
    code-backed prompt methods.
    """
    return skill_source_type(skill).casefold() == "built_in"


def format_skill_timestamp(value: Any) -> str:
    """Return a compact timestamp string for the selected skill details panel."""
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
    return str(value).strip()


def read_skill_context_preview(filename: Any, *, max_chars: int = 1_800) -> str:
    """Return a bounded preview of one stored skill markdown file.

    The preview is only for the sidebar details panel. It reads from the same
    approved skill-storage directory as the registry and truncates long files so
    the sidebar remains responsive.
    """
    try:
        path = _skill_context_path(filename)
        if not path.exists() or not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n\n[...truncated...]"
    except Exception:
        return ""


def build_skill_context_preview(*, instruction: str, context_files: list[Any]) -> str:
    """Build the read-only preview of context passed to the agent for a skill.

    This mirrors the registry path closely enough for users to inspect what the
    skill will contribute: SQL instruction first, then each configured markdown
    support file. It is display-only and does not execute the skill.
    """
    parts: list[str] = []
    if instruction.strip():
        parts.append("SQL instruction:\n" + instruction.strip())
    for filename in context_files:
        preview = read_skill_context_preview(filename)
        if preview:
            parts.append(f"{Path(str(filename)).name}:\n{preview}")
    return "\n\n---\n\n".join(parts).strip()


def format_skill_option(skill: Any) -> str:
    """Return the skill selector label with symbol, name, and active state."""
    status = "active" if getattr(skill, "is_active", False) else "inactive"
    return f"{skill_button_text(skill)} ({status})"
