"""Qdrant-backed semantic index for FurnaceMind quick skills.

SQL remains the source of truth for skills: active state, display text,
instructions, and metadata all live in ``furnace_mind.skills``. This module only
stores retrieval vectors so FurnaceMind can find additional relevant skills for a
chat turn without embedding every skill on every request.

Each saved skill is indexed as one compact ``skill_profile`` point plus optional
``skill_context`` points for configured markdown files. Runtime retrieval uses
Qdrant to identify candidate skill ids, then the registry reloads full skill
context from SQL/files before injecting it into the prompt.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from utils.settings import settings

_SKILL_STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "furnacemind"
_POINT_NAMESPACE = uuid.UUID("2c2cb7b3-6eca-4ef7-8c63-06d0da7117aa")
_INDEXED_PAYLOAD_FIELDS = ("source", "skill_id", "slug", "chunk_type")
_PROFILE_CONTEXT_CHARS = 4000
_CONTEXT_CHUNK_CHARS = 2800
_CONTEXT_CHUNK_OVERLAP = 250


@dataclass(frozen=True)
class SkillVectorMatch:
    """Best Qdrant match found for a single FurnaceMind skill.

    Search can return multiple points for the same skill because a skill is
    indexed as a profile point plus optional markdown context chunks. The store
    collapses those points to one ``SkillVectorMatch`` per skill id and keeps the
    highest-scoring payload for downstream registry resolution.
    """

    skill_id: str
    slug: str
    score: float | None
    payload: dict[str, Any]


def _embed_text(embedding_client: Any, text: str, *, input_type: str) -> list[float]:
    """Return a float vector for query or document text.

    Newer embedding clients support an ``input_type`` argument so document
    embeddings and query embeddings can use the provider's intended retrieval
    mode. Older clients only accept the text, so this helper keeps the vector
    store compatible with both interfaces.
    """
    try:
        vector = embedding_client.embed_text(text, input_type=input_type)
    except TypeError:
        vector = embedding_client.embed_text(text)
    return [float(value) for value in vector]


def _metadata_for(skill: Any) -> dict[str, Any]:
    """Extract metadata from either an ORM row or a registry definition.

    The SQL repository exposes JSON as ``metadata_json`` while some in-memory
    skill definitions use ``metadata``. The vector index treats both shapes the
    same so indexing works for persisted skills and fallback definitions.
    """
    metadata = getattr(skill, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    metadata_json = getattr(skill, "metadata_json", None)
    return metadata_json if isinstance(metadata_json, dict) else {}


def _string_value(value: Any, default: str = "") -> str:
    """Normalize optional database values into safe non-null strings.

    Skill fields can come from SQL rows, metadata dictionaries, or small test
    doubles. This helper trims whitespace and gives callers a predictable string
    without leaking ``None`` into Qdrant payloads or embedded text.
    """
    if value is None:
        return default
    stripped = str(value).strip()
    return stripped if stripped else default


def _skill_slug(skill: Any) -> str:
    """Resolve the stable identifier used to display and filter a skill.

    Slug is preferred from metadata because uploaded skills store their durable
    UI key there. If it is missing, the helper falls back through common fields
    so older rows and tests can still be indexed without failing.
    """
    metadata = _metadata_for(skill)
    slug = _string_value(
        metadata.get("slug")
        or metadata.get("key")
        or metadata.get("skill_slug")
        or getattr(skill, "slug", None)
        or getattr(skill, "skill_id", None)
        or getattr(skill, "name", None)
    )
    return slug or "skill"


def _context_files(skill: Any) -> tuple[str, ...]:
    """Return de-duplicated markdown context filenames for a skill.

    Only the basename is kept and only ``.md`` files are accepted. That prevents
    metadata from pointing outside the approved skill storage folder while still
    allowing uploaded skills to attach one or more markdown prompt/context files.
    """
    metadata = _metadata_for(skill)
    value = metadata.get("context_files") or metadata.get("skill_files")
    if isinstance(value, str):
        raw_files = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_files = list(value)
    else:
        raw_files = []

    filenames: list[str] = []
    for item in raw_files:
        filename = Path(str(item or "").strip()).name
        if filename and filename.lower().endswith(".md"):
            filenames.append(filename)
    return tuple(dict.fromkeys(filenames))


def _read_context_file(filename: str) -> str:
    """Read one stored skill markdown file, returning empty text on failure.

    Skill context is optional. Missing files, deleted files, or encoding issues
    should not break page load or Qdrant indexing, so the vector store simply
    skips unreadable context files.
    """
    try:
        path = _SKILL_STORAGE_DIR / Path(filename).name
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _bounded_text(text: str, *, max_chars: int) -> str:
    """Trim embedded or payload text to a deterministic maximum size.

    Embedding providers and Qdrant payloads should not receive unbounded skill
    instructions or markdown content. The truncation marker makes inspection in
    Qdrant clear without affecting the original SQL row or markdown file.
    """
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "\n\n[...truncated...]"


def _chunk_text(text: str) -> list[str]:
    """Split long markdown skill context into searchable overlapping chunks.

    Each chunk is embedded separately so a long SOP or prompt guide can match a
    specific user question. Overlap keeps boundary sentences available in both
    neighboring chunks, which improves retrieval for text near chunk edges.
    """
    clean = str(text or "").strip()
    if not clean:
        return []
    if len(clean) <= _CONTEXT_CHUNK_CHARS:
        return [clean]

    chunks: list[str] = []
    step = max(_CONTEXT_CHUNK_CHARS - _CONTEXT_CHUNK_OVERLAP, 1)
    for start in range(0, len(clean), step):
        chunk = clean[start : start + _CONTEXT_CHUNK_CHARS].strip()
        if chunk:
            chunks.append(chunk)
        if start + _CONTEXT_CHUNK_CHARS >= len(clean):
            break
    return chunks


def _point_id(skill_id: str, chunk_type: str, chunk_index: int = 0) -> str:
    """Build a stable Qdrant point id for one skill profile or context chunk.

    The uuid5 namespace keeps ids deterministic across app restarts. Re-indexing
    the same skill therefore replaces the same logical points instead of adding
    random duplicates.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{skill_id}:{chunk_type}:{chunk_index}"))


class SkillVectorStore:
    """Persist and search FurnaceMind skill retrieval vectors in Qdrant.

    This store is deliberately not the source of truth for skills. It keeps only
    the semantic search representation of SQL skills so the registry can augment
    explicitly selected skills with contextually relevant ones. The registry
    reloads the matching skill rows from SQL and verifies active status before
    anything is injected into the system prompt.
    """

    def __init__(self, embedding_client: Any) -> None:
        """Connect to Qdrant and validate collection settings.

        The embedding dimension in ``SKILL_QDRANT_EMBED_DIM`` must match the
        active embedding client. Failing fast here avoids writing incompatible
        vectors that would later break retrieval.
        """
        qcfg = settings.qdrant_skills
        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )
        self.embedding = embedding_client
        self.embedding_dim = int(embedding_client.dimension)
        self.collection_name = qcfg.collection_name

        if qcfg.embedding_dim != self.embedding_dim:
            raise RuntimeError(
                f"SKILL_QDRANT_EMBED_DIM ({qcfg.embedding_dim}) does not match "
                f"CLOUD_EMBEDDING_DIM ({self.embedding_dim}). Fix your .env values."
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection or validate an existing one.

        New environments get a cosine-similarity collection with the configured
        embedding size. Existing environments are checked for an unnamed vector
        schema and matching dimensions before any skill vectors are written.
        """
        collections = self.client.get_collections().collections
        existing = {collection.name for collection in collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            self._ensure_payload_indexes()
            return

        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if not hasattr(vectors, "size"):
            raise RuntimeError(
                f"Invalid vector schema for collection '{self.collection_name}': "
                f"{vectors}. Expected unnamed VectorParams."
            )
        if vectors.size != self.embedding_dim:
            raise RuntimeError(
                f"Vector dimension mismatch for collection '{self.collection_name}': "
                f"expected {self.embedding_dim}, got {vectors.size}"
            )
        self._ensure_payload_indexes(getattr(info, "payload_schema", None) or {})

    def _ensure_payload_indexes(self, payload_schema: dict | None = None) -> None:
        """Ensure payload indexes needed for filtering and maintenance exist.

        The app filters, deletes, and inspects skill points by fields such as
        ``skill_id`` and ``chunk_type``. Index creation is best-effort because
        another Streamlit worker may create the same index at startup.
        """
        existing = payload_schema or {}
        for field_name in _INDEXED_PAYLOAD_FIELDS:
            if field_name in existing:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                # Another app worker may have created the index first. Search still
                # works without the local process failing startup.
                continue

    def index_skill(self, skill: Any) -> list[str]:
        """Embed and upsert all Qdrant points for one SQL skill row.

        The method first removes previous points for the same ``skill_id`` so
        edited names, instructions, symbols, active state, or markdown files are
        reflected cleanly. Inactive skills may still be indexed, but runtime
        search receives active ids from SQL and excludes inactive rows.
        """
        skill_id = _string_value(getattr(skill, "skill_id", None))
        if not skill_id:
            return []

        self.delete_skill(skill_id)

        points: list[PointStruct] = []
        for point_id, text, payload in self._skill_points(skill):
            vector = _embed_text(self.embedding, text, input_type="document")
            if len(vector) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension {len(vector)} does not match "
                    f"expected {self.embedding_dim}"
                )
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if not points:
            return []
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return [str(point.id) for point in points]

    def delete_skill(self, skill_id: str) -> None:
        """Delete every vector point associated with one skill id.

        This is used before re-indexing and can also be used for hard cleanup
        when a skill is removed from SQL. Empty ids are ignored to avoid broad
        accidental deletes.
        """
        clean_id = str(skill_id or "").strip()
        if not clean_id:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="skill_id",
                        match=MatchValue(value=clean_id),
                    )
                ]
            ),
            wait=True,
        )

    def search(
        self,
        *,
        query: str,
        top_k: int = 2,
        active_skill_ids: Iterable[str] | None = None,
        exclude_skill_ids: Iterable[str] | None = None,
        min_score: float | None = None,
    ) -> list[SkillVectorMatch]:
        """Return semantically relevant skills after applying SQL allow-lists.

        ``active_skill_ids`` is the critical safety filter. When it is provided,
        only those SQL-active skill ids can be returned, even if stale vectors
        still exist in Qdrant. ``exclude_skill_ids`` prevents already selected
        skills from being returned again as semantic additions.
        """
        clean_query = str(query or "").strip()
        if not clean_query or top_k <= 0:
            return []

        active_ids: set[str] | None = None
        if active_skill_ids is not None:
            active_ids = {
                str(item).strip() for item in active_skill_ids if str(item).strip()
            }
            if not active_ids:
                return []

        query_vector = _embed_text(self.embedding, clean_query, input_type="query")
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(top_k * 8, top_k),
            with_payload=True,
        )

        excluded = {
            str(item).strip().lower()
            for item in (exclude_skill_ids or [])
            if str(item).strip()
        }
        by_skill: dict[str, SkillVectorMatch] = {}
        for point in results.points:
            payload = point.payload or {}
            skill_id = _string_value(payload.get("skill_id"))
            slug = _string_value(payload.get("slug"))
            if not skill_id:
                continue
            if active_ids is not None and skill_id not in active_ids:
                continue
            if skill_id.lower() in excluded or slug.lower() in excluded:
                continue
            if (
                min_score is not None
                and point.score is not None
                and point.score < min_score
            ):
                continue
            current = by_skill.get(skill_id)
            if current is not None and (current.score or 0.0) >= (point.score or 0.0):
                continue
            by_skill[skill_id] = SkillVectorMatch(
                skill_id=skill_id,
                slug=slug,
                score=point.score,
                payload=payload,
            )

        matches = sorted(
            by_skill.values(),
            key=lambda match: (-(match.score or -math.inf), match.skill_id),
        )
        return matches[:top_k]

    def _skill_points(self, skill: Any) -> list[tuple[str, str, dict[str, Any]]]:
        """Build the profile and markdown-context points for one skill.

        A skill always gets one compact ``skill_profile`` point from its SQL
        fields. Any configured markdown files are read from storage, chunked,
        and added as ``skill_context`` points so long SOP-style skills can be
        retrieved by the specific section that matches the user message.
        """
        skill_id = _string_value(getattr(skill, "skill_id", None))
        name = _string_value(getattr(skill, "name", None), "Skill")
        symbol = _string_value(getattr(skill, "symbol", None))
        description = _string_value(getattr(skill, "description", None))
        instruction = _string_value(getattr(skill, "instruction", None))
        source_type = _string_value(getattr(skill, "source_type", None), "uploaded")
        slug = _skill_slug(skill)
        is_active = bool(getattr(skill, "is_active", True))

        profile_text = _bounded_text(
            "\n".join(
                part
                for part in (
                    f"Skill name: {name}",
                    f"Skill slug: {slug}",
                    f"Skill symbol: {symbol}" if symbol else "",
                    f"Source type: {source_type}",
                    f"Description: {description}" if description else "",
                    f"Instruction: {instruction}" if instruction else "",
                )
                if part
            ),
            max_chars=_PROFILE_CONTEXT_CHARS,
        )

        common_payload = {
            "source": "furnacemind_skill",
            "skill_id": skill_id,
            "slug": slug,
            "name": name,
            "symbol": symbol,
            "source_type": source_type,
            "is_active": is_active,
        }
        points: list[tuple[str, str, dict[str, Any]]] = []
        if profile_text:
            points.append(
                (
                    _point_id(skill_id, "profile"),
                    profile_text,
                    {
                        **common_payload,
                        "chunk_type": "skill_profile",
                        "chunk_index": 0,
                        "content": profile_text,
                    },
                )
            )

        chunk_index = 1
        for filename in _context_files(skill):
            text = _read_context_file(filename)
            if not text:
                continue
            for chunk in _chunk_text(text):
                content = f"{filename}\n\n{chunk}"
                points.append(
                    (
                        _point_id(skill_id, "context", chunk_index),
                        content,
                        {
                            **common_payload,
                            "chunk_type": "skill_context",
                            "chunk_index": chunk_index,
                            "context_file": filename,
                            "content": _bounded_text(content, max_chars=3_500),
                        },
                    )
                )
                chunk_index += 1
        return points
