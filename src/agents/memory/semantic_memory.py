"""Long-term semantic memory for FurnaceMind.

Durable memories are saved to PostgreSQL first, then indexed into Qdrant for
runtime retrieval. PostgreSQL is the backup/recovery layer; Qdrant is the only
store used for semantic search and prompt-context injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from agents.furnacemind import prompts
from furnace_data import relational
from utils.logger import get_logger
from utils.settings import SemanticMemoryConfig, settings

logger = get_logger(__name__)

_CONTEXT_TITLE = "LONG-TERM SEMANTIC MEMORY (retrieved by relevance):"
_SUMMARY_SOURCE = "furnacemind_summary"
_SNAPSHOT_SOURCE = "furnacemind_summary_snapshot"
_MAX_EXTRACTED_FACTS = 8


@dataclass(frozen=True)
class SemanticMemoryRecord:
    """
    Prompt-ready memory returned from the Qdrant runtime store.

    Attributes:
        memory: Durable memory text to inject into the LLM context.
        score: Optional Qdrant similarity score for the query match.
        metadata: Raw Qdrant payload retained for debugging/audit use.
        memory_id: PostgreSQL ``memory_facts.fact_id`` when available.
    """

    memory: str
    score: float | None = None
    metadata: dict[str, Any] | None = None
    memory_id: str | None = None


@dataclass(frozen=True)
class _MemoryDraft:
    """
    Candidate memory prepared from one cumulative conversation summary.

    ``SemanticMemoryService.add_summary`` converts each draft into a PostgreSQL
    ``memory_facts`` row first. Only successfully saved rows are indexed into
    Qdrant.
    """

    memory: str
    metadata: dict[str, Any]


def _point_id(fact_id: str) -> str:
    """
    Convert a PostgreSQL fact id into a deterministic Qdrant point id.

    The existing SQL table stores ids as arbitrary text, while Qdrant accepts
    UUID strings or unsigned integers. Using UUIDv5 makes retries idempotent:
    the same SQL fact always maps to the same Qdrant point.

    Args:
        fact_id: Primary key from ``furnace_mind.memory_facts``.

    Returns:
        UUID string safe to use as a Qdrant point id.
    """
    return str(uuid5(NAMESPACE_URL, f"furnacemind-memory-fact:{fact_id}"))


class SemanticMemoryVectorStore:
    """
    Qdrant adapter used for semantic-memory indexing and runtime retrieval.

    This class does not talk to PostgreSQL. It receives SQL-backed fact ids from
    ``SemanticMemoryService``, embeds the memory text, writes vectors to Qdrant,
    and later searches Qdrant for memories owned by the current user.
    """

    def __init__(
        self,
        embedding_client: Any,
        *,
        config: SemanticMemoryConfig,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        """
        Connect to Qdrant and ensure the memory collection is usable.

        The active embedding client's dimension is used for collection creation
        and validation. Missing collections are created with cosine distance.

        Args:
            embedding_client: Client exposing ``embed_text(text)`` and
                ``dimension``.
            config: Semantic-memory runtime settings.
            qdrant_url: Optional Qdrant endpoint override, mainly for tests.
            qdrant_api_key: Optional Qdrant API key override.

        Raises:
            RuntimeError: If an existing Qdrant collection has an unexpected
                vector schema or dimension.
        """
        qcfg = settings.qdrant_knowledge
        self.client = QdrantClient(
            url=qdrant_url or qcfg.url,
            api_key=qdrant_api_key if qdrant_api_key is not None else qcfg.api_key,
            timeout=qcfg.timeout,
        )
        self.embedding = embedding_client
        self.embedding_dim = int(embedding_client.dimension)
        self.collection_name = config.collection_name
        self.search_threshold = config.search_threshold

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create the Qdrant collection or validate the existing vector schema.

        The memory collection uses one unnamed dense vector. If a collection
        with the configured name already exists, its vector size must match the
        active cloud embedding client.

        Raises:
            RuntimeError: If the existing collection has an unexpected vector
                schema or dimension.
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
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """
        Create Qdrant payload indexes used by search filters and inspection.

        Index creation is best-effort because Qdrant may report a conflict if
        another worker created the index first. Those conflicts are logged at
        debug level and do not block startup.
        """
        info = self.client.get_collection(self.collection_name)
        schema = info.payload_schema or {}
        for field_name in (
            "source",
            "conversation_id",
            "user_id",
            "fact_id",
        ):
            if field_name in schema:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.debug(
                    "Payload index %s may already exist in %s: %s",
                    field_name,
                    self.collection_name,
                    exc,
                )

    def upsert_fact(
        self,
        *,
        fact_id: str,
        user_id: str,
        fact: str,
        metadata: dict[str, Any],
    ) -> str:
        """
        Embed and upsert one PostgreSQL-backed memory fact into Qdrant.

        This method is called only after the fact has been committed to
        ``furnace_mind.memory_facts``. The payload keeps both ``fact_id`` and
        ``memory`` so dashboard inspection and later recovery can trace a
        vector back to SQL.

        Args:
            fact_id: PostgreSQL ``memory_facts.fact_id`` value.
            user_id: Owner of the memory.
            fact: Memory text to embed and retrieve later.
            metadata: Audit/source metadata copied from the SQL write path.

        Returns:
            Qdrant point id used for the upsert.

        Raises:
            ValueError: If the embedding provider returns the wrong vector size.
        """
        vector = self.embedding.embed_text(fact)
        if len(vector) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match "
                f"expected {self.embedding_dim}"
            )

        point_id = _point_id(fact_id)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        **metadata,
                        "user_id": user_id,
                        "fact_id": fact_id,
                        "data": fact,
                        "memory": fact,
                    },
                )
            ],
            wait=True,
        )
        return point_id

    def delete_points(self, point_ids: list[str]) -> int:
        """Delete semantic-memory Qdrant points by id."""
        ids = [str(point_id) for point_id in point_ids if str(point_id).strip()]
        if not ids:
            return 0
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
            wait=True,
        )
        return len(ids)

    def search_memories(
        self,
        *,
        query: str,
        user_id: str,
        limit: int,
    ) -> list[SemanticMemoryRecord]:
        """
        Search Qdrant for memories relevant to one user query.

        Runtime retrieval intentionally uses Qdrant only. PostgreSQL is not read
        here; it is the durable backup and recovery source. Results are scoped
        by ``user_id`` before they are normalized for prompt injection.

        Args:
            query: Current user request.
            user_id: User whose long-term memories may be retrieved.
            limit: Maximum number of Qdrant matches to return.

        Returns:
            Qdrant matches normalized as ``SemanticMemoryRecord`` objects.
        """
        query_vector = self.embedding.embed_text(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    ),
                ]
            ),
            limit=limit,
            with_payload=True,
        )

        records: list[SemanticMemoryRecord] = []
        for point in results.points:
            if (
                self.search_threshold is not None
                and point.score is not None
                and point.score < self.search_threshold
            ):
                continue
            payload = point.payload or {}
            memory_text = str(
                payload.get("memory") or payload.get("data") or ""
            ).strip()
            if not memory_text:
                continue
            fact_id = payload.get("fact_id") or payload.get("memory_id")
            records.append(
                SemanticMemoryRecord(
                    memory=memory_text,
                    score=point.score,
                    metadata=payload,
                    memory_id=str(fact_id) if fact_id else None,
                )
            )
        return records


class SemanticMemoryService:
    """
    Orchestrates SQL-first persistence and Qdrant-only runtime retrieval.

    The service has two separate modes:
    - read path: search Qdrant and format memories for prompt context.
    - write path: extract facts from cumulative summaries, save them to
      PostgreSQL, then index the saved facts into Qdrant.
    """

    def __init__(
        self,
        *,
        config: SemanticMemoryConfig | None = None,
        repository: Any | None = None,
        vector_store: Any | None = None,
        extraction_llm: Any | None = None,
        embedding_client: Any | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        """
        Initialize the semantic-memory service and optional provider clients.

        Tests may pass fake repository, vector store, or LLM instances. In
        production, missing dependencies are built lazily from application
        settings. Initialization failures are stored in ``last_error`` and
        logged, allowing FurnaceMind chat to continue without semantic memory.

        Args:
            config: Optional semantic-memory settings override.
            repository: Optional SQL repository override.
            vector_store: Optional Qdrant vector-store override.
            extraction_llm: Optional LLM used to extract durable facts.
            embedding_client: Optional embedding client used by the vector store.
            qdrant_url: Optional Qdrant endpoint override.
            qdrant_api_key: Optional Qdrant API key override.
        """
        self.config = config or settings.semantic_memory
        self._repository = repository
        self._vector_store = vector_store
        self._extraction_llm = extraction_llm
        self._last_error: str | None = None

        if not self.config.enabled:
            return

        if self._repository is None:
            try:
                engine = relational.build_relational_engine()
                session_factory = relational.build_relational_session_factory(engine)
                repository_cls = getattr(relational, "MemoryFactRepository", None)
                if repository_cls is None:
                    from furnace_data.relational.repositories import (
                        MemoryFactRepository,
                    )

                    repository_cls = MemoryFactRepository
                self._repository = repository_cls(session_factory)
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Semantic memory SQL repository unavailable: %s", exc)

        if self._vector_store is None:
            try:
                from agents.embeddings.cloud_embedding import CloudEmbeddingClient

                self._vector_store = SemanticMemoryVectorStore(
                    embedding_client or CloudEmbeddingClient(),
                    config=self.config,
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key,
                )
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Semantic memory Qdrant store unavailable: %s", exc)

        if self._extraction_llm is None:
            try:
                from agents.llm.llm_client import OpenRouterClient

                self._extraction_llm = OpenRouterClient(
                    model_name=self.config.llm_model
                )
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Semantic memory extraction LLM unavailable: %s", exc)

    @property
    def available(self) -> bool:
        """
        Report whether runtime Qdrant retrieval can be used.

        Returns:
            True when semantic memory is enabled and the vector store was
            initialized successfully.
        """
        return self._vector_store is not None and self.config.enabled

    @property
    def storage_available(self) -> bool:
        """
        Report whether the full summary-to-memory write path can run.

        Returns:
            True only when semantic memory is enabled and SQL repository,
            Qdrant vector store, and extraction LLM are all available.
        """
        return (
            self.config.enabled
            and self._repository is not None
            and self._vector_store is not None
            and self._extraction_llm is not None
        )

    @property
    def last_error(self) -> str | None:
        """
        Return the latest semantic-memory error captured by this service.

        Returns:
            Error message from initialization, extraction, SQL persistence,
            Qdrant indexing, or Qdrant retrieval. ``None`` means no error has
            been observed by this service instance.
        """
        return self._last_error

    def search(
        self,
        *,
        query: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[SemanticMemoryRecord]:
        """
        Retrieve relevant long-term memories from Qdrant.

        SQL is deliberately not queried in this method; runtime memory search
        must stay on the vector store. Any Qdrant/embedding error is logged and
        converted to an empty result so chat responses still work.

        Args:
            query: Current user request.
            user_id: User whose memories should be searched.
            limit: Optional override for maximum returned memories.

        Returns:
            Relevant Qdrant memories, or an empty list when unavailable.
        """
        if not self.available or not user_id or not query.strip():
            return []

        try:
            return self._vector_store.search_memories(
                query=query,
                user_id=user_id,
                limit=limit or self.config.max_memories,
            )
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Semantic memory Qdrant search failed: %s", exc)
            return []

    def context_for_query(self, *, query: str, user_id: str) -> str:
        """
        Build the prompt block containing relevant long-term memories.

        Args:
            query: Current user request.
            user_id: User whose memories should be searched.

        Returns:
            A labeled bullet-list context block, or an empty string when no
            memories are found.
        """
        records = self.search(query=query, user_id=user_id)
        if not records:
            return ""

        lines = []
        for record in records:
            text = record.memory.strip()
            if not text:
                continue
            score = f" (score={record.score:.3f})" if record.score is not None else ""
            lines.append(f"- {text}{score}")

        if not lines:
            return ""
        return _CONTEXT_TITLE + "\n" + "\n".join(lines)

    def add_summary(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        summary: str,
        source_message_id_start: str | None = None,
        source_message_id_end: str | None = None,
        summarized_message_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Extract durable memories from a summary and persist them SQL-first.

        The input is the latest cumulative conversation summary, not raw chat.
        Extracted facts are de-duplicated against SQL facts for the user. Each
        new fact is committed to PostgreSQL first; only then is it embedded and
        upserted to Qdrant. If Qdrant indexing fails after SQL save, the SQL row
        remains available for recovery via ``sync_unindexed_facts``.

        Args:
            user_id: User that owns the extracted memories.
            conversation_id: Source conversation id for auditability.
            summary: Latest cumulative conversation summary.
            source_message_id_start: First message represented by the summary window.
            source_message_id_end: Last message represented by the summary window.
            summarized_message_count: Number of chat messages covered by the summary.
            metadata: Extra source metadata stored with every memory fact.

        Returns:
            PostgreSQL fact ids created during this call. Existing duplicate
            facts and failed SQL writes are omitted.
        """
        summary_text = str(summary or "").strip()
        if not self.storage_available or not user_id or not summary_text:
            return []

        try:
            facts = _extract_memory_facts(
                summary=summary_text,
                llm=self._extraction_llm,
            )
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Semantic memory extraction failed: %s", exc)
            return []

        drafts = _build_memory_drafts(
            facts=facts,
            summary=summary_text,
            conversation_id=conversation_id,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
            summarized_message_count=summarized_message_count,
            metadata=metadata,
            extraction_model=getattr(self._extraction_llm, "model", None),
        )
        saved_fact_ids: list[str] = []
        for draft in drafts:
            fact_row = None
            try:
                if _fact_exists(
                    self._repository,
                    user_id=user_id,
                    fact_text=draft.memory,
                ):
                    continue
                fact_row = self._repository.create_fact(
                    user_id=user_id,
                    fact_text=draft.memory,
                    source_conversation_id=conversation_id,
                    qdrant_collection=None,
                    qdrant_point_id=None,
                    metadata=draft.metadata,
                )
                saved_fact_ids.append(fact_row.fact_id)
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Semantic memory SQL save failed: %s", exc)
                continue

            try:
                point_id = self._vector_store.upsert_fact(
                    fact_id=fact_row.fact_id,
                    user_id=user_id,
                    fact=draft.memory,
                    metadata=draft.metadata,
                )
                if hasattr(self._repository, "mark_fact_indexed"):
                    self._repository.mark_fact_indexed(
                        fact_id=fact_row.fact_id,
                        qdrant_collection=self._vector_store.collection_name,
                        qdrant_point_id=point_id,
                    )
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning(
                    "Semantic memory Qdrant index failed after SQL save %s: %s",
                    getattr(fact_row, "fact_id", ""),
                    exc,
                )
        return saved_fact_ids

    def sync_unindexed_facts(self, *, limit: int = 100) -> int:
        """
        Index SQL-saved memory facts that are missing Qdrant point ids.

        This is the recovery path for cases where PostgreSQL persistence
        succeeded but Qdrant indexing or embedding failed. It reads unindexed
        rows from SQL, upserts them into Qdrant, and writes the Qdrant location
        back to SQL.

        Args:
            limit: Maximum number of SQL rows to attempt in one call.

        Returns:
            Count of facts successfully indexed into Qdrant.
        """
        if (
            not self.config.enabled
            or self._repository is None
            or self._vector_store is None
        ):
            return 0
        if not hasattr(self._repository, "list_unindexed_facts"):
            return 0

        indexed = 0
        for fact_row in self._repository.list_unindexed_facts(limit=limit):
            metadata = (
                fact_row.metadata_json
                if isinstance(fact_row.metadata_json, dict)
                else {}
            )
            try:
                point_id = self._vector_store.upsert_fact(
                    fact_id=fact_row.fact_id,
                    user_id=str(fact_row.user_id),
                    fact=fact_row.fact_text,
                    metadata=metadata,
                )
                if hasattr(self._repository, "mark_fact_indexed"):
                    self._repository.mark_fact_indexed(
                        fact_id=fact_row.fact_id,
                        qdrant_collection=self._vector_store.collection_name,
                        qdrant_point_id=point_id,
                    )
                indexed += 1
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning(
                    "Semantic memory recovery index failed for %s: %s",
                    getattr(fact_row, "fact_id", ""),
                    exc,
                )
        return indexed

    def delete_document_related_memories(
        self,
        *,
        user_id: str,
        sql_document_id: str,
        mrag_document_id: str = "",
        filename: str = "",
    ) -> int:
        """Remove semantic memories derived from a deleted knowledge document.

        Memory facts are looked up in PostgreSQL first. Their Qdrant vectors are
        deleted before the SQL rows are removed, so a Qdrant failure does not
        leave untracked runtime memories behind.
        """
        if not self.config.enabled or self._repository is None or not user_id:
            return 0
        if not hasattr(self._repository, "list_document_related_facts"):
            return 0

        try:
            facts = self._repository.list_document_related_facts(
                user_id=user_id,
                sql_document_id=sql_document_id,
                mrag_document_id=mrag_document_id,
                filename=filename,
            )
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Semantic memory document lookup failed: %s", exc)
            raise

        if not facts:
            return 0

        point_ids = [
            str(getattr(fact, "qdrant_point_id", "") or "").strip()
            for fact in facts
            if str(getattr(fact, "qdrant_point_id", "") or "").strip()
        ]
        if point_ids and self._vector_store is not None:
            try:
                self._vector_store.delete_points(point_ids)
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning("Semantic memory Qdrant delete failed: %s", exc)
                raise

        fact_ids = [str(getattr(fact, "fact_id", "") or "") for fact in facts]
        try:
            if hasattr(self._repository, "delete_facts"):
                return self._repository.delete_facts(fact_ids)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Semantic memory SQL delete failed: %s", exc)
            raise
        return 0


def _extract_memory_facts(*, summary: str, llm: Any) -> list[str]:
    """
    Ask the extraction LLM for durable memory facts from a cumulative summary.

    Args:
        summary: Latest cumulative FurnaceMind conversation summary.
        llm: Client exposing ``generate(system_prompt=..., user_prompt=...)``.

    Returns:
        De-duplicated fact strings parsed from the LLM response.
    """
    response = llm.generate(
        system_prompt=prompts.semantic_memory_fact_extraction_prompt(),
        user_prompt=(
            "Cumulative FurnaceMind conversation summary:\n\n" f"{summary.strip()}"
        ),
    )
    return _parse_facts_response(response)


def _parse_facts_response(response: str) -> list[str]:
    """
    Parse and normalize facts returned by the extraction LLM.

    The expected response is JSON in the shape ``{"facts": ["..."]}``, but the
    parser also tolerates fenced JSON, a raw JSON list, or a plain bullet list.
    Blank facts and duplicate facts are removed while preserving first-seen
    order.

    Args:
        response: Raw LLM response text.

    Returns:
        Up to ``_MAX_EXTRACTED_FACTS`` normalized fact strings.
    """
    text = str(response or "").strip()
    if not text:
        return []

    payload: Any | None = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        json_text = _extract_json_object(text)
        if json_text:
            try:
                payload = json.loads(json_text)
            except json.JSONDecodeError:
                payload = None

    if isinstance(payload, dict):
        raw_facts = payload.get("facts") or []
    elif isinstance(payload, list):
        raw_facts = payload
    else:
        raw_facts = _fallback_fact_lines(text)

    facts: list[str] = []
    seen: set[str] = set()
    for item in raw_facts:
        fact = str(item or "").strip()
        if not fact:
            continue
        normalized = " ".join(fact.split())
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(normalized)
        if len(facts) >= _MAX_EXTRACTED_FACTS:
            break
    return facts


def _extract_json_object(text: str) -> str | None:
    """
    Return the first JSON object embedded in a possibly fenced response.

    Args:
        text: Raw response text that may include markdown code fences.

    Returns:
        JSON object text including braces, or ``None`` when none is found.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return cleaned[start : end + 1]


def _fallback_fact_lines(text: str) -> list[str]:
    """
    Parse a non-JSON response as one fact per non-empty line.

    Args:
        text: Raw response text, usually a bullet or numbered list.

    Returns:
        Cleaned line items with common bullet/number prefixes removed.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("-*0123456789. )").strip()
        if stripped:
            lines.append(stripped)
    return lines


def _build_memory_drafts(
    *,
    facts: list[str],
    summary: str,
    conversation_id: str | None,
    source_message_id_start: str | None,
    source_message_id_end: str | None,
    summarized_message_count: int | None,
    metadata: dict[str, Any] | None,
    extraction_model: str | None,
) -> list[_MemoryDraft]:
    """
    Build memory drafts and source metadata from extracted facts.

    The service stores each extracted fact and one summary snapshot. The
    snapshot preserves the exact cumulative summary that produced the facts,
    which helps auditing and recovery without reading chat transcripts.

    Args:
        facts: Durable facts extracted from the cumulative summary.
        summary: Cumulative summary text used as extraction input.
        conversation_id: Source conversation id.
        source_message_id_start: First summarized message id.
        source_message_id_end: Last summarized message id.
        summarized_message_count: Number of messages covered by the summary.
        metadata: Extra caller metadata copied to every draft.
        extraction_model: LLM model name used for extraction, when known.

    Returns:
        Drafts ready for SQL duplicate checks and SQL-first persistence.
    """
    base_metadata = {
        **(metadata or {}),
        "conversation_id": conversation_id or "",
        "source_message_id_start": source_message_id_start or "",
        "source_message_id_end": source_message_id_end or "",
        "extraction_model": extraction_model or "",
    }
    if summarized_message_count is not None:
        base_metadata["summarized_message_count"] = summarized_message_count

    drafts = [
        _MemoryDraft(
            memory=fact,
            metadata={
                **base_metadata,
                "source": _SUMMARY_SOURCE,
                "memory_kind": "extracted_fact",
            },
        )
        for fact in facts
    ]
    snapshot = "FurnaceMind cumulative conversation summary snapshot:\n\n" f"{summary}"
    drafts.append(
        _MemoryDraft(
            memory=snapshot,
            metadata={
                **base_metadata,
                "source": _SNAPSHOT_SOURCE,
                "memory_kind": "summary_snapshot",
            },
        )
    )
    return drafts


def _fact_exists(repository: Any, *, user_id: str, fact_text: str) -> bool:
    """
    Check whether the repository already has a matching fact.

    Args:
        repository: SQL repository, optionally exposing ``fact_exists``.
        user_id: User that owns the memory.
        fact_text: Candidate fact text.

    Returns:
        True when the repository can confirm a duplicate; otherwise False so
        older/fake repositories remain compatible.
    """
    if not hasattr(repository, "fact_exists"):
        return False
    return bool(repository.fact_exists(user_id=user_id, fact_text=fact_text))
