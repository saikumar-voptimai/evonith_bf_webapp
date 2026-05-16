"""PostgreSQL-backed conversation history helpers for FurnaceMind."""

from __future__ import annotations

from typing import Any

from furnace_data.relational import (
    ConversationMessageRepository,
    ConversationRepository,
    build_relational_engine,
    build_relational_session_factory,
)


def estimate_text_tokens(text: str | None) -> int:
    """
    Estimate the number of tokens in a text payload.

    Args:
         - text: str | None - Text content to estimate.

    Returns:
         - return: int - Approximate token count for persistence and context budgeting.
    """
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


class ConversationHistoryStore:
    """Persistence adapter for FurnaceMind chat conversations and messages."""

    def __init__(self) -> None:
        """
        Create repositories using the configured relational database URL.

        Args:
             - None

        Returns:
             - return: None - This function does not return a value.
        """
        engine = build_relational_engine()
        session_factory = build_relational_session_factory(engine)
        self._conversations = ConversationRepository(session_factory)
        self._messages = ConversationMessageRepository(session_factory)

    def ensure_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        model_mode: str | None = None,
    ) -> str:
        """
        Return an existing user conversation or create a new one.

        Args:
             - user_id: str - User that owns the conversation.
             - conversation_id: str | None - Optional existing conversation id.
             - model_mode: str | None - Optional model or reasoning mode label.

        Returns:
             - return: str - Conversation id that should be used for the chat.
        """
        if conversation_id:
            conversation = self._conversations.get_conversation(conversation_id)
            if conversation is not None and str(conversation.user_id) == user_id:
                self._conversations.touch_conversation(
                    conversation_id=conversation_id,
                    model_mode=model_mode,
                )
                return conversation_id

        conversation = self._conversations.create_conversation(
            user_id=user_id,
            model_mode=model_mode,
            metadata={"source": "furnacemind"},
        )
        return conversation.conversation_id

    def load_chat_history(self, *, conversation_id: str) -> list[dict[str, Any]]:
        """
        Load persisted messages in Streamlit chat-history format.

        Args:
             - conversation_id: str - Conversation id to load.

        Returns:
             - return: list[dict[str, Any]] - Session-compatible chat history items.
        """
        rows = self._messages.list_recent_messages(
            conversation_id=conversation_id,
            limit=200,
        )
        history = []
        for row in rows:
            if row.role not in {"user", "assistant"} or not row.content.strip():
                continue
            metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
            history.append(
                {
                    "role": row.role,
                    "content": row.content,
                    "display": metadata.get("display") or row.content,
                    "type": "text",
                    "message_id": row.message_id,
                    "conversation_id": row.conversation_id,
                }
            )
        return history

    def add_user_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        content: str,
        display: str | None = None,
    ) -> str:
        """
        Persist one user message and return its id.

        Args:
             - conversation_id: str - Conversation that owns the message.
             - user_id: str - User that owns the conversation.
             - content: str - Raw user message content.
             - display: str | None - Optional display text shown in the UI.

        Returns:
             - return: str - Created message id.
        """
        message = self._messages.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=content,
            token_count=estimate_text_tokens(content),
            metadata={"display": display or content},
        )
        self._conversations.touch_conversation(conversation_id=conversation_id)
        return message.message_id

    def add_assistant_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        content: str,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Persist one assistant message and return its id.

        Args:
             - conversation_id: str - Conversation that owns the message.
             - user_id: str - User that owns the conversation.
             - content: str - Assistant response content.
             - model: str | None - Optional model name used for the response.
             - metadata: dict[str, Any] | None - Optional message metadata.

        Returns:
             - return: str - Created message id.
        """
        message = self._messages.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=content,
            token_count=estimate_text_tokens(content),
            model=model,
            metadata=metadata or {},
        )
        self._conversations.touch_conversation(conversation_id=conversation_id)
        return message.message_id
