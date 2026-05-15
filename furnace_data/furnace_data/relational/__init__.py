"""Shared relational persistence utilities for BF2 services.

Provides:
- SQLAlchemy 2.0 engine/session helpers
- ORM models for core operational tables
- Repository classes used by webapp and dataset services
"""

from .engine import (
    build_relational_engine,
    build_relational_session_factory,
    resolve_database_url,
)
from .models import (
    Base,
    BURDEN_VALUE_COLUMNS,
    HOPPER_COLUMNS,
    BurdenHistory,
    BurdenDistributionHistory,
    Conversation,
    ConversationMessage,
    FeedbackItem,
    Hopper,
    HopperRawMaterialHistory,
    HopperMaterialHistory,
    LongTermMemory,
    Material,
    MaterialCategory,
    MemoryDocument,
    MemorySummary,
    Skill,
    Unit,
    User,
    UserRoleAssignment,
    UserRole,
)
from .repositories import (
    BurdenHistoryRepository,
    ConversationRepository,
    ConversationMessageRepository,
    FeedbackItemRepository,
    HopperHistoryRepository,
    LongTermMemoryRepository,
    MemoryDocumentRepository,
    MemorySummaryRepository,
    PlantMasterRepository,
    SkillRepository,
    UserRepository,
)

__all__ = [
    "Base",
    "BURDEN_VALUE_COLUMNS",
    "HOPPER_COLUMNS",
    "BurdenHistory",
    "BurdenDistributionHistory",
    "BurdenHistoryRepository",
    "Hopper",
    "HopperHistoryRepository",
    "HopperRawMaterialHistory",
    "HopperMaterialHistory",
    "Material",
    "MaterialCategory",
    "Unit",
    "User",
    "UserRoleAssignment",
    "PlantMasterRepository",
    "UserRepository",
    "UserRole",
    "build_relational_engine",
    "build_relational_session_factory",
    "resolve_database_url",
    "Conversation",
    "ConversationMessage",
    "ConversationMessageRepository",
    "ConversationRepository",
    "FeedbackItem",
    "FeedbackItemRepository",
    "LongTermMemory",
    "LongTermMemoryRepository",
    "MemoryDocument",
    "MemoryDocumentRepository",
    "MemorySummary",
    "MemorySummaryRepository",
    "Skill",
    "SkillRepository",
]