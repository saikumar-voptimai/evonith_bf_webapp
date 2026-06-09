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
    BURDEN_VALUE_COLUMNS,
    HOPPER_COLUMNS,
    Base,
    BurdenDistributionHistory,
    BurdenHistory,
    Conversation,
    ConversationMessage,
    FeedbackItem,
    Hopper,
    HopperMaterialHistory,
    HopperRawMaterialHistory,
    Material,
    MaterialCategory,
    MemoryDocument,
    MemoryFact,
    MemorySummary,
    Skill,
    Unit,
    User,
    UserRole,
    UserRoleAssignment,
)
from .repositories import (
    BurdenHistoryRepository,
    ConversationMessageRepository,
    ConversationRepository,
    FeedbackItemRepository,
    HopperHistoryRepository,
    MemoryDocumentRepository,
    MemoryFactRepository,
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
    "MemoryFact",
    "MemoryFactRepository",
    "MemoryDocument",
    "MemoryDocumentRepository",
    "MemorySummary",
    "MemorySummaryRepository",
    "Skill",
    "SkillRepository",
]
