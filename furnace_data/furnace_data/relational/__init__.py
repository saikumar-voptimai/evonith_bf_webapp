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
    BurdenDistributionHistory,
    HopperMaterialHistory,
    User,
    UserRole,
)
from .repositories import (
    BurdenHistoryRepository,
    HopperHistoryRepository,
    UserRepository,
)

__all__ = [
    "Base",
    "BurdenDistributionHistory",
    "BurdenHistoryRepository",
    "HopperHistoryRepository",
    "HopperMaterialHistory",
    "User",
    "UserRepository",
    "UserRole",
    "build_relational_engine",
    "build_relational_session_factory",
    "resolve_database_url",
]
