"""Ticketing persistence and service interfaces.

This package provides a SQLite-first, PostgreSQL-ready ticketing layer used by
the Feedback page. The public entry point is :class:`TicketService`.
"""

from .models import TicketCriticality, TicketStatus
from .service import (
    TicketCreateRequest,
    TicketEventView,
    TicketQueryFilter,
    TicketService,
    TicketStatusUpdateRequest,
    TicketView,
)

__all__ = [
    "TicketCreateRequest",
    "TicketCriticality",
    "TicketEventView",
    "TicketQueryFilter",
    "TicketService",
    "TicketStatus",
    "TicketStatusUpdateRequest",
    "TicketView",
]
