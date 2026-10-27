"""The twin store for the routes (SV-1.6, #286): ``services.store.twin_store("twin-api")``.

``twin_store`` answers 503 when no database is configured (``BF_TWIN_DB_URL`` / ``DATABASE_URL``); routes that also
work without one take ``optional_twin_store``. A database that cannot be reached answers 503 through the error handler
of ``api.errors``.
"""

from __future__ import annotations

from api.errors import ApiError
from data.twin.repositories import TwinStore
from services import store as store_service


def optional_twin_store() -> TwinStore | None:
    return store_service.twin_store("twin-api")


def twin_store() -> TwinStore:
    store = optional_twin_store()
    if store is None:
        raise ApiError(503, "The twin store is not configured: set BF_TWIN_DB_URL or DATABASE_URL.")
    return store
