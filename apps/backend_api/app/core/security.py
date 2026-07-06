"""Backend security helpers."""

from __future__ import annotations


def bearer_token_from_authorization(authorization: str | None) -> str | None:
    """Extract a bearer token from an Authorization header value."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.strip().lower() != "bearer" or not token.strip():
        return None
    return token.strip()
