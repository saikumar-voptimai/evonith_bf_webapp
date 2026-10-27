"""Session tokens: JSON Web Tokens (RFC 7519) signed with HMAC-SHA256 (JWS HS256, RFC 7515), via PyJWT.

Only HS256 is accepted (``algorithms=["HS256"]``), so ``"alg": "none"`` and other algorithms are refused. PyJWT checks
the signature and the presence of ``exp``; the validity period is checked here against the caller's clock, so the API
and its tests use one time source. Times are whole seconds since the Unix epoch (UTC).
"""

from __future__ import annotations

from typing import Any

import jwt

ALGORITHM = "HS256"


class TokenError(Exception):
    """The token is malformed, carries a bad signature, or is outside its validity period."""


def encode(claims: dict[str, Any], key: bytes) -> str:
    return jwt.encode(claims, key, algorithm=ALGORITHM)


def _int_claim(claims: dict[str, Any], name: str) -> int | None:
    value = claims.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TokenError(f"Claim {name} must be an integer.")
    return value


def decode(token: str, key: bytes, *, now: int, leeway: int = 30) -> dict[str, Any]:
    """Verify the signature and the validity period; return the claims.

    ``exp`` is required; ``iat`` and ``nbf``, when present, may not lie in the future by more than ``leeway`` seconds.
    """
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=[ALGORITHM],
            options={"require": ["exp"], "verify_exp": False, "verify_iat": False, "verify_nbf": False},
        )
    except jwt.InvalidTokenError as exc:
        raise TokenError(str(exc)) from exc
    exp = _int_claim(claims, "exp")
    if exp is None or now > exp + leeway:
        raise TokenError("The token has expired.")
    for name in ("iat", "nbf"):
        value = _int_claim(claims, name)
        if value is not None and value > now + leeway:
            raise TokenError(f"The token is not valid yet ({name}).")
    return claims
