"""Session tokens: JSON Web Tokens (RFC 7519) signed with HMAC-SHA256 (JWS HS256, RFC 7515), standard library only.

Only HS256 is accepted. The algorithm named in the header is checked against HS256 and never used to select the
verification method, so ``"alg": "none"`` and asymmetric algorithms are refused. The signature is compared in constant
time before the claims are parsed. Times are whole seconds since the Unix epoch (UTC).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from typing import Any

_B64URL = re.compile(r"^[A-Za-z0-9_-]*$")


class TokenError(Exception):
    """The token is malformed, carries a bad signature, or is outside its validity period."""


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    if not _B64URL.match(text) or len(text) % 4 == 1:
        raise TokenError("Invalid base64url segment.")
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (binascii.Error, ValueError) as exc:
        raise TokenError("Invalid base64url segment.") from exc


def sign(signing_input: bytes, key: bytes) -> bytes:
    """HMAC-SHA256 of the JWS signing input (``BASE64URL(header) || '.' || BASE64URL(payload)``)."""
    return hmac.new(key, signing_input, hashlib.sha256).digest()


_HEADER = b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())


def encode(claims: dict[str, Any], key: bytes) -> str:
    """Compact JWS of ``claims`` (JSON with sorted keys, no whitespace)."""
    payload = b64url_encode(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
    signing_input = f"{_HEADER}.{payload}".encode("ascii")
    return f"{_HEADER}.{payload}.{b64url_encode(sign(signing_input, key))}"


def _json_object(segment: str, what: str) -> dict[str, Any]:
    try:
        value = json.loads(b64url_decode(segment))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TokenError(f"Invalid {what}.") from exc
    if not isinstance(value, dict):
        raise TokenError(f"Invalid {what}.")
    return value


def decode(token: str, key: bytes, *, now: int, leeway: int = 30) -> dict[str, Any]:
    """Verify the signature and the validity period; return the claims.

    ``exp`` is required; ``iat`` and ``nbf``, when present, may not lie in the future by more than ``leeway`` seconds.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("A token has three segments.")
    header_b64, payload_b64, signature_b64 = parts
    header = _json_object(header_b64, "header")
    if header.get("alg") != "HS256":
        raise TokenError("Only HS256 tokens are accepted.")
    expected = sign(f"{header_b64}.{payload_b64}".encode("ascii"), key)
    if not hmac.compare_digest(expected, b64url_decode(signature_b64)):
        raise TokenError("Invalid signature.")
    claims = _json_object(payload_b64, "payload")
    exp = claims.get("exp")
    if not isinstance(exp, int) or isinstance(exp, bool):
        raise TokenError("The token has no expiry.")
    if now > exp + leeway:
        raise TokenError("The token has expired.")
    for name in ("iat", "nbf"):
        value = claims.get(name)
        if value is not None and (not isinstance(value, int) or value > now + leeway):
            raise TokenError(f"The token is not valid yet ({name}).")
    return claims
