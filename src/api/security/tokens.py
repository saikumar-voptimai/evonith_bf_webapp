"""Session tokens: JSON Web Tokens (RFC 7519) signed with HMAC-SHA256 (JWS HS256, RFC 7515), via PyJWT.

The implementation lives in ``services.session_tokens`` (SV-6.6, #320), so the Streamlit pages under ``/legacy`` verify
the API session cookie by the same rules without importing FastAPI; this module keeps the API's import path.
"""

from __future__ import annotations

from services.session_tokens import ALGORITHM, TokenError, decode, encode

__all__ = ["ALGORITHM", "TokenError", "decode", "encode"]
