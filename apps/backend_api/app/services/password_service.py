"""Password hashing and legacy verification for backend auth."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass

import bcrypt

from app.core.config import BackendSettings, load_backend_settings

try:  # Optional; bcrypt is the installed Phase 5 dependency.
    from argon2 import PasswordHasher
    from argon2.exceptions import VerificationError
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    PasswordHasher = None
    VerificationError = ValueError

_SHA256_HEX_RE = re.compile(r"^[a-fA-F0-9]{64}$")
_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")


class PasswordConfigurationError(RuntimeError):
    """Raised when configured password hashing cannot be used."""


class PasswordValidationError(ValueError):
    """Raised when a password fails local policy validation."""


@dataclass(frozen=True)
class PasswordVerificationResult:
    valid: bool
    needs_rehash: bool = False
    legacy_hash: bool = False


class PasswordService:
    """Hash and verify passwords without logging sensitive values."""

    def __init__(self, settings: BackendSettings | None = None) -> None:
        self.settings = settings or load_backend_settings()

    def validate_new_password(self, password: str) -> None:
        """Validate a new plaintext password against configured policy."""
        if len(password or "") < self.settings.auth_min_password_length:
            raise PasswordValidationError(
                f"Password must be at least {self.settings.auth_min_password_length} characters."
            )

    def hash_password(self, password: str, *, validate: bool = True) -> str:
        """Return a modern password hash."""
        if validate:
            self.validate_new_password(password)
        scheme = self.settings.auth_password_hash_scheme
        if scheme == "bcrypt":
            return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode(
                "utf-8"
            )
        if scheme == "argon2":
            if PasswordHasher is None:
                raise PasswordConfigurationError(
                    "Argon2 is configured but argon2-cffi is not installed. Use bcrypt or add argon2-cffi."
                )
            return PasswordHasher().hash(password)
        raise PasswordConfigurationError(f"Unsupported password hash scheme: {scheme}")

    def verify_password(
        self,
        *,
        password: str,
        stored_hash: str,
    ) -> PasswordVerificationResult:
        """Verify plaintext against a stored modern or legacy hash."""
        stored_hash = str(stored_hash or "").strip()
        if not stored_hash:
            return PasswordVerificationResult(valid=False)

        if stored_hash.startswith(_BCRYPT_PREFIXES):
            try:
                valid = bcrypt.checkpw(
                    password.encode("utf-8"),
                    stored_hash.encode("utf-8"),
                )
            except ValueError:
                valid = False
            return PasswordVerificationResult(
                valid=valid,
                needs_rehash=valid
                and self.settings.auth_password_hash_scheme != "bcrypt",
            )

        if stored_hash.startswith("$argon2"):
            if PasswordHasher is None:
                return PasswordVerificationResult(valid=False)
            try:
                hasher = PasswordHasher()
                valid = hasher.verify(stored_hash, password)
            except VerificationError:
                valid = False
            return PasswordVerificationResult(
                valid=valid,
                needs_rehash=valid
                and self.settings.auth_password_hash_scheme != "argon2",
            )

        if (
            self.settings.auth_allow_legacy_password_hashes
            and _SHA256_HEX_RE.fullmatch(stored_hash)
        ):
            digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
            valid = hmac.compare_digest(digest, stored_hash.lower())
            return PasswordVerificationResult(
                valid=valid,
                needs_rehash=valid
                and self.settings.auth_upgrade_legacy_hash_on_login,
                legacy_hash=valid,
            )

        return PasswordVerificationResult(valid=False)
