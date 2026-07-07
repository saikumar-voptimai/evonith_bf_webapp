"""Safely create an initial backend admin user.

This script is intentionally opt-in. It never runs at application startup and
does not provide default credentials.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.backend_api.app.services.admin_service import AdminService


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def main() -> int:
    enabled = _env("EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED").lower()
    if enabled not in {"1", "true", "yes", "y", "on"}:
        print("Bootstrap admin disabled. Set EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED=true to run.")
        return 2

    username = _env("EVONITH_BOOTSTRAP_ADMIN_USERNAME")
    password = _env("EVONITH_BOOTSTRAP_ADMIN_PASSWORD")
    email = _env("EVONITH_BOOTSTRAP_ADMIN_EMAIL") or None
    full_name = _env("EVONITH_BOOTSTRAP_ADMIN_FULL_NAME") or None

    if not username or not password:
        print("EVONITH_BOOTSTRAP_ADMIN_USERNAME and EVONITH_BOOTSTRAP_ADMIN_PASSWORD are required.")
        return 2

    service = AdminService()
    existing = service.repository.find_by_username_or_email(username)
    if existing is not None:
        print(f"Admin bootstrap skipped: user '{username}' already exists.")
        return 0

    user = service.create_user(
        username=username,
        password=password,
        role="admin",
        email=email,
        full_name=full_name,
        is_active=True,
    )
    print(f"Created admin user: {user['username']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
