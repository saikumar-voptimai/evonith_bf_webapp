"""Authentication service wrapping the :class:`~data.db.Database` layer.

Provides a thin facade so that UI pages and admin panels call
high-level ``authenticate`` / ``register`` methods instead of
accessing the database object directly.
"""

# domain/auth_service.py
# Using the database class for authentication and registration


class AuthService:
    """High-level authentication and user registration service.

    Attributes:
        db: A :class:`~data.db.Database` instance used for credential storage.
    """

    def __init__(self, db) -> None:
        """Initialise the service with a database backend.

        Args:
            db: Initialised :class:`~data.db.Database` instance.
        """
        self.db = db

    def authenticate(self, username: str, password: str) -> bool:
        """Validate a username/password pair against the database.

        Args:
            username: Plaintext username to look up.
            password: Plaintext password to verify (hashed in DB).

        Returns:
            ``True`` if credentials are valid, ``False`` otherwise.
        """
        return self.db.validate_user(username, password)

    def register(self, username: str, password: str, role: str = "user") -> None:
        """Register a new user in the database.

        Args:
            username: Username for the new account.
            password: Plaintext password (will be hashed by the DB layer).
            role:     Assigned role — ``"admin"``, ``"supervisor"``, or ``"user"``.
        """
        self.db.add_user(username, password, role)
