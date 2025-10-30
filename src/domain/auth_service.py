# domain/auth_service.py
# Using the database class for authentication and registration

class AuthService:
    def __init__(self, db):
        self.db = db

    def authenticate(self, username, password) -> bool:
        return self.db.validate_user(username, password)

    def register(self, username, password, role='user') -> None:
        self.db.add_user(username, password, role)
