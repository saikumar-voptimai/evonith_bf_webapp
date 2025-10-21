# domain/auth_service.py
# Using the database class for authentication and registration

class AuthService:
    def __init__(self, db):
        self.db = db

    def authenticate(self, username, password):
        return self.db.validate_user(username, password)

    def register(self, username, password, role='user'):
        self.db.add_user(username, password, role)
