# src/data/db.py
import os
import hashlib
from collections import defaultdict
import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#  Use your Neon/Supabase DB URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("Missing DATABASE_URL environment variable. Please set it in your .env file.")

# Create SQLAlchemy engine (connection pooling included)
engine = create_engine(DATABASE_URL, future=True)

# Path to materials.yml
MATERIALS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "materials.yml"
)


class Database:
    """
    A PostgreSQL (Neon) database handler for managing users and material-to-hopper mappings.

    This class handles:
        - User authentication and management.
        - Loading and mapping materials from a YAML configuration file.
        - Managing material-hopper assignments.

    Attributes:
        materials (list): List of materials loaded from the YAML file.
    """

    def __init__(self) -> None:
        """Initializes the database by loading materials and creating necessary tables."""
        self.materials = self.load_materials_from_yml()
        self.create_users_table()
        self.create_material_hoppers_table()

    # ---------------- USERS ---------------
    def create_users_table(self)-> None:
        """
        Creates the 'users' table if it does not exist.

        The table includes the following fields:
            - username (TEXT, primary key)
            - password_hash (TEXT)
            - role (TEXT, must be either 'admin' or 'user')

        If no 'admin' user exists, a default admin user ('admin' / 'admin123') is created.
        """
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT,
                    role TEXT CHECK (role IN ('admin', 'user')) NOT NULL
                )
            '''))

        # Ensure default admin user exists
        with engine.begin() as conn:
            cur = conn.execute(text("SELECT 1 FROM users WHERE username='admin'"))
            if cur.fetchone() is None:
                self.add_user("admin", "admin123", "admin")

    def add_user(self, username: str, password: str, role: str = "user")-> None:
        """
        Adds a new user to the 'users' table.

        Args:
            username (str): The username for the new user.
            password (str): The plaintext password to be hashed.
            role (str, optional): The user's role ('admin' or 'user'). Defaults to 'user'.

        Raises:
            ValueError: If the username already exists in the database.
        """
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            with engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO users (username, password_hash, role) VALUES (:u, :p, :r)"),
                    {"u": username, "p": password_hash, "r": role}
                )
        except IntegrityError:
            raise ValueError("Username already exists.")

    def validate_user(self, username: str, password: str) -> tuple[str, str] | None:
        """
        Validates a user's credentials.

        Args:
            username (str): The username to validate.
            password (str): The plaintext password to check.

        Returns:
            tuple: (username, role) if valid; otherwise None.
        """
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT username, role FROM users WHERE username=:u AND password_hash=:p"),
                {"u": username, "p": password_hash}
            ).fetchone()
            return tuple(row) if row else None

    # ---------------- MATERIAL HOPPERS ----------------
    def load_materials_from_yml(self) -> list[str]:
        """
        Loads the materials list from the `materials.yml` configuration file.

        The YAML file should have the structure:
            materials:
              - Material1
              - Material2
              - ...

        Returns:
            list: A list of material names.

        Raises:
            FileNotFoundError: If the YAML file is missing.
            ValueError: If the file structure is invalid.
        """
        if not os.path.exists(MATERIALS_FILE):
            raise FileNotFoundError(f"Missing YAML file: {MATERIALS_FILE}")

        with open(MATERIALS_FILE, "r") as f:
            data = yaml.safe_load(f)
            materials = data.get("materials", [])
            if not isinstance(materials, list):
                raise ValueError("Invalid format: 'materials' must be a list in YAML file.")
            return materials

    def create_material_hoppers_table(self) -> None:

        """
        Creates the 'material_hoppers' table if it does not exist.

        The table includes:
            - material (TEXT, primary key part)
            - hopper (TEXT, primary key part)

        If no hopper exists for a material, assigns a default 'HOPPER_0_ACT'.
        """
        with engine.begin() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS material_hoppers (
                    material TEXT NOT NULL,
                    hopper TEXT NOT NULL,
                    PRIMARY KEY (material, hopper)
                )
            '''))

            for material in self.materials:
                cur = conn.execute(
                    text("SELECT hopper FROM material_hoppers WHERE material=:m"),
                    {"m": material}
                ).fetchall()
                if not cur:
                    conn.execute(
                        text("INSERT INTO material_hoppers (material, hopper) VALUES (:m, :h)"),
                        {"m": material, "h": "HOPPER_0_ACT"}
                    )

    def get_material_hoppers(self)-> dict[str, list[str]]:
        """
        Fetches all material-to-hopper mappings from the database.

        Returns:
            dict: A dictionary where keys are materials and values are lists of hoppers.
        """
        with engine.begin() as conn:
            rows = conn.execute(text("SELECT material, hopper FROM material_hoppers")).fetchall()

        result = defaultdict(list)
        for material, hopper in rows:
            result[material].append(hopper)
        return dict(result)

    def update_material_hoppers(self, material: str, hopper_numbers: list[int]):
        """
        Updates the hopper assignments for a given material.

        Args:
            material (str): The material name to update.
            hopper_numbers (list[int]): List of hopper numbers to assign.

        Behavior:
            - Replaces old hopper mappings for the given material.
            - Ensures a hopper is not assigned to multiple materials.
            - If a material loses all hoppers, assigns a default 'HOPPER_0_ACT'.
        """
        new_hoppers = [f"HOPPER_{n}_ACT" for n in hopper_numbers]

        with engine.begin() as conn:
            all_rows = conn.execute(text("SELECT material, hopper FROM material_hoppers")).fetchall()
            hopper_to_material = {h: m for m, h in all_rows}

            for hopper in new_hoppers:
                if hopper in hopper_to_material:
                    old_material = hopper_to_material[hopper]
                    if old_material != material:
                        conn.execute(
                            text("DELETE FROM material_hoppers WHERE material=:m AND hopper=:h"),
                            {"m": old_material, "h": hopper}
                        )
                        left = conn.execute(
                            text("SELECT 1 FROM material_hoppers WHERE material=:m LIMIT 1"),
                            {"m": old_material}
                        ).fetchone()
                        if not left:
                            conn.execute(
                                text("INSERT INTO material_hoppers (material, hopper) VALUES (:m, :h)"),
                                {"m": old_material, "h": "HOPPER_0_ACT"}
                            )

            conn.execute(text("DELETE FROM material_hoppers WHERE material=:m"), {"m": material})
            for hopper in new_hoppers:
                conn.execute(
                    text("INSERT INTO material_hoppers (material, hopper) VALUES (:m, :h)"),
                    {"m": material, "h": hopper}
                )
