# import os
# import hashlib
# import yaml
# from sqlalchemy import create_engine, text
# from sqlalchemy.exc import IntegrityError
# from dotenv import load_dotenv

# # ------------------------------------------------------------
# #  Load environment variables
# # ------------------------------------------------------------
# load_dotenv()

# DATABASE_URL = os.getenv("DATABASE_URL")
# if not DATABASE_URL:
#     raise ValueError("❌ Missing DATABASE_URL environment variable. Please set it in your .env file.")

# # Create SQLAlchemy engine
# engine = create_engine(DATABASE_URL, future=True)

# # ------------------------------------------------------------
# #  Path Configuration
# # ------------------------------------------------------------
# # Dynamically locate the YAML file regardless of where Streamlit is launched
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(BASE_DIR)
# MATERIALS_FILE = os.path.normpath(os.path.join(PROJECT_ROOT, "config", "materials.yml"))



# class Database:
#     """
#     A PostgreSQL (Neon) database handler for managing users and hopper-to-material mappings.
#     """

#     def __init__(self) -> None:
#         """Initializes the database by loading configuration and creating necessary tables."""
#         # Load hoppers + materials correctly (fix order)
#         try:
#             self.hoppers, self.materials = self.load_materials_from_yml()
#         except Exception as e:
#             self.materials = []
#             self.hoppers = []

#         # Create required tables
#         self.create_users_table()
#         self.create_material_hoppers_table()

#     # ------------------------------------------------------------
#     #  USERS TABLE
#     # ------------------------------------------------------------
#     def create_users_table(self) -> None:
#         with engine.begin() as conn:
#             conn.execute(text('''
#                 CREATE TABLE IF NOT EXISTS users (
#                     username TEXT PRIMARY KEY,
#                     password_hash TEXT,
#                     role TEXT CHECK (role IN ('admin', 'user')) NOT NULL
#                 )
#             '''))

#         # Ensure default admin exists
#         with engine.begin() as conn:
#             cur = conn.execute(text("SELECT 1 FROM users WHERE username='admin'"))
#             if cur.fetchone() is None:
#                 self.add_user("admin", "admin123", "admin")

#     def add_user(self, username: str, password: str, role: str = "user") -> None:
#         password_hash = hashlib.sha256(password.encode()).hexdigest()
#         try:
#             with engine.begin() as conn:
#                 conn.execute(
#                     text("INSERT INTO users (username, password_hash, role) VALUES (:u, :p, :r)"),
#                     {"u": username, "p": password_hash, "r": role}
#                 )
#         except IntegrityError:
#             raise ValueError("Username already exists.")

#     def validate_user(self, username: str, password: str) -> tuple[str, str] | None:
#         password_hash = hashlib.sha256(password.encode()).hexdigest()
#         with engine.begin() as conn:
#             row = conn.execute(
#                 text("SELECT username, role FROM users WHERE username=:u AND password_hash=:p"),
#                 {"u": username, "p": password_hash}
#             ).fetchone()
#             return tuple(row) if row else None

#     # ------------------------------------------------------------
#     #  LOAD MATERIALS FROM YAML
#     # ------------------------------------------------------------
#     def load_materials_from_yml(self) -> tuple[list[str], list[str]]:
#         """Loads hoppers and materials lists from the materials.yml configuration file."""
#         if not os.path.exists(MATERIALS_FILE):
#             raise FileNotFoundError(f" Missing YAML file: {MATERIALS_FILE}")

#         with open(MATERIALS_FILE, "r", encoding="utf-8-sig") as f:  # <-- important fix
#             data = yaml.safe_load(f)

#         # Normalize keys (remove BOMs, spaces, etc.)
#         data = {k.strip(): v for k, v in data.items()} if isinstance(data, dict) else {}



#         hoppers = data.get("hoppers", [])
#         materials = data.get("materials", [])

#         if not isinstance(hoppers, list):
#             raise ValueError("Invalid format: 'hoppers' must be a list in YAML file.")
#         if not isinstance(materials, list):
#             raise ValueError("Invalid format: 'materials' must be a list in YAML file.")

#         return hoppers, materials

#     # ------------------------------------------------------------
#     #  HOPPER ↔ MATERIALS TABLE
#     # ------------------------------------------------------------
#     def create_material_hoppers_table(self) -> None:
#         """Ensures the 'hopper_materials' table exists and syncs new hoppers from YAML."""
#         with engine.begin() as conn:
#             # Create table if missing
#             conn.execute(text('''
#                 CREATE TABLE IF NOT EXISTS hopper_materials (
#                     hopper TEXT PRIMARY KEY,
#                     material TEXT
#                 )
#             '''))

#             # Fetch existing hoppers in DB
#             db_hoppers = {
#                 row[0] for row in conn.execute(text("SELECT hopper FROM hopper_materials")).fetchall()
#             }

#             # Add missing ones from YAML
#             missing_hoppers = [h for h in self.hoppers if h not in db_hoppers]

#             if missing_hoppers:

#                 for hopper in missing_hoppers:
#                     conn.execute(
#                         text("INSERT INTO hopper_materials (hopper, material) VALUES (:h, :m)"),
#                         {"h": hopper, "m": "UNASSIGNED"}
#                     )


#     def get_hopper_materials(self) -> dict[str, str]:
#         """Fetches the mapping of hopper → material."""
#         with engine.begin() as conn:
#             rows = conn.execute(text("SELECT hopper, material FROM hopper_materials")).fetchall()
#         return {hopper: material for hopper, material in rows}

#     def update_hopper_material(self, hopper: str, material: str) -> None:
#         """Updates the material assigned to a given hopper."""
#         if hopper not in self.hoppers:
#             raise ValueError(f"Invalid hopper '{hopper}'. Must exist in materials.yml.")
#         if material not in self.materials and material != "UNASSIGNED":
#             raise ValueError(f"Invalid material '{material}'. Must exist in materials.yml or be 'UNASSIGNED'.")

#         with engine.begin() as conn:
#             conn.execute(
#                 text("UPDATE hopper_materials SET material=:m WHERE hopper=:h"),
#                 {"h": hopper, "m": material}
#             )



import os
import hashlib
import yaml
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv

# ------------------------------------------------------------
#  Load Environment Variables
# ------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ Missing DATABASE_URL environment variable. Please set it in your .env file.")

# Create SQLAlchemy engine (connection pooling, future mode)
engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

# ------------------------------------------------------------
#  Path Configuration
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MATERIALS_FILE = os.path.join(PROJECT_ROOT, "config", "materials.yml")


class Database:
    """
    PostgreSQL database handler for:
    - User management
    - Hopper ↔ Material mapping
    - Hopper ↔ Material history (time-based)
    """

    def __init__(self) -> None:
        """Initialize configuration and ensure required tables exist."""
        self.hoppers, self.materials = self._safe_load_materials()
        self._init_tables()

    # ============================================================
    #  Initialization Helpers
    # ============================================================
    def _init_tables(self) -> None:
        """Ensures required tables exist and are seeded."""
        self._create_users_table()
        self._create_hopper_materials_table()
        self._create_hopper_material_history_table()

    def _safe_load_materials(self) -> tuple[list[str], list[str]]:
        """Safely loads hoppers and materials from YAML configuration."""
        try:
            return self._load_materials_from_yaml()
        except Exception as e:
            print(f"⚠️ Warning: Failed to load materials.yml ({e})")
            return [], []

    # ============================================================
    #  USER MANAGEMENT
    # ============================================================
    def _create_users_table(self) -> None:
        """Creates 'users' table if not exists and ensures default admin."""
        #  Create table (committed first)
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS public.users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT CHECK (role IN ('admin', 'user')) NOT NULL
                )
            """))

        #  Seed admin user in a separate committed transaction
        with engine.begin() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM public.users WHERE username = 'admin'")
            ).fetchone()

            if not exists:
                password_hash = hashlib.sha256("admin123".encode()).hexdigest()
                conn.execute(text("""
                    INSERT INTO public.users (username, password_hash, role)
                    VALUES ('admin', :p, 'admin')
                """), {"p": password_hash})


    def add_user(self, username: str, password: str, role: str = "user") -> None:
        """Adds a new user with hashed password."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO users (username, password_hash, role)
                    VALUES (:u, :p, :r)
                """), {"u": username, "p": password_hash, "r": role})
        except IntegrityError:
            raise ValueError(f"Username '{username}' already exists.")

    def validate_user(self, username: str, password: str) -> tuple[str, str] | None:
        """Validates credentials and returns (username, role) if correct."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT username, role
                FROM users
                WHERE username = :u AND password_hash = :p
            """), {"u": username, "p": password_hash}).fetchone()
            return tuple(row) if row else None

    # ============================================================
    #  YAML CONFIG LOADING
    # ============================================================
    def _load_materials_from_yaml(self) -> tuple[list[str], list[str]]:
        """Loads hoppers and materials from the YAML configuration file."""
        if not os.path.exists(MATERIALS_FILE):
            raise FileNotFoundError(f"Missing configuration: {MATERIALS_FILE}")

        with open(MATERIALS_FILE, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("Invalid YAML structure: expected a dictionary at root.")

        hoppers = data.get("hoppers", [])
        materials = data.get("materials", [])

        if not isinstance(hoppers, list):
            raise ValueError("'hoppers' must be a list in materials.yml.")
        if not isinstance(materials, list):
            raise ValueError("'materials' must be a list in materials.yml.")

        return hoppers, materials

    # ============================================================
    #  HOPPER ↔ MATERIAL SNAPSHOT
    # ============================================================
    def _create_hopper_materials_table(self) -> None:
        """Ensures 'hopper_materials' table exists and syncs YAML hoppers."""
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hopper_materials (
                    hopper TEXT PRIMARY KEY,
                    material TEXT DEFAULT 'UNASSIGNED'
                )
            """))

            # Add missing hoppers
            db_hoppers = {
                row[0] for row in conn.execute(text("SELECT hopper FROM hopper_materials"))
            }
            missing = [h for h in self.hoppers if h not in db_hoppers]
            if missing:
                conn.execute(
                    text("""
                        INSERT INTO hopper_materials (hopper, material)
                        VALUES (:h, 'UNASSIGNED')
                    """),
                    [{"h": hopper} for hopper in missing]
                )

    def get_hopper_materials(self) -> dict[str, str]:
        """Returns {hopper: material} mapping."""
        with engine.begin() as conn:
            rows = conn.execute(
                text("SELECT hopper, material FROM hopper_materials ORDER BY hopper")
            ).fetchall()
        return dict(rows)

    def update_hopper_material(self, hopper: str, material: str) -> None:
        """Updates the current material snapshot."""
        if hopper not in self.hoppers:
            raise ValueError(f"Invalid hopper '{hopper}'. Must exist in materials.yml.")
        if material not in self.materials and material != "UNASSIGNED":
            raise ValueError(f"Invalid material '{material}'.")

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE hopper_materials
                SET material = :m
                WHERE hopper = :h
            """), {"h": hopper, "m": material})

    # ============================================================
    #  TIME-BASED HISTORY MANAGEMENT (with modifier)
    # ============================================================
    def _create_hopper_material_history_table(self) -> None:
        """Creates hopper_material_history table if missing."""
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hopper_material_history (
                    id SERIAL PRIMARY KEY,
                    hopper TEXT NOT NULL,
                    material TEXT NOT NULL,
                    valid_from TIMESTAMP NOT NULL,
                    valid_upto TIMESTAMP,
                    modifier TEXT NOT NULL DEFAULT 'system',
                    FOREIGN KEY (hopper) REFERENCES hopper_materials (hopper)
                )
            """))

    def update_hopper_material_with_time(
        self,
        hopper: str,
        material: str,
        from_time: datetime,
        modifier: str,
    ) -> None:
        """
        Updates hopper material history with modifier tracking.
        Closes previous record (valid_upto = from_time - 1 second)
        and inserts a new record (valid_from = from_time, valid_upto = NULL).
        """
        if hopper not in self.hoppers:
            raise ValueError(f"Invalid hopper '{hopper}'.")
        if material not in self.materials and material != "UNASSIGNED":
            raise ValueError(f"Invalid material '{material}'.")

        # Update the current snapshot table
        self.update_hopper_material(hopper, material)

        with engine.begin() as conn:
            # Ensure modifier column exists
            col_check = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'hopper_material_history' AND column_name = 'modifier'
            """)).fetchone()
            if not col_check:
                conn.execute(text("ALTER TABLE hopper_material_history ADD COLUMN modifier TEXT DEFAULT 'system';"))

            # Close previous active record if any
            conn.execute(text("""
                UPDATE hopper_material_history
                SET valid_upto = :prev_upto
                WHERE hopper = :hopper AND valid_upto IS NULL
            """), {
                "hopper": hopper,
                "prev_upto": from_time - timedelta(seconds=1)
            })

            # Insert new record
            conn.execute(text("""
                INSERT INTO hopper_material_history (hopper, material, valid_from, valid_upto, modifier)
                VALUES (:hopper, :material, :valid_from, NULL, :modifier)
            """), {
                "hopper": hopper,
                "material": material,
                "valid_from": from_time,
                "modifier": modifier,
            })

    def get_hopper_material_at(self, hopper: str, timestamp: datetime) -> str | None:
        """Returns the material assigned to a hopper at a specific timestamp."""
        with engine.begin() as conn:
            row = conn.execute(
                text("""
                    SELECT material
                    FROM hopper_material_history
                    WHERE hopper = :h
                      AND valid_from <= :ts
                      AND (valid_upto IS NULL OR valid_upto >= :ts)
                    ORDER BY valid_from DESC
                    LIMIT 1
                """),
                {"h": hopper, "ts": timestamp}
            ).fetchone()
            return row[0] if row else None


