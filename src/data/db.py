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
        self.engine = engine   # ✅ STORE ENGINE HERE

        self.hoppers, self.materials = self._safe_load_materials()
        self.burden_fields = self._safe_load_burden_fields()

        self._create_users_table()
        self._create_hopper_materials_table()
        self._create_hopper_material_history_table()
        self._create_burden_distribution_history_table()


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
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS public.users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT CHECK (role IN ('admin', 'user')) NOT NULL
                )
            """))

        #  Seed admin user in a separate committed transaction
        with self.engine.begin() as conn:
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
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO users (username, password_hash, role)
                    VALUES (:u, :p, :r)
                """), {"u": username, "p": password_hash, "r": role})
        except IntegrityError:
            raise ValueError(f"Username '{username}' already exists.")

    def validate_user(self, username: str, password: str) -> tuple[str, str] | None:
        """Validates credentials and returns (username, role) if correct."""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        with self.engine.begin() as conn:
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
        with self.engine.begin() as conn:
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
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT hopper, material FROM hopper_materials ORDER BY hopper")
            ).fetchall()
        return dict(rows)
    def get_hopper_material_history(self):
        """
        Return full hopper → material history with timestamps.
        """
        query = text("""
            SELECT 
                id,
                hopper,
                material,
                valid_from,
                valid_upto,
                modifier,
                ip_address
            FROM hopper_material_history
            ORDER BY 
                hopper,
                valid_from DESC
        """)

        with self.engine.begin() as conn:
            rows = conn.execute(query).fetchall()

        history = []
        for row in rows:
            history.append({
                "id": row.id,
                "hopper": row.hopper,
                "material": row.material,
                "valid_from": row.valid_from,
                "valid_upto": row.valid_upto,
                "modifier": row.modifier,
                "ip_address": row.ip_address
            })

        return history

    def update_hopper_material(self, hopper: str, material: str) -> None:
        """Updates the current material snapshot."""
        if hopper not in self.hoppers:
            raise ValueError(f"Invalid hopper '{hopper}'. Must exist in materials.yml.")
        if material not in self.materials and material != "UNASSIGNED":
            raise ValueError(f"Invalid material '{material}'.")

        with self.engine.begin() as conn:
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
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hopper_material_history (
                    id SERIAL PRIMARY KEY,
                    hopper TEXT NOT NULL,
                    material TEXT NOT NULL,
                    valid_from TIMESTAMP NOT NULL,
                    valid_upto TIMESTAMP,
                    modifier TEXT NOT NULL DEFAULT 'system',
                    ip_address TEXT, 
                    FOREIGN KEY (hopper) REFERENCES hopper_materials (hopper)
                )
            """))

    def update_hopper_material_with_time(
        self,
        hopper: str,
        material: str,
        from_time: datetime,
        modifier: str,
        ip_address: str,
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

        with self.engine.begin() as conn:
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
                INSERT INTO hopper_material_history (hopper, material, valid_from, valid_upto, modifier,ip_address)
                VALUES (:hopper, :material, :valid_from, NULL, :modifier, :ip)
            """), {
                "hopper": hopper,
                "material": material,
                "valid_from": from_time,
                "modifier": modifier,
                "ip": ip_address,
            })

    def get_hopper_material_at(self, hopper: str, timestamp: datetime) -> str | None:
        """Returns the material assigned to a hopper at a specific timestamp."""
        with self.engine.begin() as conn:
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
    
    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """
        Deletes hopper_material_history records by ID.
        """
        if not record_ids:
            return

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    DELETE FROM hopper_material_history
                    WHERE id = ANY(:ids)
                """),
                {"ids": record_ids}
            )


    # ============================================================
    #  LOAD BURDEN FIELDS FROM YAML
    # ============================================================
    def _safe_load_burden_fields(self):
        """Safely loads burden distribution field names from materials.yml."""
        try:
            return self._load_burden_fields_from_yaml()
        except Exception as e:
            print(f"⚠️ Warning: Failed to load burden_fields from YAML: {e}")
            return []


    def _load_burden_fields_from_yaml(self):
        if not os.path.exists(MATERIALS_FILE):
            raise FileNotFoundError(f"Missing configuration file: {MATERIALS_FILE}")

        with open(MATERIALS_FILE, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}

        fields = data.get("burden_fields", [])
        if not isinstance(fields, list):
            raise ValueError("'burden_fields' must be a list in materials.yml")

        return fields



    # --------------------------------------------------------
    # Create burden distribution history table
    # --------------------------------------------------------
    def _create_burden_distribution_history_table(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS burden_distribution_history (
                    id SERIAL PRIMARY KEY,
                    field_name TEXT NOT NULL,
                    field_value_float DOUBLE PRECISION,
                    field_value_text TEXT,
                    valid_from TIMESTAMPTZ NOT NULL,
                    valid_upto TIMESTAMPTZ,
                    modifier TEXT DEFAULT 'system',
                    ip_address TEXT
                );
            """))

            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_burden_active_record
                ON burden_distribution_history (field_name, valid_upto);
            """))



    # --------------------------------------------------------
    # Update (SCD Type-2)
    # --------------------------------------------------------
    def update_burden_field(self, field_name, value, valid_from, modifier="system", ip=""):

        # Pattern fields use TEXT values, others use FLOAT
        is_text_field = field_name in [
            "COKE_CHARGE_PATTERN",
            "NON_COKE_CHARGE_PATTERN"
        ]

        
        with self.engine.begin() as conn:
            conn.execute(text("""
                UPDATE burden_distribution_history
                SET valid_upto = :end_time
                WHERE field_name = :f AND valid_upto IS NULL
            """), {
                "f": field_name,
                "end_time": valid_from - timedelta(seconds=1)
            })

            if is_text_field:
                conn.execute(text("""
                    INSERT INTO burden_distribution_history
                        (field_name, field_value_text, valid_from, valid_upto, modifier, ip_address)
                    VALUES (:f, :v_text, :start, NULL, :m, :ip)
                """), {
                    "f": field_name,
                    "v_text": str(value),
                    "start": valid_from,
                    "m": modifier,
                    "ip": ip
                })
            else:
                conn.execute(text("""
                    INSERT INTO burden_distribution_history
                        (field_name, field_value_float, valid_from, valid_upto, modifier, ip_address)
                    VALUES (:f, :v_float, :start, NULL, :m, :ip)
                """), {
                    "f": field_name,
                    "v_float": float(value),
                    "start": valid_from,
                    "m": modifier,
                    "ip": ip
                })

    # --------------------------------------------------------
    # Bulk update from DataFrame row (timestamp-indexed)
    # --------------------------------------------------------
    def update_burden_row(self, df_row, timestamp, modifier="system", ip=""):
        for field, value in df_row.items():
            if field in self.burden_fields and value is not None:
                self.update_burden_field(field, value, timestamp, modifier, ip)



    # --------------------------------------------------------
    # Read history
    # --------------------------------------------------------
    def get_burden_history(self):
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT id, field_name,
                    field_value_float, field_value_text,
                    valid_from, valid_upto, modifier, ip_address
                FROM burden_distribution_history
                ORDER BY field_name, valid_from DESC
            """)).fetchall()


        output = []
        for r in rows:
            value = r.field_value_text if r.field_value_text is not None else r.field_value_float

            output.append({
                "id": r.id,
                "field_name": r.field_name,
                "value": value,
                "valid_from": r.valid_from,
                "valid_upto": r.valid_upto,
                "modifier": r.modifier,
                "ip_address": r.ip_address
            })

        return output


    # --------------------------------------------------------
    # Query field value at a specific time
    # --------------------------------------------------------
    def get_all_current_burden_values(self, ts):
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT ON (field_name)
                    field_name,
                    field_value_float,
                    field_value_text
                FROM burden_distribution_history
                WHERE valid_from <= :ts
                AND (valid_upto IS NULL OR valid_upto >= :ts)
                ORDER BY field_name, valid_from DESC
            """), {"ts": ts}).fetchall()

        return {
            r.field_name: r.field_value_text if r.field_value_text is not None else r.field_value_float
            for r in rows
        }




