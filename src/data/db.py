"""PostgreSQL database layer for the BF2 blast furnace web application.

Manages user authentication, hopper-to-material assignments, and burden
distribution history using a SCD Type-2 pattern (``valid_upto IS NULL``
identifies the current row).
"""

import hashlib
import os
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

# ------------------------------------------------------------
# Load Environment Variables
# ------------------------------------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("❌ Missing DATABASE_URL environment variable.")

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MATERIALS_FILE = os.path.join(PROJECT_ROOT, "config", "materials.yml")


class Database:
    """Thin ORM wrapper around the PostgreSQL schema for BF2 operational data.

    On first instantiation the class ensures all required tables exist and
    seeds the database with initial hopper entries if none are present.

    Attributes:
        engine:        SQLAlchemy engine connected to the configured database.
        hoppers:       List of valid hopper names loaded from ``materials.yml``.
        materials:     List of valid material names loaded from ``materials.yml``.
        burden_fields: List of burden distribution field names from ``materials.yml``.
    """

    def __init__(self) -> None:
        self.engine = engine

        self.hoppers, self.materials = self._safe_load_materials()
        self.burden_fields = self._safe_load_burden_fields()

        self._create_users_table()
        self._create_hopper_material_history_table()
        self._create_burden_distribution_history_table()
        self._seed_hoppers_if_missing()

    # ============================================================
    # USERS
    # ============================================================
    def _create_users_table(self) -> None:
        """Create the ``users`` table and seed the ``admin`` account if absent."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT CHECK (role IN ('admin','supervisor','user')) NOT NULL
                )
            """))

            if not conn.execute(
                text("SELECT 1 FROM users WHERE username='admin'")
            ).fetchone():
                conn.execute(
                    text("""
                    INSERT INTO users VALUES ('admin', :p, 'admin')
                """),
                    {"p": hashlib.sha256("admin123".encode()).hexdigest()},
                )

    def add_user(self, username: str, password: str, role: str = "user") -> None:
        """Add a new user to the database.

        Args:
            username: Unique username string.
            password: Plain-text password (stored as SHA-256 digest).
            role:     One of ``"admin"``, ``"supervisor"``, or ``"user"``.

        Raises:
            ValueError: If *username* already exists.
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text("""
                    INSERT INTO users VALUES (:u,:p,:r)
                """),
                    {
                        "u": username,
                        "p": hashlib.sha256(password.encode()).hexdigest(),
                        "r": role,
                    },
                )
        except IntegrityError:
            raise ValueError("User already exists")

    def validate_user(self, username: str, password: str):
        """Validate login credentials against the stored SHA-256 password hash.

        Args:
            username: Account username.
            password: Plain-text password to verify.

        Returns:
            A row tuple ``(username, role)`` if credentials are valid,
            else ``None``.
        """
        with self.engine.begin() as conn:
            return conn.execute(
                text("""
                SELECT username, role FROM users
                WHERE username=:u AND password_hash=:p
            """),
                {"u": username, "p": hashlib.sha256(password.encode()).hexdigest()},
            ).fetchone()

    # ============================================================
    # YAML
    # ============================================================
    def _safe_load_materials(self) -> tuple:
        """Load hoppers and materials from YAML, returning empty lists on error."""
        try:
            return self._load_materials_from_yaml()
        except Exception:
            return [], []

    def _load_materials_from_yaml(self) -> tuple:
        """Parse ``materials.yml`` and return ``(hoppers, materials)`` lists."""
        with open(MATERIALS_FILE, "r", encoding="utf-8-sig") as f:
            data = yaml.safe_load(f) or {}
        return data.get("hoppers", []), data.get("materials", [])

    # ============================================================
    # HOPPER ↔ MATERIAL HISTORY (ONLY TABLE)
    # ============================================================
    def _create_hopper_material_history_table(self) -> None:
        """Create the ``hopper_material_history`` table with a partial index for active rows."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hopper_material_history (
                    id SERIAL PRIMARY KEY,
                    hopper TEXT NOT NULL,
                    material TEXT NOT NULL,
                    valid_from TIMESTAMP NOT NULL,
                    valid_upto TIMESTAMP,
                    modifier TEXT DEFAULT 'system',
                    ip_address TEXT
                )
            """))

            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_hopper_active
                ON hopper_material_history (hopper)
                WHERE valid_upto IS NULL
            """))

    def _seed_hoppers_if_missing(self) -> None:
        """Insert an ``UNASSIGNED`` entry for any hopper not yet in the history table."""
        now = datetime.utcnow()
        with self.engine.begin() as conn:
            existing = {
                r[0]
                for r in conn.execute(
                    text("SELECT DISTINCT hopper FROM hopper_material_history")
                )
            }
            for hopper in self.hoppers:
                if hopper not in existing:
                    conn.execute(
                        text("""
                        INSERT INTO hopper_material_history
                        (hopper, material, valid_from)
                        VALUES (:h, 'UNASSIGNED', :t)
                    """),
                        {"h": hopper, "t": now},
                    )

    def update_hopper_material_with_time(
        self, hopper: str, material: str, from_time, modifier: str, ip_address: str
    ) -> None:
        """Record a new hopper-to-material assignment using SCD Type-2.

        Closes the previous active row (sets ``valid_upto``) and inserts a new
        row starting from *from_time*.

        Args:
            hopper:     Hopper identifier (must be in ``self.hoppers``).
            material:   Material name (must be in ``self.materials`` or
                        ``"UNASSIGNED"``).
            from_time:  Datetime at which the new assignment takes effect.
            modifier:   Username of the operator making the change.
            ip_address: IP address of the requesting client.

        Raises:
            ValueError: If *hopper* or *material* is invalid.
        """
        if hopper not in self.hoppers:
            raise ValueError("Invalid hopper")
        if material not in self.materials and material != "UNASSIGNED":
            raise ValueError("Invalid material")

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE hopper_material_history
                SET valid_upto=:u
                WHERE hopper=:h AND valid_upto IS NULL
            """),
                {"h": hopper, "u": from_time - timedelta(seconds=1)},
            )

            conn.execute(
                text("""
                INSERT INTO hopper_material_history
                (hopper, material, valid_from, modifier, ip_address)
                VALUES (:h,:m,:t,:mod,:ip)
            """),
                {
                    "h": hopper,
                    "m": material,
                    "t": from_time,
                    "mod": modifier,
                    "ip": ip_address,
                },
            )

    def get_current_hopper_materials(self):
        with self.engine.begin() as conn:
            return dict(conn.execute(text("""
                SELECT hopper, material
                FROM hopper_material_history
                WHERE valid_upto IS NULL
                ORDER BY hopper
            """)).fetchall())

    def get_hopper_material_at(self, hopper: str, ts) -> str | None:
        """Return the material assigned to *hopper* at timestamp *ts*.

        Args:
            hopper: Hopper identifier.
            ts:     Timestamp to query.

        Returns:
            Material name string, or ``None`` if no record matches.
        """
        with self.engine.begin() as conn:
            r = conn.execute(
                text("""
                SELECT material FROM hopper_material_history
                WHERE hopper=:h AND valid_from<=:t
                AND (valid_upto IS NULL OR valid_upto>=:t)
                ORDER BY valid_from DESC LIMIT 1
            """),
                {"h": hopper, "t": ts},
            ).fetchone()
            return r[0] if r else None

    def get_current_hopper_materials(self) -> dict[str, str]:
        """
        Returns the CURRENT hopper->material mapping.
        Current = valid_upto IS NULL
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT hopper, material
                FROM hopper_material_history
                WHERE valid_upto IS NULL
                ORDER BY hopper
            """)).fetchall()
        return dict(rows)

    def get_hopper_material_history(self) -> list[dict]:
        """
        Returns FULL hopper->material history (all rows).
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text("""
                SELECT
                    id,
                    hopper,
                    material,
                    valid_from,
                    valid_upto,
                    modifier,
                    ip_address
                FROM hopper_material_history
                ORDER BY hopper, valid_from DESC
            """)).fetchall()

        return [
            {
                "id": r.id,
                "hopper": r.hopper,
                "material": r.material,
                "valid_from": r.valid_from,
                "valid_upto": r.valid_upto,
                "modifier": r.modifier,
                "ip_address": r.ip_address,
            }
            for r in rows
        ]

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """
        Deletes hopper_material_history records by ID.
        """
        if not record_ids:
            return

        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM hopper_material_history WHERE id = ANY(:ids)"),
                {"ids": record_ids},
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
    def update_burden_field(
        self, field_name: str, value, valid_from, modifier: str = "system", ip: str = ""
    ) -> None:
        """Record a new burden distribution field value using SCD Type-2.

        Text fields (``*_PATTERN`` and ``BURDEN_CHANGING_PURPOSE``) are stored
        in ``field_value_text``; all other fields use ``field_value_float``.

        Args:
            field_name:  Name of the burden distribution field.
            value:       New value (numeric or string depending on field type).
            valid_from:  Datetime at which the value becomes effective.
            modifier:    Username of the person making the change.
            ip:          Client IP address.
        """
        # Pattern fields use TEXT values, others use FLOAT
        is_text_field = field_name in [
            "COKE_CHARGE_PATTERN",
            "NON_COKE_CHARGE_PATTERN",
            "BURDEN_CHANGING_PURPOSE",
        ]

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                UPDATE burden_distribution_history
                SET valid_upto = :end_time
                WHERE field_name = :f AND valid_upto IS NULL
            """),
                {"f": field_name, "end_time": valid_from - timedelta(seconds=1)},
            )

            if is_text_field:
                conn.execute(
                    text("""
                    INSERT INTO burden_distribution_history
                        (field_name, field_value_text, valid_from, valid_upto, modifier, ip_address)
                    VALUES (:f, :v_text, :start, NULL, :m, :ip)
                """),
                    {
                        "f": field_name,
                        "v_text": str(value),
                        "start": valid_from,
                        "m": modifier,
                        "ip": ip,
                    },
                )
            else:
                conn.execute(
                    text("""
                    INSERT INTO burden_distribution_history
                        (field_name, field_value_float, valid_from, valid_upto, modifier, ip_address)
                    VALUES (:f, :v_float, :start, NULL, :m, :ip)
                """),
                    {
                        "f": field_name,
                        "v_float": float(value),
                        "start": valid_from,
                        "m": modifier,
                        "ip": ip,
                    },
                )

    # --------------------------------------------------------
    # Bulk update from DataFrame row (timestamp-indexed)
    # --------------------------------------------------------
    def update_burden_row(
        self, df_row, timestamp, modifier: str = "system", ip: str = ""
    ) -> None:
        """Bulk-update burden distribution fields from a single DataFrame row.

        Iterates over the row and calls :meth:`update_burden_field` for every
        column that is listed in ``self.burden_fields`` and has a non-``None``
        value.

        Args:
            df_row:    A pandas Series or dict mapping field names to values.
            timestamp: Effective datetime for the update.
            modifier:  Username of the operator.
            ip:        Client IP address.
        """
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
            value = (
                r.field_value_text
                if r.field_value_text is not None
                else r.field_value_float
            )

            output.append(
                {
                    "id": r.id,
                    "field_name": r.field_name,
                    "value": value,
                    "valid_from": r.valid_from,
                    "valid_upto": r.valid_upto,
                    "modifier": r.modifier,
                    "ip_address": r.ip_address,
                }
            )

        return output

    # --------------------------------------------------------
    # Query field value at a specific time
    # --------------------------------------------------------
    def get_all_current_burden_values(self, ts):
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                SELECT DISTINCT ON (field_name)
                    field_name,
                    field_value_float,
                    field_value_text
                FROM burden_distribution_history
                WHERE valid_from <= :ts
                AND (valid_upto IS NULL OR valid_upto >= :ts)
                ORDER BY field_name, valid_from DESC
            """),
                {"ts": ts},
            ).fetchall()

        return {
            r.field_name: (
                r.field_value_text
                if r.field_value_text is not None
                else r.field_value_float
            )
            for r in rows
        }
